"""Claude Code subprocess runner."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from moss.models.config import RunnerConfig
from moss.runner.base import Runner, StageResult

logger = logging.getLogger(__name__)


class ClaudeCodeRunner(Runner):
    """Spawns `claude` CLI as a subprocess for each stage invocation."""

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()

    def name(self) -> str:
        return "claude-code"

    def capabilities(self) -> dict[str, Any]:
        return {
            "read_files": True,
            "write_files": True,
            "shell_commands": True,
            "git_operations": True,
        }

    async def invoke(
        self,
        stage: str,
        prompt: str,
        workdir: Path,
        context: dict[str, Any],
    ) -> StageResult:
        """Spawn claude CLI with the stage prompt."""
        cmd = [self.config.command] + self.config.flags
        cmd.extend(["--output-format", "text"])

        # Build the full prompt with context
        full_prompt = self._build_prompt(stage, prompt, context)

        logger.info("Invoking Claude Code for stage '%s' in %s", stage, workdir)
        logger.debug("Prompt length: %d chars", len(full_prompt))

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=full_prompt.encode()),
                timeout=self.config.timeout,
            )

            exit_code = process.returncode or 0
            output = stdout.decode(errors="replace")
            error_output = stderr.decode(errors="replace")

            if exit_code != 0:
                logger.warning(
                    "Claude Code exited with code %d for stage '%s': %s",
                    exit_code,
                    stage,
                    error_output[:500],
                )
                return StageResult(
                    output=output,
                    success=False,
                    exit_code=exit_code,
                    error=error_output or f"Exit code {exit_code}",
                )

            return StageResult(output=output, success=True, exit_code=0)

        except asyncio.TimeoutError:
            logger.error(
                "Claude Code timed out after %ds for stage '%s'",
                self.config.timeout,
                stage,
            )
            if process.returncode is None:
                process.kill()
                await process.wait()
            return StageResult(
                output="",
                success=False,
                exit_code=-1,
                error=f"Timeout after {self.config.timeout}s",
            )
        except FileNotFoundError:
            return StageResult(
                output="",
                success=False,
                exit_code=-1,
                error=f"Command not found: {self.config.command}",
            )

    async def health_check(self) -> bool:
        """Check if the claude CLI is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.command,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except (FileNotFoundError, asyncio.TimeoutError):
            return False

    @staticmethod
    def _build_prompt(stage: str, prompt: str, context: dict[str, Any]) -> str:
        """Construct the full prompt with stage context."""
        parts = [f"# MOSS Evolution — Stage: {stage.upper()}\n"]

        if context:
            parts.append("## Context\n")
            for key, value in context.items():
                if isinstance(value, str) and len(value) > 200:
                    parts.append(f"### {key}\n{value}\n")
                else:
                    parts.append(f"- **{key}**: {value}\n")
            parts.append("")

        parts.append("## Instructions\n")
        parts.append(prompt)

        return "\n".join(parts)
