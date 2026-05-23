"""Stage runner: wraps runner invocation with retry, budgets, and archival."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from moss.runner.base import Runner, StageResult
from moss.state.store import StateStore

logger = logging.getLogger(__name__)


class StageRunner:
    """Invokes a coding agent per stage with retry logic and artifact archival."""

    def __init__(
        self,
        runner: Runner,
        store: StateStore,
        evo_id: str,
        max_retries: int = 2,
    ) -> None:
        self.runner = runner
        self.store = store
        self.evo_id = evo_id
        self.max_retries = max_retries

    async def run_stage(
        self,
        stage_name: str,
        prompt: str,
        workdir: Path,
        iteration: int,
        context: dict[str, Any] | None = None,
        artifact_filename: str | None = None,
    ) -> StageResult:
        """Run a stage with retries and archive the output."""
        last_result: StageResult | None = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Running stage '%s' iter=%d attempt=%d/%d",
                stage_name,
                iteration,
                attempt,
                self.max_retries,
            )

            result = await self.runner.invoke(
                stage=stage_name,
                prompt=prompt,
                workdir=workdir,
                context={**(context or {}), "attempt": attempt},
            )

            last_result = result

            if result.success:
                break

            logger.warning(
                "Stage '%s' attempt %d failed: %s",
                stage_name,
                attempt,
                result.error,
            )

        if last_result is None:
            last_result = StageResult(
                output="", success=False, error="No attempts made"
            )

        # Archive artifact
        if artifact_filename:
            self.store.save_stage_artifact(
                self.evo_id, iteration, artifact_filename, last_result.output
            )

        return last_result
