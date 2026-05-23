"""Stage 2: Root cause analysis to fix specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moss.runner.base import Runner, StageResult


PLAN_PROMPT_TEMPLATE = """\
You are designing a fix for failures in an autonomous agent's source code (harness layer).

## Diagnosis (from Locate stage)

{diagnosis}

## Task

Based on the diagnosis, produce a **fix specification** that:
1. Identifies the exact source files and functions that need to change.
2. For each file, describes the specific logic to add, remove, or modify.
3. Explains the expected effect of each change on the failing tasks.
4. Lists any risks or side effects of the proposed changes.
5. Specifies the order in which changes should be applied.

Be precise: name files, functions, line ranges. Do NOT write code — that is a later stage.
Output as structured markdown.
"""


@dataclass
class PlanStage:
    """Stage 2: Produce a fix specification from diagnosis."""

    runner: Runner

    async def run(
        self,
        diagnosis: str,
        workdir: Path,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        prompt = PLAN_PROMPT_TEMPLATE.format(diagnosis=diagnosis)

        return await self.runner.invoke(
            stage="plan",
            prompt=prompt,
            workdir=workdir,
            context=context or {},
        )
