"""Stage 4: Write the fix as a git commit in the inner repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moss.runner.base import Runner, StageResult


IMPLEMENT_PROMPT_TEMPLATE = """\
You are implementing a fix to an autonomous agent's source code based on an approved plan.

## Approved Fix Plan

{plan}

## Task

Implement the changes described in the fix plan:
1. Read the relevant source files to understand current implementation.
2. Apply each change specified in the plan, in the prescribed order.
3. Ensure all changes are consistent with each other.
4. After making changes, create a git commit with a descriptive message that summarizes
   what was changed and why (reference the failure clusters being fixed).

Important rules:
- Make ONLY the changes specified in the plan. Do not refactor unrelated code.
- Preserve existing code style and conventions.
- If a change in the plan is unclear, implement the most conservative interpretation.
- The commit message should start with "moss(evo): " followed by a concise summary.

Output a summary of what was changed, including file paths and a brief description.
"""


@dataclass
class ImplementStage:
    """Stage 4: Implement the fix via the coding agent."""

    runner: Runner

    async def run(
        self,
        plan: str,
        workdir: Path,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        prompt = IMPLEMENT_PROMPT_TEMPLATE.format(plan=plan)

        return await self.runner.invoke(
            stage="implement",
            prompt=prompt,
            workdir=workdir,
            context=context or {},
        )
