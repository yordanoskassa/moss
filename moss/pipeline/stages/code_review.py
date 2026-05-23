"""Stage 5: Diff quality gate (multi-round with Implement)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from moss.runner.base import Runner, StageResult


class CodeReviewDecision(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request-changes"


CODE_REVIEW_PROMPT_TEMPLATE = """\
You are reviewing a code diff against its approved fix plan.

## Approved Fix Plan

{plan}

## Implementation Summary

{implementation}

## Diff

{diff}

{previous_feedback}

## Task

Review the diff against the plan:
1. **Completeness**: Does the diff implement all changes in the plan?
2. **Correctness**: Are the changes logically correct? Will they fix the identified failures?
3. **Safety**: Could any change introduce regressions or break existing functionality?
4. **Style**: Does the code follow the existing codebase conventions?
5. **Scope**: Does the diff contain changes NOT specified in the plan? (Flag these.)

Output your decision as one of:
- APPROVE — diff is ready for testing
- REQUEST-CHANGES — diff needs modifications (provide specific feedback)

Start your response with the decision keyword on its own line.
"""


@dataclass
class CodeReviewStage:
    """Stage 5: Multi-round code review of the implementation diff."""

    runner: Runner
    max_rounds: int = 3

    async def run(
        self,
        plan: str,
        implementation: str,
        diff: str,
        workdir: Path,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        previous_feedback = ""

        for round_num in range(1, self.max_rounds + 1):
            prompt = CODE_REVIEW_PROMPT_TEMPLATE.format(
                plan=plan,
                implementation=implementation,
                diff=diff,
                previous_feedback=(
                    f"\n## Previous Review Feedback (Round {round_num - 1})\n{previous_feedback}"
                    if previous_feedback
                    else ""
                ),
            )

            result = await self.runner.invoke(
                stage="code_review",
                prompt=prompt,
                workdir=workdir,
                context={**(context or {}), "round": round_num},
            )

            if not result.success:
                return result

            decision = self._parse_decision(result.output)

            if decision == CodeReviewDecision.APPROVE:
                result.metadata["decision"] = decision.value
                result.metadata["rounds"] = round_num
                return result

            previous_feedback = result.output
            result.metadata["decision"] = decision.value

        result.metadata["max_rounds_reached"] = True
        return result

    @staticmethod
    def _parse_decision(output: str) -> CodeReviewDecision:
        first_line = output.strip().split("\n")[0].strip().upper()
        if "APPROVE" in first_line:
            return CodeReviewDecision.APPROVE
        return CodeReviewDecision.REQUEST_CHANGES
