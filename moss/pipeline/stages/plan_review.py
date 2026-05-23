"""Stage 3: Quality gate for the fix plan (multi-round)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from moss.runner.base import Runner, StageResult


class PlanReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT_OFF_TARGET = "reject-off-target"
    REJECT_TOO_NARROW = "reject-too-narrow"


PLAN_REVIEW_PROMPT_TEMPLATE = """\
You are reviewing a fix specification for an autonomous agent's source code.

## Original Diagnosis

{diagnosis}

## Proposed Fix Plan

{plan}

{previous_feedback}

## Task

Evaluate the plan against these criteria:
1. **Targeting**: Does the plan address the root causes identified in the diagnosis?
   - If it fixes symptoms but not root causes, output: REJECT-OFF-TARGET
2. **Completeness**: Does the plan cover all failure clusters, or only a subset?
   - If it leaves major clusters unaddressed, output: REJECT-TOO-NARROW
3. **Safety**: Are the proposed changes unlikely to break existing functionality?
4. **Precision**: Are the file/function references specific enough to implement?

Output your decision as one of:
- APPROVE — plan is ready for implementation
- REJECT-OFF-TARGET — plan misidentifies or ignores root causes
- REJECT-TOO-NARROW — plan addresses some but not enough failures

If rejecting, provide specific feedback on what to change.
Start your response with the decision keyword on its own line.
"""


@dataclass
class PlanReviewStage:
    """Stage 3: Multi-round quality gate for the fix plan."""

    runner: Runner
    max_rounds: int = 3

    async def run(
        self,
        diagnosis: str,
        plan: str,
        workdir: Path,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        """Run review, potentially multiple rounds with the Plan stage."""
        previous_feedback = ""

        for round_num in range(1, self.max_rounds + 1):
            prompt = PLAN_REVIEW_PROMPT_TEMPLATE.format(
                diagnosis=diagnosis,
                plan=plan,
                previous_feedback=(
                    f"\n## Previous Review Feedback (Round {round_num - 1})\n{previous_feedback}"
                    if previous_feedback
                    else ""
                ),
            )

            result = await self.runner.invoke(
                stage="plan_review",
                prompt=prompt,
                workdir=workdir,
                context={**(context or {}), "round": round_num},
            )

            if not result.success:
                return result

            decision = self._parse_decision(result.output)

            if decision == PlanReviewDecision.APPROVE:
                result.metadata["decision"] = decision.value
                result.metadata["rounds"] = round_num
                return result

            # For rejections, store feedback for next round
            previous_feedback = result.output
            result.metadata["decision"] = decision.value

        # After max rounds without approval, return last result
        result.metadata["max_rounds_reached"] = True
        return result

    @staticmethod
    def _parse_decision(output: str) -> PlanReviewDecision:
        first_line = output.strip().split("\n")[0].strip().upper()
        if "APPROVE" in first_line:
            return PlanReviewDecision.APPROVE
        if "OFF-TARGET" in first_line or "OFF_TARGET" in first_line:
            return PlanReviewDecision.REJECT_OFF_TARGET
        if "TOO-NARROW" in first_line or "TOO_NARROW" in first_line:
            return PlanReviewDecision.REJECT_TOO_NARROW
        # Default to approve if unclear
        return PlanReviewDecision.APPROVE
