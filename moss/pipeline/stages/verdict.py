"""Stage 7: Convergence decision based on keypoint improvement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moss.models.keypoint import KeypointMatrix
from moss.models.verdict import Verdict
from moss.runner.base import Runner, StageResult


VERDICT_PROMPT_TEMPLATE = """\
You are deciding whether an evolution iteration has converged.

## Baseline Keypoint Scores

{baseline_json}

Baseline aggregate score: {baseline_score:.3f}

## Current Iteration Keypoint Scores

{current_json}

Current aggregate score: {current_score:.3f}

## Score Delta

{delta:+.3f} ({delta_pct:+.1f}%)

## Consecutive No-Improvement Iterations

{plateau_count} (plateau threshold: {plateau_threshold})

## Task

Decide the evolution outcome. Choose exactly one:

1. **CONVERGED** — The current scores show sufficient improvement AND further iterations
   are unlikely to yield meaningful additional gains.
2. **NEED_MORE_WORK** — There is improvement but significant room for more. Further
   iterations are likely to yield gains.
3. **FUNDAMENTAL_LIMIT_MODEL** — The failures stem from inherent limitations of the
   underlying language model (e.g., reasoning depth, context window) that no harness
   change can fix.
4. **FUNDAMENTAL_LIMIT_ARCHITECTURE** — The failures require architectural changes
   beyond the scope of source-level rewriting (e.g., the agent framework cannot
   support the needed interaction pattern).

Rules:
- If the plateau threshold is reached (consecutive no-improvement >= threshold),
  you MUST output CONVERGED regardless of score level.
- A delta < 0.01 counts as no improvement for plateau tracking.

Output your decision as a single keyword on the first line, followed by reasoning.
"""


@dataclass
class VerdictStage:
    """Stage 7: Compare keypoint matrices and emit a convergence verdict."""

    runner: Runner
    plateau_threshold: int = 3

    async def run(
        self,
        baseline: KeypointMatrix,
        current: KeypointMatrix,
        plateau_count: int,
        workdir: Path,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        # Force convergence if plateau threshold reached
        if plateau_count >= self.plateau_threshold:
            return StageResult(
                output=f"CONVERGED\n\nPlateau threshold ({self.plateau_threshold}) reached "
                f"after {plateau_count} consecutive non-improving iterations.",
                success=True,
                metadata={"verdict": Verdict.CONVERGED.value, "forced_plateau": True},
            )

        baseline_score = baseline.aggregate_score()
        current_score = current.aggregate_score()
        delta = current_score - baseline_score
        delta_pct = (delta / baseline_score * 100) if baseline_score > 0 else 0

        prompt = VERDICT_PROMPT_TEMPLATE.format(
            baseline_json=baseline.model_dump_json(indent=2),
            baseline_score=baseline_score,
            current_json=current.model_dump_json(indent=2),
            current_score=current_score,
            delta=delta,
            delta_pct=delta_pct,
            plateau_count=plateau_count,
            plateau_threshold=self.plateau_threshold,
        )

        result = await self.runner.invoke(
            stage="verdict",
            prompt=prompt,
            workdir=workdir,
            context=context or {},
        )

        if result.success:
            verdict = self._parse_verdict(result.output)
            result.metadata["verdict"] = verdict.value
            result.metadata["baseline_score"] = baseline_score
            result.metadata["current_score"] = current_score
            result.metadata["delta"] = delta

        return result

    @staticmethod
    def _parse_verdict(output: str) -> Verdict:
        first_line = output.strip().split("\n")[0].strip().upper()
        for v in Verdict:
            if v.value.upper().replace("_", " ") in first_line or v.value.upper() in first_line:
                return v
        # Default to need more work if parsing fails
        return Verdict.NEED_MORE_WORK
