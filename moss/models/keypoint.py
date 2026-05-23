"""Keypoint scoring models for task evaluation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class KeypointScore(str, Enum):
    """Score for a single keypoint."""

    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    MISSING = "missing"

    @property
    def numeric(self) -> float:
        return {
            KeypointScore.STRONG: 1.0,
            KeypointScore.ADEQUATE: 0.67,
            KeypointScore.WEAK: 0.33,
            KeypointScore.MISSING: 0.0,
        }[self]


class Keypoint(BaseModel):
    """A single keypoint with its score."""

    name: str
    score: KeypointScore

    def improved_over(self, other: Keypoint) -> bool:
        """Check if this keypoint improved over another."""
        return self.score.numeric > other.score.numeric


class KeypointMatrix(BaseModel):
    """Matrix of keypoint scores per task."""

    tasks: dict[str, list[Keypoint]] = {}

    def aggregate_score(self) -> float:
        """Compute the average numeric score across all tasks and keypoints."""
        all_scores: list[float] = []
        for keypoints in self.tasks.values():
            all_scores.extend(kp.score.numeric for kp in keypoints)
        if not all_scores:
            return 0.0
        return sum(all_scores) / len(all_scores)

    def improved_over(self, baseline: KeypointMatrix) -> bool:
        """Check if this matrix shows improvement over the baseline."""
        return self.aggregate_score() > baseline.aggregate_score()

    def improvement_delta(self, baseline: KeypointMatrix) -> float:
        """Return the delta in aggregate scores."""
        return self.aggregate_score() - baseline.aggregate_score()

    def task_ids(self) -> list[str]:
        return list(self.tasks.keys())
