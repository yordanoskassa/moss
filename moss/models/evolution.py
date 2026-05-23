"""Evolution state tracking models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from moss.models.verdict import Verdict


class EvolutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class EvolutionState(BaseModel):
    """Persistent state for a single evolution run."""

    batch_id: str
    current_iteration: int = 0
    max_iterations: int = 5
    depth: str = "standard"
    verdict: Verdict | None = None
    status: EvolutionStatus = EvolutionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    def advance_iteration(self) -> None:
        self.current_iteration += 1
        self.updated_at = datetime.now(timezone.utc)

    @property
    def is_terminal(self) -> bool:
        if self.status in (
            EvolutionStatus.COMPLETED,
            EvolutionStatus.FAILED,
            EvolutionStatus.STOPPED,
        ):
            return True
        if self.verdict and self.verdict.is_terminal:
            return True
        return self.current_iteration >= self.max_iterations


class EvolutionRun(BaseModel):
    """Metadata for an evolution run directory."""

    evo_id: str
    state: EvolutionState
    directory: str
