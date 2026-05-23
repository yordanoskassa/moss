"""Verdict enum for evolution convergence decisions."""

from enum import Enum


class Verdict(str, Enum):
    """Outcome of a single evolution iteration's verdict stage."""

    CONVERGED = "converged"
    NEED_MORE_WORK = "need_more_work"
    FUNDAMENTAL_LIMIT_MODEL = "fundamental_limit_model"
    FUNDAMENTAL_LIMIT_ARCHITECTURE = "fundamental_limit_architecture"

    @property
    def is_terminal(self) -> bool:
        return self != Verdict.NEED_MORE_WORK
