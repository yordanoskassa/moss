"""Abstract runner interface for coding agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    """Result from a single stage invocation."""

    output: str
    success: bool
    exit_code: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Runner(ABC):
    """Abstract base class for coding-agent runners."""

    @abstractmethod
    async def invoke(
        self,
        stage: str,
        prompt: str,
        workdir: Path,
        context: dict[str, Any],
    ) -> StageResult:
        """Invoke the coding agent for a specific pipeline stage.

        The runner wraps the read/write/shell/build capabilities — the coding
        agent handles those internally. MOSS provides the stage prompt and
        working directory; the coding agent does the actual file manipulation.

        Args:
            stage: Pipeline stage name (e.g., "locate", "plan", "implement").
            prompt: Stage-specific prompt text.
            workdir: Working directory for the coding agent (inner substrate repo).
            context: Additional context (e.g., previous stage outputs, config).

        Returns:
            StageResult with the agent's output and status.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the runner backend is available."""

    @abstractmethod
    def name(self) -> str:
        """Return the runner's name identifier."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return a dict describing the runner's capabilities."""
