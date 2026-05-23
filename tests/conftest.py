"""Shared fixtures for MOSS tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from moss.models.batch import Batch, Chunk
from moss.models.config import Settings
from moss.models.keypoint import Keypoint, KeypointMatrix, KeypointScore
from moss.runner.base import Runner, StageResult
from moss.state.paths import MossPaths
from moss.state.store import StateStore


@pytest.fixture
def tmp_moss_home(tmp_path: Path) -> Path:
    """Create a temporary MOSS_HOME directory."""
    home = tmp_path / ".moss"
    home.mkdir()
    return home


@pytest.fixture
def paths(tmp_moss_home: Path) -> MossPaths:
    """MossPaths pointing to a temp directory."""
    p = MossPaths(tmp_moss_home)
    p.ensure_all()
    return p


@pytest.fixture
def store(paths: MossPaths) -> StateStore:
    """StateStore with temp paths."""
    return StateStore(paths)


@pytest.fixture
def settings() -> Settings:
    """Default settings."""
    return Settings()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunks for testing."""
    return [
        Chunk(
            session_id="session-1",
            cursor_start=0,
            cursor_end=10,
            content="User asked to file a compliance report.\nAgent failed to locate form.",
            keypoints=["task_comprehension", "action_selection"],
        ),
        Chunk(
            session_id="session-1",
            cursor_start=10,
            cursor_end=20,
            content="Agent attempted retry but used wrong endpoint.\nError: 404 Not Found.",
            keypoints=["error_recovery", "action_execution"],
        ),
    ]


@pytest.fixture
def sample_batch(sample_chunks: list[Chunk]) -> Batch:
    """A sample sealed batch."""
    batch = Batch(id="batch-001", chunks=sample_chunks, conversation_id="conv-1")
    batch.seal()
    return batch


@pytest.fixture
def sample_keypoint_matrix() -> KeypointMatrix:
    """A sample keypoint matrix for testing."""
    return KeypointMatrix(
        tasks={
            "task-1": [
                Keypoint(name="task_comprehension", score=KeypointScore.STRONG),
                Keypoint(name="action_selection", score=KeypointScore.ADEQUATE),
                Keypoint(name="error_recovery", score=KeypointScore.WEAK),
            ],
            "task-2": [
                Keypoint(name="task_comprehension", score=KeypointScore.ADEQUATE),
                Keypoint(name="action_selection", score=KeypointScore.WEAK),
                Keypoint(name="error_recovery", score=KeypointScore.MISSING),
            ],
        }
    )


@pytest.fixture
def baseline_matrix() -> KeypointMatrix:
    """A lower-scoring baseline matrix."""
    return KeypointMatrix(
        tasks={
            "task-1": [
                Keypoint(name="task_comprehension", score=KeypointScore.WEAK),
                Keypoint(name="action_selection", score=KeypointScore.WEAK),
                Keypoint(name="error_recovery", score=KeypointScore.MISSING),
            ],
            "task-2": [
                Keypoint(name="task_comprehension", score=KeypointScore.WEAK),
                Keypoint(name="action_selection", score=KeypointScore.MISSING),
                Keypoint(name="error_recovery", score=KeypointScore.MISSING),
            ],
        }
    )


class MockRunner(Runner):
    """Mock runner for testing pipeline stages."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.invocations: list[dict[str, Any]] = []

    async def invoke(
        self,
        stage: str,
        prompt: str,
        workdir: Path,
        context: dict[str, Any],
    ) -> StageResult:
        self.invocations.append({
            "stage": stage,
            "prompt": prompt,
            "workdir": workdir,
            "context": context,
        })
        output = self.responses.get(stage, f"Mock output for {stage}")
        return StageResult(output=output, success=True)

    async def health_check(self) -> bool:
        return True

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> dict[str, Any]:
        return {"mock": True}


@pytest.fixture
def mock_runner() -> MockRunner:
    """A mock runner instance."""
    return MockRunner()


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """A mock Docker client."""
    client = MagicMock()
    client.containers = MagicMock()
    client.images = MagicMock()
    client.networks = MagicMock()
    return client
