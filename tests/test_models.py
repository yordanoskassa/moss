"""Tests for Pydantic models."""

from __future__ import annotations

import json

from moss.models.batch import Batch, BatchStatus, Chunk
from moss.models.config import DepthDial
from moss.models.evolution import EvolutionState, EvolutionStatus
from moss.models.keypoint import Keypoint, KeypointMatrix, KeypointScore
from moss.models.verdict import Verdict


class TestChunk:
    def test_create_chunk(self) -> None:
        chunk = Chunk(
            session_id="s1",
            cursor_start=0,
            cursor_end=10,
            content="test content",
        )
        assert chunk.session_id == "s1"
        assert chunk.cursor_start == 0
        assert chunk.cursor_end == 10
        assert chunk.keypoints == []

    def test_chunk_serialization(self) -> None:
        chunk = Chunk(
            session_id="s1",
            cursor_start=0,
            cursor_end=5,
            content="hello",
            keypoints=["kp1", "kp2"],
        )
        data = chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored == chunk


class TestBatch:
    def test_create_batch(self) -> None:
        batch = Batch(id="batch-001", conversation_id="conv-1")
        assert batch.status == BatchStatus.OPEN
        assert batch.chunk_count == 0
        assert not batch.is_sealed

    def test_seal_batch(self) -> None:
        batch = Batch(id="batch-001", conversation_id="conv-1")
        batch.seal()
        assert batch.is_sealed
        assert batch.status == BatchStatus.SEALED

    def test_batch_serialization(self, sample_batch: Batch) -> None:
        json_str = sample_batch.model_dump_json()
        restored = Batch.model_validate_json(json_str)
        assert restored.id == sample_batch.id
        assert restored.chunk_count == sample_batch.chunk_count
        assert restored.is_sealed


class TestKeypoint:
    def test_keypoint_scores(self) -> None:
        assert KeypointScore.STRONG.numeric == 1.0
        assert KeypointScore.ADEQUATE.numeric == 0.67
        assert KeypointScore.WEAK.numeric == 0.33
        assert KeypointScore.MISSING.numeric == 0.0

    def test_keypoint_improvement(self) -> None:
        kp1 = Keypoint(name="test", score=KeypointScore.STRONG)
        kp2 = Keypoint(name="test", score=KeypointScore.WEAK)
        assert kp1.improved_over(kp2)
        assert not kp2.improved_over(kp1)

    def test_matrix_aggregate(self, sample_keypoint_matrix: KeypointMatrix) -> None:
        score = sample_keypoint_matrix.aggregate_score()
        assert 0 < score < 1

    def test_matrix_improvement(
        self,
        sample_keypoint_matrix: KeypointMatrix,
        baseline_matrix: KeypointMatrix,
    ) -> None:
        assert sample_keypoint_matrix.improved_over(baseline_matrix)
        assert not baseline_matrix.improved_over(sample_keypoint_matrix)

    def test_matrix_delta(
        self,
        sample_keypoint_matrix: KeypointMatrix,
        baseline_matrix: KeypointMatrix,
    ) -> None:
        delta = sample_keypoint_matrix.improvement_delta(baseline_matrix)
        assert delta > 0

    def test_empty_matrix(self) -> None:
        matrix = KeypointMatrix()
        assert matrix.aggregate_score() == 0.0

    def test_matrix_serialization(self, sample_keypoint_matrix: KeypointMatrix) -> None:
        json_str = sample_keypoint_matrix.model_dump_json()
        restored = KeypointMatrix.model_validate_json(json_str)
        assert restored.aggregate_score() == sample_keypoint_matrix.aggregate_score()


class TestVerdict:
    def test_terminal_verdicts(self) -> None:
        assert Verdict.CONVERGED.is_terminal
        assert Verdict.FUNDAMENTAL_LIMIT_MODEL.is_terminal
        assert Verdict.FUNDAMENTAL_LIMIT_ARCHITECTURE.is_terminal
        assert not Verdict.NEED_MORE_WORK.is_terminal

    def test_verdict_values(self) -> None:
        assert Verdict.CONVERGED.value == "converged"
        assert Verdict.NEED_MORE_WORK.value == "need_more_work"


class TestDepthDial:
    def test_light(self) -> None:
        d = DepthDial.LIGHT
        assert d.max_iterations == 2
        assert d.stage_round_budget == 1
        assert d.trials_per_task == 1
        assert d.plateau_threshold == 1

    def test_standard(self) -> None:
        d = DepthDial.STANDARD
        assert d.max_iterations == 5
        assert d.stage_round_budget == 3
        assert d.trials_per_task == 3

    def test_deep(self) -> None:
        d = DepthDial.DEEP
        assert d.max_iterations == 10
        assert d.stage_round_budget == 5


class TestEvolutionState:
    def test_create(self) -> None:
        state = EvolutionState(batch_id="batch-001")
        assert state.current_iteration == 0
        assert state.status == EvolutionStatus.PENDING
        assert not state.is_terminal

    def test_advance(self) -> None:
        state = EvolutionState(batch_id="batch-001")
        state.advance_iteration()
        assert state.current_iteration == 1

    def test_terminal_by_verdict(self) -> None:
        state = EvolutionState(batch_id="batch-001")
        state.verdict = Verdict.CONVERGED
        assert state.is_terminal

    def test_terminal_by_max_iterations(self) -> None:
        state = EvolutionState(batch_id="batch-001", max_iterations=2)
        state.current_iteration = 2
        assert state.is_terminal

    def test_terminal_by_status(self) -> None:
        state = EvolutionState(batch_id="batch-001")
        state.status = EvolutionStatus.FAILED
        assert state.is_terminal

    def test_serialization(self) -> None:
        state = EvolutionState(batch_id="batch-001", depth="deep")
        state.advance_iteration()
        state.verdict = Verdict.NEED_MORE_WORK

        json_str = state.model_dump_json()
        restored = EvolutionState.model_validate_json(json_str)
        assert restored.batch_id == "batch-001"
        assert restored.current_iteration == 1
        assert restored.verdict == Verdict.NEED_MORE_WORK
