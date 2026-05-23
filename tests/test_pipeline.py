"""Tests for pipeline stages and orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from moss.models.batch import Batch, Chunk
from moss.models.keypoint import Keypoint, KeypointMatrix, KeypointScore
from moss.pipeline.stages.locate import LocateStage
from moss.pipeline.stages.plan import PlanStage
from moss.pipeline.stages.plan_review import PlanReviewDecision, PlanReviewStage
from moss.pipeline.stages.implement import ImplementStage
from moss.pipeline.stages.code_review import CodeReviewDecision, CodeReviewStage
from moss.pipeline.stages.task_evaluate import TaskEvaluateStage
from moss.pipeline.stages.verdict import VerdictStage
from moss.models.verdict import Verdict
from tests.conftest import MockRunner


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path / "workdir"


class TestLocateStage:
    @pytest.mark.asyncio
    async def test_locate_produces_output(
        self, mock_runner: MockRunner, sample_batch: Batch, workdir: Path
    ) -> None:
        stage = LocateStage(runner=mock_runner)
        result = await stage.run(
            batch=sample_batch,
            transcripts="Sample transcript data",
            workdir=workdir,
        )
        assert result.success
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_locate_passes_chunks_to_prompt(
        self, mock_runner: MockRunner, sample_batch: Batch, workdir: Path
    ) -> None:
        stage = LocateStage(runner=mock_runner)
        await stage.run(batch=sample_batch, transcripts="", workdir=workdir)

        assert len(mock_runner.invocations) == 1
        assert mock_runner.invocations[0]["stage"] == "locate"
        prompt = mock_runner.invocations[0]["prompt"]
        assert "session-1" in prompt


class TestPlanStage:
    @pytest.mark.asyncio
    async def test_plan_produces_output(
        self, mock_runner: MockRunner, workdir: Path
    ) -> None:
        stage = PlanStage(runner=mock_runner)
        result = await stage.run(diagnosis="Found bug in routing", workdir=workdir)
        assert result.success


class TestPlanReviewStage:
    @pytest.mark.asyncio
    async def test_approve(self, workdir: Path) -> None:
        runner = MockRunner(responses={"plan_review": "APPROVE\nLooks good."})
        stage = PlanReviewStage(runner=runner, max_rounds=3)
        result = await stage.run(
            diagnosis="diagnosis", plan="plan", workdir=workdir
        )
        assert result.success
        assert result.metadata["decision"] == "approve"
        assert result.metadata["rounds"] == 1

    @pytest.mark.asyncio
    async def test_reject_then_approve(self, workdir: Path) -> None:
        call_count = 0

        class RoundedRunner(MockRunner):
            async def invoke(self, stage, prompt, workdir, context):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return self._make_result("REJECT-OFF-TARGET\nNeeds fixing")
                return self._make_result("APPROVE\nFixed now")

            def _make_result(self, output):
                from moss.runner.base import StageResult
                return StageResult(output=output, success=True)

        runner = RoundedRunner()
        stage = PlanReviewStage(runner=runner, max_rounds=3)
        result = await stage.run(
            diagnosis="diagnosis", plan="plan", workdir=workdir
        )
        assert result.metadata["decision"] == "approve"
        assert result.metadata["rounds"] == 2

    @pytest.mark.asyncio
    async def test_parse_decisions(self) -> None:
        assert PlanReviewStage._parse_decision("APPROVE\nOK") == PlanReviewDecision.APPROVE
        assert (
            PlanReviewStage._parse_decision("REJECT-OFF-TARGET\nBad")
            == PlanReviewDecision.REJECT_OFF_TARGET
        )
        assert (
            PlanReviewStage._parse_decision("REJECT-TOO-NARROW\nIncomplete")
            == PlanReviewDecision.REJECT_TOO_NARROW
        )


class TestImplementStage:
    @pytest.mark.asyncio
    async def test_implement(self, mock_runner: MockRunner, workdir: Path) -> None:
        stage = ImplementStage(runner=mock_runner)
        result = await stage.run(plan="Fix the routing", workdir=workdir)
        assert result.success
        assert mock_runner.invocations[0]["stage"] == "implement"


class TestCodeReviewStage:
    @pytest.mark.asyncio
    async def test_approve(self, workdir: Path) -> None:
        runner = MockRunner(responses={"code_review": "APPROVE\nDiff looks correct."})
        stage = CodeReviewStage(runner=runner)
        result = await stage.run(
            plan="plan", implementation="impl", diff="diff", workdir=workdir
        )
        assert result.metadata["decision"] == "approve"

    @pytest.mark.asyncio
    async def test_parse_decisions(self) -> None:
        assert CodeReviewStage._parse_decision("APPROVE") == CodeReviewDecision.APPROVE
        assert (
            CodeReviewStage._parse_decision("REQUEST-CHANGES\nFix X")
            == CodeReviewDecision.REQUEST_CHANGES
        )


class TestTaskEvaluateStage:
    @pytest.mark.asyncio
    async def test_evaluate_with_json_output(self, workdir: Path) -> None:
        matrix_json = json.dumps({
            "tasks": {
                "task-1": [
                    {"name": "comprehension", "score": "strong"},
                    {"name": "execution", "score": "adequate"},
                ],
            }
        })
        runner = MockRunner(responses={"task_evaluate": matrix_json})
        stage = TaskEvaluateStage(runner=runner)
        result = await stage.run(
            transcripts={"task-1": "transcript data"},
            keypoint_names=["comprehension", "execution"],
            workdir=workdir,
        )
        assert result.success
        assert "keypoint_matrix" in result.metadata

    def test_parse_matrix_with_code_fences(self) -> None:
        output = '```json\n{"tasks": {"t1": [{"name": "kp", "score": "strong"}]}}\n```'
        matrix = TaskEvaluateStage._parse_matrix(output)
        assert matrix is not None
        assert "t1" in matrix.tasks

    def test_parse_matrix_invalid(self) -> None:
        matrix = TaskEvaluateStage._parse_matrix("not json at all")
        assert matrix is None


class TestVerdictStage:
    @pytest.mark.asyncio
    async def test_converged_verdict(
        self, baseline_matrix: KeypointMatrix, sample_keypoint_matrix: KeypointMatrix, workdir: Path
    ) -> None:
        runner = MockRunner(responses={"verdict": "CONVERGED\nScores are sufficient."})
        stage = VerdictStage(runner=runner, plateau_threshold=3)
        result = await stage.run(
            baseline=baseline_matrix,
            current=sample_keypoint_matrix,
            plateau_count=0,
            workdir=workdir,
        )
        assert result.metadata["verdict"] == "converged"

    @pytest.mark.asyncio
    async def test_forced_plateau(
        self, baseline_matrix: KeypointMatrix, sample_keypoint_matrix: KeypointMatrix, workdir: Path
    ) -> None:
        runner = MockRunner()
        stage = VerdictStage(runner=runner, plateau_threshold=3)
        result = await stage.run(
            baseline=baseline_matrix,
            current=sample_keypoint_matrix,
            plateau_count=3,
            workdir=workdir,
        )
        assert result.metadata["verdict"] == "converged"
        assert result.metadata["forced_plateau"] is True
        # Runner should NOT have been called for forced plateau
        assert len(runner.invocations) == 0

    def test_parse_verdict(self) -> None:
        assert VerdictStage._parse_verdict("CONVERGED\nDone") == Verdict.CONVERGED
        assert VerdictStage._parse_verdict("NEED_MORE_WORK\nKeep going") == Verdict.NEED_MORE_WORK
        assert (
            VerdictStage._parse_verdict("FUNDAMENTAL_LIMIT_MODEL")
            == Verdict.FUNDAMENTAL_LIMIT_MODEL
        )
