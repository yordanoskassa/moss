"""Real ClawEval benchmark tests — run claweval against OpenAI, bridge into MOSS.

These tests:
  1. Run ClawEval's deterministic eval suite against an OpenAI model
  2. Convert results into MOSS Batch + KeypointMatrix via the bridge
  3. Optionally feed benchmark failures through the MOSS pipeline (Locate → Plan)

Usage:
    pytest tests/test_benchmark.py -v -s
    pytest tests/test_benchmark.py -v -s -k baseline      # only baseline scoring
    pytest tests/test_benchmark.py -v -s -k pipeline       # only pipeline test

Requirements:
    - OPENAI_API_KEY env var set
    - claweval installed (pip install -e /path/to/claweval)
    - For pipeline tests: `claude` CLI on PATH
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from moss.eval.claweval_bridge import (
    BRIDGE_KEYPOINTS,
    CATEGORY_TO_KEYPOINT,
    ClawEvalBridge,
)
from moss.eval.runner import ClawEvalRunner
from moss.models.batch import Batch
from moss.models.keypoint import KeypointMatrix, KeypointScore

# ---------------------------------------------------------------------------
# Markers and skip conditions
# ---------------------------------------------------------------------------

benchmark = pytest.mark.benchmark

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

skip_no_openai = pytest.mark.skipif(
    not _has_openai_key,
    reason="OPENAI_API_KEY not set — skipping benchmark tests",
)


def _claude_available() -> bool:
    if shutil.which("claude") is None:
        return False
    try:
        result = subprocess.run(
            ["claude", "--version"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


skip_no_claude = pytest.mark.skipif(
    not _claude_available(),
    reason="claude CLI not found — skipping pipeline benchmark tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("benchmark")


@pytest.fixture(scope="module")
def claweval_runner() -> ClawEvalRunner:
    """ClawEvalRunner targeting OpenAI gpt-4o-mini."""
    return ClawEvalRunner(
        model_id="gpt-4o-mini",
        scoring_mode="deterministic",
        timeout=120,
    )


@pytest.fixture(scope="module")
def claweval_results(claweval_runner: ClawEvalRunner) -> ClawEvalRunner:
    """Run a quick subset (tool_calling + coding) to keep costs low."""
    claweval_runner.run(categories=["tool_calling", "coding"])
    return claweval_runner


# ---------------------------------------------------------------------------
# Unit tests for the bridge (no API key needed)
# ---------------------------------------------------------------------------


class TestClawEvalBridgeUnit:
    """Test the bridge with synthetic data — no API calls."""

    SAMPLE_RESULTS: dict = {
        "models": {
            "test-model": {
                "name": "Test Model",
                "overall": 0.65,
                "categories": {
                    "tool_calling": 0.80,
                    "coding": 0.70,
                    "reasoning": 0.55,
                    "writing": 0.60,
                    "research": 0.50,
                    "memory": 0.75,
                    "speed": 0.65,
                },
                "tasks": [
                    {
                        "task_id": "tool_calling_001",
                        "model_id": "test-model",
                        "score": {
                            "task_id": "tool_calling_001",
                            "total_score": 0.85,
                            "breakdown": {
                                "correct_tool": 1.0,
                                "correct_params": 0.8,
                                "response_quality": 0.75,
                            },
                            "details": {},
                            "judge_score": None,
                        },
                        "timing": {
                            "wall_clock_ms": 1200,
                            "ttft_ms": 200,
                            "total_tokens": 500,
                            "prompt_tokens": 300,
                            "completion_tokens": 200,
                            "tokens_per_second": 40.0,
                            "chunk_count": 10,
                            "estimated_gen_tok_s": 45.0,
                        },
                        "response_text": "I'll read the file for you.",
                        "tool_calls_made": [
                            {"name": "read_file", "arguments": {"path": "/test.txt"}}
                        ],
                        "error": "",
                    },
                    {
                        "task_id": "coding_002",
                        "model_id": "test-model",
                        "score": {
                            "task_id": "coding_002",
                            "total_score": 0.20,
                            "breakdown": {"response_quality": 0.20},
                            "details": {},
                            "judge_score": None,
                        },
                        "timing": {
                            "wall_clock_ms": 2000,
                            "ttft_ms": 300,
                            "total_tokens": 800,
                            "prompt_tokens": 400,
                            "completion_tokens": 400,
                            "tokens_per_second": 35.0,
                            "chunk_count": 15,
                            "estimated_gen_tok_s": 40.0,
                        },
                        "response_text": "def add(a, b): return a - b",
                        "tool_calls_made": [],
                        "error": "",
                    },
                    {
                        "task_id": "reasoning_003",
                        "model_id": "test-model",
                        "score": {
                            "task_id": "reasoning_003",
                            "total_score": 0.45,
                            "breakdown": {"response_quality": 0.45},
                            "details": {},
                            "judge_score": None,
                        },
                        "timing": {
                            "wall_clock_ms": 1500,
                            "ttft_ms": 250,
                            "total_tokens": 600,
                            "prompt_tokens": 350,
                            "completion_tokens": 250,
                            "tokens_per_second": 38.0,
                            "chunk_count": 12,
                            "estimated_gen_tok_s": 42.0,
                        },
                        "response_text": "The answer is 42.",
                        "tool_calls_made": [],
                        "error": "",
                    },
                ],
            }
        }
    }

    def test_load_from_dict(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        assert bridge.overall_score() == 0.65

    def test_category_scores(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        cats = bridge.category_scores()
        assert cats["tool_calling"] == 0.80
        assert cats["coding"] == 0.70

    def test_to_keypoint_matrix(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        matrix = bridge.to_keypoint_matrix()

        assert isinstance(matrix, KeypointMatrix)
        assert len(matrix.tasks) == 3

        # tool_calling_001 scored 0.85 → primary keypoint should be STRONG
        tc_kps = matrix.tasks["tool_calling_001"]
        primary = tc_kps[0]
        assert primary.name == "action_execution"
        assert primary.score == KeypointScore.STRONG

        # coding_002 scored 0.20 → primary keypoint should be MISSING
        code_kps = matrix.tasks["coding_002"]
        primary = code_kps[0]
        assert primary.name == "action_selection"
        assert primary.score == KeypointScore.MISSING

        # Aggregate score should be a real number
        agg = matrix.aggregate_score()
        assert 0.0 < agg < 1.0

    def test_to_batch_all_tasks(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        batch = bridge.to_batch(score_threshold=1.0)

        assert isinstance(batch, Batch)
        assert batch.is_sealed
        assert batch.chunk_count == 3  # All tasks score < 1.0

    def test_to_batch_failures_only(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        batch = bridge.to_batch(score_threshold=0.6)

        assert batch.is_sealed
        # Only coding_002 (0.20) and reasoning_003 (0.45) are below 0.6
        assert batch.chunk_count == 2

        # Chunks should contain score and task info
        for chunk in batch.chunks:
            assert "Score:" in chunk.content
            assert "Task:" in chunk.content

    def test_transcripts(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        transcripts = bridge.transcripts()

        assert len(transcripts) == 3
        assert "tool_calling_001" in transcripts
        assert "read_file" in transcripts["tool_calling_001"]
        assert "Score: 0.850" in transcripts["tool_calling_001"]

    def test_model_not_found_raises(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict(self.SAMPLE_RESULTS)
        with pytest.raises(KeyError, match="nonexistent"):
            bridge.to_keypoint_matrix(model_id="nonexistent")

    def test_empty_results_raises(self) -> None:
        bridge = ClawEvalBridge()
        bridge.load_from_dict({"models": {}})
        with pytest.raises(ValueError, match="No model results"):
            bridge.to_keypoint_matrix()


# ---------------------------------------------------------------------------
# Real benchmark tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@skip_no_openai
@benchmark
class TestClawEvalBenchmark:
    """Run ClawEval, convert to MOSS structures, verify real scores."""

    async def test_baseline_scoring(
        self,
        claweval_results: ClawEvalRunner,
        benchmark_workdir: Path,
    ) -> None:
        """Run claweval, convert results, verify aggregate score is a real number."""
        bridge = claweval_results.bridge()

        # Overall score should be between 0 and 1
        overall = bridge.overall_score()
        print(f"\nOverall ClawEval score: {overall:.4f}")
        assert 0.0 <= overall <= 1.0, f"Overall score out of range: {overall}"

        # Category scores
        cats = bridge.category_scores()
        print("Category scores:")
        for cat, score in sorted(cats.items()):
            print(f"  {cat}: {score:.4f}")
        assert len(cats) > 0

        # Convert to KeypointMatrix
        matrix = bridge.to_keypoint_matrix()
        assert isinstance(matrix, KeypointMatrix)
        assert len(matrix.tasks) > 0

        agg = matrix.aggregate_score()
        print(f"\nMOSS KeypointMatrix aggregate score: {agg:.4f}")
        assert 0.0 <= agg <= 1.0

        # Per-task breakdown
        print("\nPer-task keypoint scores:")
        for task_id, kps in sorted(matrix.tasks.items()):
            scores_str = ", ".join(f"{kp.name}={kp.score.value}" for kp in kps)
            print(f"  {task_id}: {scores_str}")

        # Save results
        results_path = claweval_results.save_results(output_dir=benchmark_workdir)
        print(f"\nResults saved to: {results_path}")
        assert results_path.exists()

    async def test_batch_from_failures(
        self,
        claweval_results: ClawEvalRunner,
    ) -> None:
        """Convert low-scoring tasks into a MOSS Batch."""
        bridge = claweval_results.bridge()

        # Get tasks that scored below 0.6
        batch = bridge.to_batch(score_threshold=0.6)
        assert isinstance(batch, Batch)
        assert batch.is_sealed

        print(f"\nFailure batch: {batch.chunk_count} chunks (score < 0.6)")
        for chunk in batch.chunks:
            first_line = chunk.content.split("\n")[0]
            print(f"  {chunk.session_id}: {first_line}")

    async def test_keypoint_matrix_aggregate(
        self,
        claweval_results: ClawEvalRunner,
    ) -> None:
        """Verify KeypointMatrix aggregate_score produces a real number."""
        bridge = claweval_results.bridge()
        matrix = bridge.to_keypoint_matrix()

        score = matrix.aggregate_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\nAggregate keypoint score: {score:.4f}")
        print(f"Total tasks in matrix: {len(matrix.tasks)}")
        print(f"Total keypoints: {sum(len(kps) for kps in matrix.tasks.values())}")


# ---------------------------------------------------------------------------
# Pipeline integration (requires OPENAI_API_KEY + claude CLI)
# ---------------------------------------------------------------------------


@skip_no_openai
@skip_no_claude
@benchmark
class TestMossPipelineOnBenchmark:
    """Feed real benchmark failures through Locate → Plan → Evaluate."""

    async def test_pipeline_on_benchmark_failures(
        self,
        claweval_results: ClawEvalRunner,
        benchmark_workdir: Path,
    ) -> None:
        """Run MOSS pipeline stages on real ClawEval failure data."""
        from moss.models.config import RunnerConfig
        from moss.pipeline.stages.locate import LocateStage
        from moss.pipeline.stages.plan import PlanStage
        from moss.pipeline.stages.task_evaluate import TaskEvaluateStage
        from moss.runner.claude_code import ClaudeCodeRunner

        bridge = claweval_results.bridge()

        # Build batch from failures (score < 0.6)
        batch = bridge.to_batch(score_threshold=0.6)
        if batch.chunk_count == 0:
            pytest.skip("No failing tasks (all scored >= 0.6) — nothing to diagnose")

        transcripts = bridge.transcripts()
        # Build a transcript summary for the Locate stage
        transcript_text = "\n\n".join(
            f"--- {tid} ---\n{text}" for tid, text in transcripts.items()
        )

        runner = ClaudeCodeRunner(config=RunnerConfig(timeout=300))
        pipeline_dir = benchmark_workdir / "pipeline"
        pipeline_dir.mkdir(exist_ok=True)

        # ---- Stage 1: Locate ----
        print("\n" + "=" * 70)
        print("BENCHMARK PIPELINE — Stage 1: LOCATE")
        print("=" * 70)

        locate = LocateStage(runner=runner)
        locate_result = await locate.run(
            batch=batch,
            transcripts=transcript_text,
            workdir=pipeline_dir,
        )

        print(locate_result.output[:2000])
        assert locate_result.success, f"Locate failed: {locate_result.error}"
        (pipeline_dir / "locate.md").write_text(locate_result.output, encoding="utf-8")

        # ---- Stage 2: Plan ----
        print("\n" + "=" * 70)
        print("BENCHMARK PIPELINE — Stage 2: PLAN")
        print("=" * 70)

        plan = PlanStage(runner=runner)
        plan_result = await plan.run(
            diagnosis=locate_result.output,
            workdir=pipeline_dir,
        )

        print(plan_result.output[:2000])
        assert plan_result.success, f"Plan failed: {plan_result.error}"
        (pipeline_dir / "plan.md").write_text(plan_result.output, encoding="utf-8")

        # ---- Stage 3: TaskEvaluate ----
        print("\n" + "=" * 70)
        print("BENCHMARK PIPELINE — Stage 3: TASK EVALUATE")
        print("=" * 70)

        # Use failure transcripts for evaluation
        failure_transcripts = {
            tid: text
            for tid, text in transcripts.items()
            if any(
                chunk.session_id == f"claweval-{tid}" for chunk in batch.chunks
            )
        }
        if not failure_transcripts:
            # Fall back to all transcripts if session_id matching fails
            failure_transcripts = transcripts

        evaluate = TaskEvaluateStage(runner=runner)
        eval_result = await evaluate.run(
            transcripts=failure_transcripts,
            keypoint_names=BRIDGE_KEYPOINTS,
            workdir=pipeline_dir,
        )

        print(eval_result.output[:2000])
        assert eval_result.success, f"TaskEvaluate failed: {eval_result.error}"

        if "keypoint_matrix" in eval_result.metadata:
            moss_matrix = KeypointMatrix(**eval_result.metadata["keypoint_matrix"])
            moss_score = moss_matrix.aggregate_score()
            claweval_score = bridge.overall_score()

            print("\n" + "=" * 70)
            print("BENCHMARK PIPELINE — COMPARISON")
            print("=" * 70)
            print(f"ClawEval overall score:        {claweval_score:.4f}")
            print(f"MOSS KeypointMatrix score:     {moss_score:.4f}")
            print(f"Bridge KeypointMatrix score:   {bridge.to_keypoint_matrix().aggregate_score():.4f}")

            (pipeline_dir / "task-evaluate.json").write_text(
                json.dumps(eval_result.metadata["keypoint_matrix"], indent=2),
                encoding="utf-8",
            )

        print(f"\nPipeline artifacts saved to: {pipeline_dir}")
