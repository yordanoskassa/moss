"""Real integration tests — run Locate → Plan → Task Evaluate against the live Claude CLI.

These tests verify that the MOSS pipeline stages work end-to-end with a real
ClaudeCodeRunner subprocess instead of the MockRunner used everywhere else.

Only the read-only/analysis stages are tested (no file writes, no git commits):
  - LocateStage:      diagnose failures from batch chunks
  - PlanStage:        produce a fix specification from the diagnosis
  - TaskEvaluateStage: score keypoints per task from transcripts

Usage:
    pytest tests/test_integration.py -v -s          # -s shows Claude's real output
    pytest tests/test_integration.py -v -s -k locate   # run only Locate

Tests skip automatically when `claude` CLI is not on PATH (CI-friendly).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from moss.models.batch import Batch, Chunk
from moss.models.config import RunnerConfig
from moss.pipeline.stages.locate import LocateStage
from moss.pipeline.stages.plan import PlanStage
from moss.pipeline.stages.task_evaluate import TaskEvaluateStage
from moss.runner.claude_code import ClaudeCodeRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

integration = pytest.mark.integration


def _claude_available() -> bool:
    """Return True if the `claude` CLI is reachable."""
    if shutil.which("claude") is None:
        return False
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


skip_no_claude = pytest.mark.skipif(
    not _claude_available(),
    reason="claude CLI not found — skipping integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner() -> ClaudeCodeRunner:
    """ClaudeCodeRunner with a generous timeout for integration tests."""
    config = RunnerConfig(timeout=300)
    return ClaudeCodeRunner(config=config)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary working directory for stage invocations."""
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="module")
def failure_batch() -> Batch:
    """Realistic failure batch: compliance audit filing failures."""
    batch = Batch(id="integ-batch-001", conversation_id="integ-conv-001")
    batch.chunks = [
        Chunk(
            session_id="session-integ-1",
            cursor_start=0,
            cursor_end=42,
            content=(
                "Task: File a compliance audit report via the /api/v2/reports endpoint.\n"
                "Agent navigated to the reports page and attempted POST /api/v2/reports.\n"
                "Response: 404 Not Found — endpoint was moved to /api/v3/compliance/reports.\n"
                "Agent did not consult the API changelog or discover the new path.\n"
                "Result: Task failed — report was never submitted."
            ),
            keypoints=["task_comprehension", "action_selection", "tool_usage"],
        ),
        Chunk(
            session_id="session-integ-2",
            cursor_start=0,
            cursor_end=38,
            content=(
                "Task: Retry the compliance report submission after initial failure.\n"
                "Agent retried with the same endpoint /api/v2/reports using a cached\n"
                "session token from 15 minutes ago.\n"
                "Response: 401 Unauthorized — token had expired (TTL 10 min).\n"
                "Agent did not refresh the token before retrying.\n"
                "Result: Task failed — stale credential, no re-auth attempt."
            ),
            keypoints=["error_recovery", "action_execution", "state_management"],
        ),
        Chunk(
            session_id="session-integ-3",
            cursor_start=0,
            cursor_end=35,
            content=(
                "Task: Submit the compliance report using the correct v3 endpoint.\n"
                "Agent discovered /api/v3/compliance/reports and sent a POST.\n"
                "Response: 403 Forbidden — the agent's service account lacks the\n"
                "'compliance:write' permission scope.\n"
                "Agent logged the 403 but took no further action (did not escalate\n"
                "or request permission elevation).\n"
                "Result: Task failed — permission denied, no escalation."
            ),
            keypoints=["error_recovery", "action_selection", "tool_usage"],
        ),
    ]
    batch.seal()
    return batch


BASELINE_TRANSCRIPTS = """\
Baseline run (before any fix):
- Task compliance-001: Agent used deprecated v2 endpoint, got 404, gave up.
- Task compliance-002: Agent retried with stale token, got 401, gave up.
- Task compliance-003: Agent found v3 endpoint but was denied by permissions API, logged error, gave up.
Overall: 0/3 tasks completed. Common pattern — the agent lacks endpoint discovery,
token refresh logic, and permission-escalation handling.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_claude
@integration
class TestRealLocateStage:
    """Run LocateStage against the real Claude CLI."""

    @pytest.mark.asyncio
    async def test_locate_returns_diagnosis(
        self, runner: ClaudeCodeRunner, failure_batch: Batch, workdir: Path
    ) -> None:
        stage = LocateStage(runner=runner)
        result = await stage.run(
            batch=failure_batch,
            transcripts=BASELINE_TRANSCRIPTS,
            workdir=workdir,
        )

        # Print full output so `pytest -s` shows Claude's real reasoning
        print("\n===== LOCATE STAGE OUTPUT =====")
        print(result.output)
        print("===== END LOCATE =====\n")

        assert result.success, f"LocateStage failed: {result.error}"
        assert len(result.output) > 100, "Diagnosis too short to be meaningful"

        # The diagnosis should mention the core failure categories
        output_lower = result.output.lower()
        assert any(
            kw in output_lower for kw in ["404", "endpoint", "routing", "discovery"]
        ), "Diagnosis should reference the 404/endpoint-discovery failure"
        assert any(
            kw in output_lower for kw in ["401", "token", "stale", "expired", "credential"]
        ), "Diagnosis should reference the stale-token failure"
        assert any(
            kw in output_lower for kw in ["403", "permission", "forbidden", "scope"]
        ), "Diagnosis should reference the 403/permissions failure"

        # Save artifact to disk
        artifact_path = workdir / "locate.md"
        artifact_path.write_text(result.output, encoding="utf-8")
        assert artifact_path.exists()


@skip_no_claude
@integration
class TestRealPlanStage:
    """Run PlanStage against the real Claude CLI, fed by a realistic diagnosis."""

    SYNTHETIC_DIAGNOSIS = """\
# Diagnosis — Compliance Audit Filing Failures

## Root Cause 1: Missing Endpoint Discovery (3/3 tasks affected)
The agent hard-codes the v2 endpoint `/api/v2/reports`. When this returned 404
the agent had no fallback logic to consult the API changelog or attempt endpoint
discovery via `/api/v3/`.

**Affected traces:**
- session-integ-1 lines 0-42: POST /api/v2/reports → 404

## Root Cause 2: Stale Session Token (1/3 tasks affected)
The agent caches its session token without checking TTL. When retrying after a
failure window >10 minutes the token is expired, producing a 401.

**Affected traces:**
- session-integ-2 lines 0-38: Retry with 15-min-old token → 401

## Root Cause 3: No Permission Escalation (1/3 tasks affected)
On receiving a 403 from the permissions API the agent logs the error and stops.
It does not attempt to request the missing `compliance:write` scope or escalate
to a human operator.

**Affected traces:**
- session-integ-3 lines 0-35: POST v3 endpoint → 403, no escalation

## Severity Ranking
1. Endpoint discovery (blocks all tasks)
2. Token refresh (blocks retries)
3. Permission escalation (blocks authorized submissions)
"""

    @pytest.mark.asyncio
    async def test_plan_returns_fix_spec(
        self, runner: ClaudeCodeRunner, workdir: Path
    ) -> None:
        stage = PlanStage(runner=runner)
        result = await stage.run(
            diagnosis=self.SYNTHETIC_DIAGNOSIS,
            workdir=workdir,
        )

        print("\n===== PLAN STAGE OUTPUT =====")
        print(result.output)
        print("===== END PLAN =====\n")

        assert result.success, f"PlanStage failed: {result.error}"
        assert len(result.output) > 100, "Fix specification too short"

        output_lower = result.output.lower()
        # Plan should reference concrete remediation concepts
        assert any(
            kw in output_lower for kw in ["endpoint", "discovery", "routing", "fallback", "v3"]
        ), "Plan should address endpoint discovery"
        assert any(
            kw in output_lower for kw in ["token", "refresh", "ttl", "re-auth", "expir"]
        ), "Plan should address token refresh"
        assert any(
            kw in output_lower for kw in ["permission", "escalat", "scope", "403"]
        ), "Plan should address permission escalation"

        artifact_path = workdir / "plan.md"
        artifact_path.write_text(result.output, encoding="utf-8")
        assert artifact_path.exists()


@skip_no_claude
@integration
class TestRealTaskEvaluateStage:
    """Run TaskEvaluateStage against the real Claude CLI."""

    TASK_TRANSCRIPTS = {
        "compliance-001": (
            "Agent received task: file compliance audit report.\n"
            "Agent attempted POST /api/v2/reports.\n"
            "Server returned 404 Not Found.\n"
            "Agent logged 'endpoint not found' and terminated.\n"
            "No retry, no alternative endpoint attempted."
        ),
        "compliance-002": (
            "Agent received task: retry compliance report filing.\n"
            "Agent reused cached session token (15 min old, TTL=10 min).\n"
            "Server returned 401 Unauthorized.\n"
            "Agent logged 'authentication failed' and terminated.\n"
            "No token refresh attempted."
        ),
        "compliance-003": (
            "Agent received task: submit compliance report via v3 API.\n"
            "Agent correctly identified /api/v3/compliance/reports.\n"
            "Server returned 403 Forbidden (missing 'compliance:write' scope).\n"
            "Agent logged 'permission denied' but took no escalation action.\n"
            "Task marked as failed."
        ),
    }

    KEYPOINT_NAMES = [
        "task_comprehension",
        "action_selection",
        "error_recovery",
        "tool_usage",
    ]

    @pytest.mark.asyncio
    async def test_task_evaluate_returns_scores(
        self, runner: ClaudeCodeRunner, workdir: Path
    ) -> None:
        stage = TaskEvaluateStage(runner=runner)
        result = await stage.run(
            transcripts=self.TASK_TRANSCRIPTS,
            keypoint_names=self.KEYPOINT_NAMES,
            workdir=workdir,
        )

        print("\n===== TASK EVALUATE STAGE OUTPUT =====")
        print(result.output)
        print("===== END TASK EVALUATE =====\n")

        assert result.success, f"TaskEvaluateStage failed: {result.error}"

        # The stage should have parsed the output into a keypoint matrix
        assert "keypoint_matrix" in result.metadata, (
            "TaskEvaluateStage did not parse a keypoint matrix from Claude's output"
        )

        matrix_data = result.metadata["keypoint_matrix"]
        print("\n===== PARSED KEYPOINT MATRIX =====")
        import json
        print(json.dumps(matrix_data, indent=2))
        print("===== END MATRIX =====\n")

        # Verify structure: should have entries for our 3 tasks
        tasks = matrix_data.get("tasks", {})
        assert len(tasks) >= 2, f"Expected at least 2 tasks scored, got {len(tasks)}"

        # Each task should have keypoint scores
        for task_id, keypoints in tasks.items():
            assert len(keypoints) > 0, f"Task {task_id} has no keypoint scores"
            for kp in keypoints:
                assert "name" in kp, f"Keypoint missing 'name' in task {task_id}"
                assert "score" in kp, f"Keypoint missing 'score' in task {task_id}"
                assert kp["score"] in (
                    "strong", "adequate", "weak", "missing"
                ), f"Invalid score '{kp['score']}' for {kp['name']} in {task_id}"

        artifact_path = workdir / "task-evaluate.json"
        artifact_path.write_text(json.dumps(matrix_data, indent=2), encoding="utf-8")
        assert artifact_path.exists()


@skip_no_claude
@integration
class TestRealHealthCheck:
    """Verify ClaudeCodeRunner.health_check() works with the real CLI."""

    @pytest.mark.asyncio
    async def test_health_check_passes(self, runner: ClaudeCodeRunner) -> None:
        healthy = await runner.health_check()
        assert healthy, "health_check() should return True when claude CLI is available"


# ---------------------------------------------------------------------------
# Chained pipeline test — real output flows between stages, no synthetic seams
# ---------------------------------------------------------------------------

CHAINED_TRANSCRIPTS = {
    "compliance-001": (
        "Task: File a compliance audit report via the /api/v2/reports endpoint.\n"
        "Agent navigated to the reports page and attempted POST /api/v2/reports.\n"
        "Response: 404 Not Found — endpoint was moved to /api/v3/compliance/reports.\n"
        "Agent did not consult the API changelog or discover the new path.\n"
        "Result: Task failed — report was never submitted."
    ),
    "compliance-002": (
        "Task: Retry the compliance report submission after initial failure.\n"
        "Agent retried with the same endpoint /api/v2/reports using a cached\n"
        "session token from 15 minutes ago.\n"
        "Response: 401 Unauthorized — token had expired (TTL 10 min).\n"
        "Agent did not refresh the token before retrying.\n"
        "Result: Task failed — stale credential, no re-auth attempt."
    ),
    "compliance-003": (
        "Task: Submit the compliance report using the correct v3 endpoint.\n"
        "Agent discovered /api/v3/compliance/reports and sent a POST.\n"
        "Response: 403 Forbidden — the agent's service account lacks the\n"
        "'compliance:write' permission scope.\n"
        "Agent logged the 403 but took no further action (did not escalate\n"
        "or request permission elevation).\n"
        "Result: Task failed — permission denied, no escalation."
    ),
}

CHAINED_KEYPOINTS = [
    "task_comprehension",
    "action_selection",
    "error_recovery",
    "tool_usage",
    "state_management",
]


@skip_no_claude
@integration
class TestChainedPipeline:
    """Run Locate → Plan → TaskEvaluate as a real chain.

    Each stage receives the *actual* output of the previous stage —
    no synthetic inputs injected between them. This proves the stages
    compose end-to-end through the real Claude CLI.
    """

    @pytest.mark.asyncio
    async def test_chained_locate_plan_evaluate(
        self, runner: ClaudeCodeRunner, failure_batch: Batch, workdir: Path
    ) -> None:
        import json
        from moss.models.keypoint import KeypointMatrix

        chain_dir = workdir / "chained"
        chain_dir.mkdir(exist_ok=True)

        # ---- Stage 1: Locate ------------------------------------------------
        print("\n" + "=" * 70)
        print("CHAINED PIPELINE — Stage 1: LOCATE")
        print("=" * 70)

        locate = LocateStage(runner=runner)
        locate_result = await locate.run(
            batch=failure_batch,
            transcripts=BASELINE_TRANSCRIPTS,
            workdir=chain_dir,
        )

        print(locate_result.output)
        assert locate_result.success, f"Locate failed: {locate_result.error}"
        assert len(locate_result.output) > 100, "Locate diagnosis too short"

        diagnosis = locate_result.output
        (chain_dir / "1-locate.md").write_text(diagnosis, encoding="utf-8")

        # ---- Stage 2: Plan (fed by real Locate output) -----------------------
        print("\n" + "=" * 70)
        print("CHAINED PIPELINE — Stage 2: PLAN")
        print("=" * 70)

        plan = PlanStage(runner=runner)
        plan_result = await plan.run(
            diagnosis=diagnosis,
            workdir=chain_dir,
        )

        print(plan_result.output)
        assert plan_result.success, f"Plan failed: {plan_result.error}"
        assert len(plan_result.output) > 100, "Plan fix spec too short"

        fix_spec = plan_result.output
        (chain_dir / "2-plan.md").write_text(fix_spec, encoding="utf-8")

        # ---- Stage 3: TaskEvaluate (uses same transcripts) -------------------
        print("\n" + "=" * 70)
        print("CHAINED PIPELINE — Stage 3: TASK EVALUATE")
        print("=" * 70)

        evaluate = TaskEvaluateStage(runner=runner)
        eval_result = await evaluate.run(
            transcripts=CHAINED_TRANSCRIPTS,
            keypoint_names=CHAINED_KEYPOINTS,
            workdir=chain_dir,
        )

        print(eval_result.output)
        assert eval_result.success, f"TaskEvaluate failed: {eval_result.error}"
        assert "keypoint_matrix" in eval_result.metadata, (
            "TaskEvaluate did not produce a keypoint matrix"
        )

        matrix_data = eval_result.metadata["keypoint_matrix"]
        matrix = KeypointMatrix(**matrix_data)
        (chain_dir / "3-task-evaluate.json").write_text(
            json.dumps(matrix_data, indent=2), encoding="utf-8"
        )

        # ---- Verify the full chain produced coherent results -----------------
        print("\n" + "=" * 70)
        print("CHAINED PIPELINE — SUMMARY")
        print("=" * 70)

        score = matrix.aggregate_score()
        print(f"Aggregate keypoint score: {score:.4f}")
        print(f"Tasks scored: {list(matrix.tasks.keys())}")
        for task_id, kps in matrix.tasks.items():
            scores_str = ", ".join(f"{kp.name}={kp.score.value}" for kp in kps)
            print(f"  {task_id}: {scores_str}")

        # The score should be low — these are failure transcripts
        assert score < 0.8, (
            f"Score {score:.2f} is suspiciously high for failure transcripts"
        )
        assert len(matrix.tasks) >= 2, (
            f"Expected at least 2 tasks scored, got {len(matrix.tasks)}"
        )

        # Verify artifacts are all on disk
        assert (chain_dir / "1-locate.md").exists()
        assert (chain_dir / "2-plan.md").exists()
        assert (chain_dir / "3-task-evaluate.json").exists()

        print(f"\nArtifacts saved to: {chain_dir}")
        print("CHAINED PIPELINE — PASSED")
        print("=" * 70)
