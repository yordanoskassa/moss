"""Evolution orchestrator: baseline → iterate → verdict loop."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moss.models.batch import Batch
from moss.models.config import DepthConfig, DepthDial, Settings
from moss.models.evolution import EvolutionState, EvolutionStatus
from moss.models.keypoint import KeypointMatrix
from moss.models.verdict import Verdict
from moss.pipeline.stage_runner import StageRunner
from moss.pipeline.stages.code_review import CodeReviewStage
from moss.pipeline.stages.implement import ImplementStage
from moss.pipeline.stages.locate import LocateStage
from moss.pipeline.stages.plan import PlanStage
from moss.pipeline.stages.plan_review import PlanReviewStage
from moss.pipeline.stages.task_evaluate import TaskEvaluateStage
from moss.pipeline.stages.verdict import VerdictStage
from moss.runner.base import Runner
from moss.state.paths import MossPaths
from moss.state.store import StateStore

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the evolution loop: baseline → iterate until terminal verdict."""

    def __init__(
        self,
        runner: Runner,
        store: StateStore,
        paths: MossPaths,
        settings: Settings,
    ) -> None:
        self.runner = runner
        self.store = store
        self.paths = paths
        self.settings = settings
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        """Signal the orchestrator to stop after the current iteration."""
        self._stop_event.set()

    async def run_evolution(
        self,
        batch: Batch,
        depth: DepthDial = DepthDial.STANDARD,
        workdir: Path | None = None,
    ) -> EvolutionState:
        """Run a complete evolution cycle for a batch."""
        evo_id = f"evo-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        depth_config = self.settings.depth_for(depth)

        state = EvolutionState(
            batch_id=batch.id,
            max_iterations=depth_config.max_iterations,
            depth=depth.value,
            status=EvolutionStatus.RUNNING,
        )

        # Ensure directories exist
        self.paths.evolution_dir(evo_id).mkdir(parents=True, exist_ok=True)
        self.store.save_evolution_state(evo_id, state)

        if workdir is None:
            workdir = Path.cwd()

        stage_runner = StageRunner(
            runner=self.runner,
            store=self.store,
            evo_id=evo_id,
        )

        try:
            # Phase 1: Baseline scoring
            baseline = await self._run_baseline(
                batch, evo_id, stage_runner, workdir, depth_config
            )
            self.store.save_keypoints(evo_id, None, baseline)

            # Phase 2: Iterative improvement
            plateau_count = 0
            best_matrix = baseline

            for iteration in range(1, depth_config.max_iterations + 1):
                if self._stop_event.is_set():
                    state.status = EvolutionStatus.STOPPED
                    break

                state.advance_iteration()
                self.store.save_evolution_state(evo_id, state)

                logger.info("Starting iteration %d/%d", iteration, depth_config.max_iterations)

                verdict, current_matrix = await self._run_iteration(
                    batch=batch,
                    evo_id=evo_id,
                    iteration=iteration,
                    baseline=baseline,
                    best_matrix=best_matrix,
                    plateau_count=plateau_count,
                    stage_runner=stage_runner,
                    workdir=workdir,
                    depth_config=depth_config,
                )

                state.verdict = verdict
                self.store.save_evolution_state(evo_id, state)

                # Track plateau
                if current_matrix and current_matrix.improvement_delta(best_matrix) < 0.01:
                    plateau_count += 1
                else:
                    plateau_count = 0
                    if current_matrix:
                        best_matrix = current_matrix

                if verdict.is_terminal:
                    break

            # Final status
            if state.status != EvolutionStatus.STOPPED:
                if state.verdict == Verdict.CONVERGED:
                    state.status = EvolutionStatus.COMPLETED
                elif state.verdict and state.verdict.is_terminal:
                    state.status = EvolutionStatus.COMPLETED
                else:
                    state.status = EvolutionStatus.COMPLETED

        except Exception as e:
            logger.exception("Evolution %s failed", evo_id)
            state.status = EvolutionStatus.FAILED
            state.error = str(e)

        state.updated_at = datetime.now(timezone.utc)
        self.store.save_evolution_state(evo_id, state)

        return state

    async def _run_baseline(
        self,
        batch: Batch,
        evo_id: str,
        stage_runner: StageRunner,
        workdir: Path,
        depth_config: DepthConfig,
    ) -> KeypointMatrix:
        """Score pre-captured transcripts to lock the keypoint set."""
        logger.info("Running baseline evaluation")

        # Collect transcripts from batch chunks
        transcripts: dict[str, str] = {}
        for chunk in batch.chunks:
            tid = chunk.session_id
            if tid not in transcripts:
                transcripts[tid] = chunk.content
            else:
                transcripts[tid] += "\n" + chunk.content

        # Determine keypoints from the first evaluation
        keypoint_names = [
            "task_comprehension",
            "action_selection",
            "action_execution",
            "error_recovery",
            "goal_completion",
            "state_management",
            "output_quality",
        ]

        evaluate_stage = TaskEvaluateStage(runner=self.runner)
        result = await evaluate_stage.run(
            transcripts=transcripts,
            keypoint_names=keypoint_names,
            workdir=workdir,
        )

        if result.success and "keypoint_matrix" in result.metadata:
            matrix = KeypointMatrix.model_validate(result.metadata["keypoint_matrix"])
        else:
            # Create a minimal baseline matrix
            from moss.models.keypoint import Keypoint, KeypointScore

            tasks = {}
            for tid in transcripts:
                tasks[tid] = [
                    Keypoint(name=name, score=KeypointScore.WEAK)
                    for name in keypoint_names
                ]
            matrix = KeypointMatrix(tasks=tasks)

        self.store.save_stage_artifact(
            evo_id, 0, "baseline-eval.md", result.output
        )

        return matrix

    async def _run_iteration(
        self,
        batch: Batch,
        evo_id: str,
        iteration: int,
        baseline: KeypointMatrix,
        best_matrix: KeypointMatrix,
        plateau_count: int,
        stage_runner: StageRunner,
        workdir: Path,
        depth_config: DepthConfig,
    ) -> tuple[Verdict, KeypointMatrix | None]:
        """Execute all 7 stages for one iteration."""

        iter_dir = self.paths.iteration_dir(evo_id, iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Collect transcripts
        transcripts = "\n\n".join(c.content for c in batch.chunks)

        # Stage 1: Locate
        locate_stage = LocateStage(runner=self.runner)
        locate_result = await locate_stage.run(
            batch=batch, transcripts=transcripts, workdir=workdir
        )
        self.store.save_stage_artifact(evo_id, iteration, "locate.md", locate_result.output)

        if not locate_result.success:
            return Verdict.NEED_MORE_WORK, None

        # Stage 2: Plan
        plan_stage = PlanStage(runner=self.runner)
        plan_result = await plan_stage.run(
            diagnosis=locate_result.output, workdir=workdir
        )
        self.store.save_stage_artifact(evo_id, iteration, "plan.md", plan_result.output)

        if not plan_result.success:
            return Verdict.NEED_MORE_WORK, None

        # Stage 3: Plan Review (multi-round)
        review_stage = PlanReviewStage(
            runner=self.runner, max_rounds=depth_config.stage_round_budget
        )
        review_result = await review_stage.run(
            diagnosis=locate_result.output,
            plan=plan_result.output,
            workdir=workdir,
        )
        self.store.save_stage_artifact(evo_id, iteration, "plan-review.md", review_result.output)

        # If plan rejected after all rounds, still continue with best available plan
        plan_text = plan_result.output

        # Stage 4: Implement
        impl_stage = ImplementStage(runner=self.runner)
        impl_result = await impl_stage.run(plan=plan_text, workdir=workdir)
        self.store.save_stage_artifact(evo_id, iteration, "implement.md", impl_result.output)

        if not impl_result.success:
            return Verdict.NEED_MORE_WORK, None

        # Capture diff
        diff = await self._get_git_diff(workdir)
        self.store.save_stage_artifact(evo_id, iteration, "diff.patch", diff)

        # Stage 5: Code Review (multi-round)
        cr_stage = CodeReviewStage(
            runner=self.runner, max_rounds=depth_config.stage_round_budget
        )
        cr_result = await cr_stage.run(
            plan=plan_text,
            implementation=impl_result.output,
            diff=diff,
            workdir=workdir,
        )
        self.store.save_stage_artifact(evo_id, iteration, "code-review.md", cr_result.output)

        # Stage 6: Task Evaluate
        # In a real run, we'd run trials here. For now, re-evaluate with current code.
        task_transcripts: dict[str, str] = {}
        for chunk in batch.chunks:
            tid = chunk.session_id
            if tid not in task_transcripts:
                task_transcripts[tid] = chunk.content
            else:
                task_transcripts[tid] += "\n" + chunk.content

        eval_stage = TaskEvaluateStage(runner=self.runner)
        eval_result = await eval_stage.run(
            transcripts=task_transcripts,
            keypoint_names=[kp.name for kp in next(iter(baseline.tasks.values()), [])],
            workdir=workdir,
        )
        self.store.save_stage_artifact(evo_id, iteration, "task-evaluate.md", eval_result.output)

        current_matrix: KeypointMatrix | None = None
        if eval_result.success and "keypoint_matrix" in eval_result.metadata:
            current_matrix = KeypointMatrix.model_validate(eval_result.metadata["keypoint_matrix"])
            self.store.save_keypoints(evo_id, iteration, current_matrix)

        # Stage 7: Verdict
        if current_matrix is None:
            return Verdict.NEED_MORE_WORK, None

        verdict_stage = VerdictStage(
            runner=self.runner, plateau_threshold=depth_config.plateau_threshold
        )
        verdict_result = await verdict_stage.run(
            baseline=baseline,
            current=current_matrix,
            plateau_count=plateau_count,
            workdir=workdir,
        )
        self.store.save_stage_artifact(evo_id, iteration, "verdict.md", verdict_result.output)

        # Save verdict JSON
        verdict_data = {
            "verdict": verdict_result.metadata.get("verdict", "need_more_work"),
            "baseline_score": verdict_result.metadata.get("baseline_score"),
            "current_score": verdict_result.metadata.get("current_score"),
            "delta": verdict_result.metadata.get("delta"),
            "plateau_count": plateau_count,
        }
        self.store.save_stage_artifact(
            evo_id, iteration, "verdict.json", json.dumps(verdict_data, indent=2)
        )

        verdict_value = verdict_result.metadata.get("verdict", "need_more_work")
        try:
            verdict = Verdict(verdict_value)
        except ValueError:
            verdict = Verdict.NEED_MORE_WORK

        return verdict, current_matrix

    @staticmethod
    async def _get_git_diff(workdir: Path) -> str:
        """Get the git diff for the current changes."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "HEAD~1",
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode(errors="replace")
        except Exception:
            return "(git diff unavailable)"
