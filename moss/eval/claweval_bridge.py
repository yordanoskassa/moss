"""Bridge between ClawEval results and MOSS data structures."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from moss.models.batch import Batch, Chunk
from moss.models.keypoint import Keypoint, KeypointMatrix, KeypointScore

# ClawEval's 7 categories mapped to MOSS keypoint names.
CATEGORY_TO_KEYPOINT: dict[str, str] = {
    "tool_calling": "action_execution",
    "coding": "action_selection",
    "reasoning": "task_comprehension",
    "writing": "output_quality",
    "research": "goal_completion",
    "memory": "state_management",
    "speed": "error_recovery",
}

# All MOSS keypoint names used by the bridge.
BRIDGE_KEYPOINTS: list[str] = sorted(set(CATEGORY_TO_KEYPOINT.values()))

# Pre-sorted by length descending so longer prefixes match first.
_CATEGORIES_BY_LENGTH = sorted(CATEGORY_TO_KEYPOINT.keys(), key=len, reverse=True)


def _extract_category(task_id: str) -> str:
    """Extract the category from a task ID like 'tool_calling_001'.

    Matches against known category names (longest first) so that
    'tool_calling' is matched before 'tool'.
    """
    for cat in _CATEGORIES_BY_LENGTH:
        if task_id.startswith(cat):
            return cat
    return "unknown"


def _score_to_keypoint_score(score: float) -> KeypointScore:
    """Convert a 0–1 numeric score to a MOSS KeypointScore enum."""
    if score >= 0.8:
        return KeypointScore.STRONG
    if score >= 0.6:
        return KeypointScore.ADEQUATE
    if score >= 0.3:
        return KeypointScore.WEAK
    return KeypointScore.MISSING


class ClawEvalBridge:
    """Converts ClawEval result data into MOSS Batch and KeypointMatrix objects."""

    def __init__(self, results_path: Path | str | None = None) -> None:
        self._raw: dict[str, Any] = {}
        if results_path is not None:
            self.load(results_path)

    def load(self, path: Path | str) -> None:
        """Load a ClawEval results JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ClawEval results not found: {path}")
        with open(path) as f:
            self._raw = json.load(f)

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load results directly from a dictionary (e.g. from runner output)."""
        self._raw = data

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def to_batch(
        self,
        model_id: str | None = None,
        score_threshold: float = 1.0,
    ) -> Batch:
        """Build a sealed MOSS Batch from ClawEval results.

        Args:
            model_id: Which model's results to use.  If *None*, uses the first
                       model found in the results file.
            score_threshold: Only include tasks scoring strictly below this
                             value.  Default 1.0 includes all tasks.

        Returns:
            A sealed ``Batch`` whose chunks are the failure/low-scoring task
            transcripts.
        """
        model_data = self._resolve_model(model_id)
        tasks: list[dict[str, Any]] = model_data.get("tasks", [])

        chunks: list[Chunk] = []
        for idx, task in enumerate(tasks):
            score_block = task.get("score") or {}
            total_score = score_block.get("total_score", 0.0)
            if total_score >= score_threshold:
                continue

            # Build chunk content from the task's response and tool calls.
            content_parts: list[str] = [f"Task: {task.get('task_id', 'unknown')}"]
            if task.get("error"):
                content_parts.append(f"Error: {task['error']}")
            if task.get("response_text"):
                content_parts.append(f"Response: {task['response_text'][:2000]}")
            if task.get("tool_calls_made"):
                content_parts.append(
                    f"Tool calls: {json.dumps(task['tool_calls_made'], indent=2)[:1000]}"
                )
            content_parts.append(f"Score: {total_score:.3f}")
            if score_block.get("breakdown"):
                content_parts.append(
                    f"Breakdown: {json.dumps(score_block['breakdown'])}"
                )

            task_category = _extract_category(task.get("task_id", ""))
            keypoint_name = CATEGORY_TO_KEYPOINT.get(task_category, "goal_completion")

            chunks.append(
                Chunk(
                    session_id=f"claweval-{task.get('task_id', idx)}",
                    cursor_start=0,
                    cursor_end=len("\n".join(content_parts)),
                    content="\n".join(content_parts),
                    keypoints=[keypoint_name],
                )
            )

        batch_id = f"claweval-{uuid.uuid4().hex[:8]}"
        batch = Batch(id=batch_id, chunks=chunks, conversation_id="claweval-run")
        batch.seal()
        return batch

    # ------------------------------------------------------------------
    # KeypointMatrix construction
    # ------------------------------------------------------------------

    def to_keypoint_matrix(self, model_id: str | None = None) -> KeypointMatrix:
        """Build a MOSS KeypointMatrix from ClawEval per-task scores.

        Each ClawEval task becomes a MOSS "task" entry.  The task's category
        determines which keypoint name is assigned, and the task's total_score
        is mapped to a ``KeypointScore`` via thresholds.
        """
        model_data = self._resolve_model(model_id)
        tasks: list[dict[str, Any]] = model_data.get("tasks", [])

        matrix_tasks: dict[str, list[Keypoint]] = {}
        for task in tasks:
            task_id = task.get("task_id", "unknown")
            score_block = task.get("score") or {}
            total_score = score_block.get("total_score", 0.0)

            # Determine the primary keypoint from the task's category.
            category = _extract_category(task_id)
            primary_kp_name = CATEGORY_TO_KEYPOINT.get(category, "goal_completion")
            primary_kp = Keypoint(
                name=primary_kp_name,
                score=_score_to_keypoint_score(total_score),
            )

            # Also generate keypoints from the score breakdown if present.
            keypoints = [primary_kp]
            breakdown = score_block.get("breakdown", {})
            if breakdown:
                for component, value in breakdown.items():
                    kp_name = f"{category}_{component}"
                    keypoints.append(
                        Keypoint(
                            name=kp_name,
                            score=_score_to_keypoint_score(value),
                        )
                    )

            matrix_tasks[task_id] = keypoints

        return KeypointMatrix(tasks=matrix_tasks)

    def category_scores(self, model_id: str | None = None) -> dict[str, float]:
        """Return the per-category aggregate scores from the results."""
        model_data = self._resolve_model(model_id)
        return dict(model_data.get("categories", {}))

    def overall_score(self, model_id: str | None = None) -> float:
        """Return the overall aggregate score from the results."""
        model_data = self._resolve_model(model_id)
        return float(model_data.get("overall", 0.0))

    # ------------------------------------------------------------------
    # Transcript extraction
    # ------------------------------------------------------------------

    def transcripts(self, model_id: str | None = None) -> dict[str, str]:
        """Extract per-task transcripts suitable for MOSS pipeline stages.

        Returns a dict mapping task_id → transcript text.
        """
        model_data = self._resolve_model(model_id)
        result: dict[str, str] = {}
        for task in model_data.get("tasks", []):
            task_id = task.get("task_id", "unknown")
            parts = [f"Task ID: {task_id}"]
            if task.get("response_text"):
                parts.append(f"Model response:\n{task['response_text']}")
            if task.get("tool_calls_made"):
                parts.append(
                    f"Tool calls:\n{json.dumps(task['tool_calls_made'], indent=2)}"
                )
            if task.get("error"):
                parts.append(f"Error: {task['error']}")
            score_block = task.get("score") or {}
            if score_block:
                parts.append(f"Score: {score_block.get('total_score', 0.0):.3f}")
            result[task_id] = "\n\n".join(parts)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, model_id: str | None) -> dict[str, Any]:
        """Look up model data in the raw results dict."""
        models = self._raw.get("models", {})
        if not models:
            raise ValueError("No model results found in ClawEval data")
        if model_id is not None:
            if model_id not in models:
                raise KeyError(
                    f"Model '{model_id}' not found.  Available: {list(models.keys())}"
                )
            return models[model_id]
        # Default to first model.
        return next(iter(models.values()))
