"""Stage 6: Keypoint scoring (strong/adequate/weak/missing)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moss.models.keypoint import Keypoint, KeypointMatrix, KeypointScore
from moss.runner.base import Runner, StageResult


TASK_EVALUATE_PROMPT_TEMPLATE = """\
You are evaluating an autonomous agent's performance on a set of tasks after a code change.

## Task Transcripts

{transcripts}

## Keypoints to Evaluate

{keypoint_names}

## Task

For each task, score every keypoint on this 4-level scale:
- **strong**: The agent fully and correctly handles this aspect.
- **adequate**: The agent handles this aspect with minor issues.
- **weak**: The agent partially handles this aspect with significant issues.
- **missing**: The agent does not handle this aspect at all.

Output a JSON object with this exact structure:
```json
{{
  "tasks": {{
    "<task_id>": [
      {{"name": "<keypoint_name>", "score": "<strong|adequate|weak|missing>"}},
      ...
    ],
    ...
  }}
}}
```

Be precise and evidence-based. Reference specific parts of the transcript for each score.
Output ONLY the JSON — no markdown fences, no commentary.
"""


@dataclass
class TaskEvaluateStage:
    """Stage 6: Score keypoints per task from trial transcripts."""

    runner: Runner

    async def run(
        self,
        transcripts: dict[str, str],
        keypoint_names: list[str],
        workdir: Path,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        transcripts_text = "\n\n---\n\n".join(
            f"### Task: {task_id}\n{transcript}"
            for task_id, transcript in transcripts.items()
        )

        keypoint_list = "\n".join(f"- {name}" for name in keypoint_names)

        prompt = TASK_EVALUATE_PROMPT_TEMPLATE.format(
            transcripts=transcripts_text,
            keypoint_names=keypoint_list,
        )

        result = await self.runner.invoke(
            stage="task_evaluate",
            prompt=prompt,
            workdir=workdir,
            context=context or {},
        )

        if result.success:
            matrix = self._parse_matrix(result.output)
            if matrix:
                result.metadata["keypoint_matrix"] = matrix.model_dump(mode="json")

        return result

    @staticmethod
    def _parse_matrix(output: str) -> KeypointMatrix | None:
        """Parse the JSON keypoint matrix from the runner output."""
        # Try to extract JSON from the output
        text = output.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the output
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        tasks: dict[str, list[Keypoint]] = {}
        raw_tasks = data.get("tasks", data)

        for task_id, keypoints in raw_tasks.items():
            kp_list: list[Keypoint] = []
            for kp in keypoints:
                try:
                    score = KeypointScore(kp["score"].lower())
                except (ValueError, KeyError):
                    score = KeypointScore.MISSING
                kp_list.append(Keypoint(name=kp.get("name", "unknown"), score=score))
            tasks[task_id] = kp_list

        return KeypointMatrix(tasks=tasks)
