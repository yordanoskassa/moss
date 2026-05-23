"""Stage 1: Diagnosis from traces + batch failure chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from moss.models.batch import Batch
from moss.runner.base import Runner, StageResult


LOCATE_PROMPT_TEMPLATE = """\
You are analysing failure traces from an autonomous agent running against a task batch.

## Batch Failure Chunks

{chunks}

## Baseline Transcripts

{transcripts}

## Task

Produce a diagnosis document that:
1. Identifies which agent harness components are involved in each failure.
2. Groups failures by root cause (e.g., routing logic, hook dispatch, state management).
3. For each root-cause cluster, quotes the relevant trace lines and explains what went wrong.
4. Ranks clusters by severity (how many tasks they affect).

Output the diagnosis as structured markdown with clear section headers.
Do NOT propose fixes yet — that is a later stage.
"""


@dataclass
class LocateStage:
    """Stage 1: Locate failures in traces and produce a diagnosis."""

    runner: Runner

    async def run(
        self,
        batch: Batch,
        transcripts: str,
        workdir: Any,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        chunks_text = "\n\n---\n\n".join(
            f"### Chunk [{c.session_id}] (lines {c.cursor_start}-{c.cursor_end})\n{c.content}"
            for c in batch.chunks
        )

        prompt = LOCATE_PROMPT_TEMPLATE.format(
            chunks=chunks_text,
            transcripts=transcripts or "(no baseline transcripts available)",
        )

        return await self.runner.invoke(
            stage="locate",
            prompt=prompt,
            workdir=workdir,
            context=context or {},
        )
