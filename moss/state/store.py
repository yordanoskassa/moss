"""Read/write JSONL state files with atomic operations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from moss.models.batch import Batch, Chunk
from moss.models.evolution import EvolutionState
from moss.models.keypoint import KeypointMatrix
from moss.state.paths import MossPaths


class StateStore:
    """File-based state persistence using JSONL and JSON."""

    def __init__(self, paths: MossPaths) -> None:
        self.paths = paths

    # -- Atomic file writes --

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write content atomically using a temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, content.encode())
            os.close(fd)
            os.rename(tmp, path)
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def _atomic_write_json(path: Path, data: dict | BaseModel) -> None:
        if isinstance(data, BaseModel):
            content = data.model_dump_json(indent=2)
        else:
            content = json.dumps(data, indent=2, default=str)
        StateStore._atomic_write(path, content)

    # -- JSONL operations --

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read all lines from a JSONL file."""
        if not path.exists():
            return []
        lines: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines

    @staticmethod
    def append_jsonl(path: Path, record: dict[str, Any] | BaseModel) -> None:
        """Append a single record to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(record, BaseModel):
            data = record.model_dump(mode="json")
        else:
            data = record
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    # -- Batch operations --

    def save_batch(self, batch: Batch) -> None:
        if batch.is_sealed:
            path = self.paths.batch_file(batch.id)
        else:
            path = self.paths.open_batch_file(batch.conversation_id)
        self._atomic_write_json(path, batch)

    def load_batch(self, batch_id: str) -> Batch | None:
        path = self.paths.batch_file(batch_id)
        if not path.exists():
            return None
        with open(path) as f:
            return Batch.model_validate_json(f.read())

    def list_batches(self) -> list[Batch]:
        batches: list[Batch] = []
        for path in self.paths.batches_dir.glob("batch-*.jsonl"):
            with open(path) as f:
                batches.append(Batch.model_validate_json(f.read()))
        return sorted(batches, key=lambda b: b.created_at)

    def seal_batch(self, batch: Batch, batch_id: str) -> Batch:
        """Seal an open batch and move it to the sealed location."""
        batch.id = batch_id
        batch.seal()
        self.save_batch(batch)
        # Remove old open batch file
        open_path = self.paths.open_batch_file(batch.conversation_id)
        if open_path.exists():
            open_path.unlink()
        return batch

    # -- Cursor operations --

    def read_cursor(self, session_id: str) -> int:
        path = self.paths.cursor_file(session_id)
        if not path.exists():
            return 0
        with open(path) as f:
            data = json.load(f)
        return data.get("position", 0)

    def write_cursor(self, session_id: str, position: int) -> None:
        path = self.paths.cursor_file(session_id)
        self._atomic_write_json(path, {"session_id": session_id, "position": position})

    # -- Evolution state --

    def save_evolution_state(self, evo_id: str, state: EvolutionState) -> None:
        path = self.paths.evolution_state_file(evo_id)
        self._atomic_write_json(path, state)

    def load_evolution_state(self, evo_id: str) -> EvolutionState | None:
        path = self.paths.evolution_state_file(evo_id)
        if not path.exists():
            return None
        with open(path) as f:
            return EvolutionState.model_validate_json(f.read())

    def list_evolutions(self) -> list[str]:
        if not self.paths.evolutions_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.paths.evolutions_dir.iterdir()
            if d.is_dir() and d.name.startswith("evo-")
        )

    # -- Keypoint matrix --

    def save_keypoints(self, evo_id: str, iteration: int | None, matrix: KeypointMatrix) -> None:
        if iteration is None:
            directory = self.paths.baseline_dir(evo_id)
        else:
            directory = self.paths.iteration_dir(evo_id, iteration)
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(directory / "keypoints.json", matrix)

    def load_keypoints(self, evo_id: str, iteration: int | None) -> KeypointMatrix | None:
        if iteration is None:
            path = self.paths.baseline_dir(evo_id) / "keypoints.json"
        else:
            path = self.paths.iteration_dir(evo_id, iteration) / "keypoints.json"
        if not path.exists():
            return None
        with open(path) as f:
            return KeypointMatrix.model_validate_json(f.read())

    # -- Stage artifacts --

    def save_stage_artifact(
        self, evo_id: str, iteration: int, filename: str, content: str
    ) -> Path:
        directory = self.paths.iteration_dir(evo_id, iteration)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        self._atomic_write(path, content)
        return path

    def load_stage_artifact(self, evo_id: str, iteration: int, filename: str) -> str | None:
        path = self.paths.iteration_dir(evo_id, iteration) / filename
        if not path.exists():
            return None
        return path.read_text()

    # -- Swap --

    def write_swap_request(self, image_tag: str) -> Path:
        """Write a swap-request file for the swap supervisor to pick up."""
        import time

        filename = f"swap-{int(time.time())}.json"
        path = self.paths.swap_requests_dir / filename
        self._atomic_write_json(path, {"image_tag": image_tag, "requested_at": time.time()})
        return path

    def read_last_known_good(self) -> str | None:
        path = self.paths.last_known_good
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get("image_tag")

    def write_last_known_good(self, image_tag: str) -> None:
        self._atomic_write_json(self.paths.last_known_good, {"image_tag": image_tag})
