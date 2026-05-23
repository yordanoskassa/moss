"""Session JSONL scanner, chunker, and batch manager."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from moss.models.batch import Batch, BatchStatus, Chunk
from moss.models.config import Settings
from moss.state.paths import MossPaths
from moss.state.store import StateStore

logger = logging.getLogger(__name__)


class SessionScanner:
    """Reads session JSONL files from a cursor position, producing raw text slices."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def scan(self, session_path: Path, session_id: str) -> list[Chunk]:
        """Scan a session JSONL from the last cursor, producing chunks."""
        if not session_path.exists():
            return []

        cursor = self.store.read_cursor(session_id)
        lines = session_path.read_text().splitlines()

        if cursor >= len(lines):
            return []

        new_lines = lines[cursor:]
        chunks: list[Chunk] = []

        # Group lines into chunks of meaningful content
        chunk_lines: list[str] = []
        chunk_start = cursor

        for i, line in enumerate(new_lines):
            line = line.strip()
            if not line:
                continue

            chunk_lines.append(line)

            # Produce a chunk when we accumulate enough content
            # or when we detect a task boundary (assistant response)
            try:
                parsed = json.loads(line)
                is_boundary = parsed.get("role") == "assistant" and len(chunk_lines) > 1
            except (json.JSONDecodeError, AttributeError):
                is_boundary = False

            if is_boundary or len(chunk_lines) >= 20:
                chunk = Chunk(
                    session_id=session_id,
                    cursor_start=chunk_start,
                    cursor_end=cursor + i + 1,
                    content="\n".join(chunk_lines),
                )
                chunks.append(chunk)
                chunk_lines = []
                chunk_start = cursor + i + 1

        # Remaining lines
        if chunk_lines:
            chunk = Chunk(
                session_id=session_id,
                cursor_start=chunk_start,
                cursor_end=cursor + len(new_lines),
                content="\n".join(chunk_lines),
            )
            chunks.append(chunk)

        # Advance cursor
        self.store.write_cursor(session_id, cursor + len(new_lines))

        return chunks


class BatchManager:
    """Maintains open batches per conversation, seals at threshold."""

    def __init__(self, store: StateStore, paths: MossPaths, settings: Settings) -> None:
        self.store = store
        self.paths = paths
        self.chunk_threshold = settings.batch.chunk_threshold
        self._next_batch_id = self._compute_next_id()

    def _compute_next_id(self) -> int:
        existing = self.store.list_batches()
        if not existing:
            return 1
        ids = []
        for b in existing:
            try:
                ids.append(int(b.id.replace("batch-", "")))
            except ValueError:
                pass
        return max(ids, default=0) + 1

    def _allocate_id(self) -> str:
        bid = f"batch-{self._next_batch_id:03d}"
        self._next_batch_id += 1
        return bid

    def add_chunks(self, conversation_id: str, chunks: list[Chunk]) -> list[Batch]:
        """Add chunks to an open batch, sealing when threshold is reached.

        Returns list of any newly sealed batches.
        """
        if not chunks:
            return []

        # Load or create open batch
        open_path = self.paths.open_batch_file(conversation_id)
        if open_path.exists():
            with open(open_path) as f:
                batch = Batch.model_validate_json(f.read())
        else:
            batch = Batch(id="open", conversation_id=conversation_id)

        sealed: list[Batch] = []

        for chunk in chunks:
            batch.chunks.append(chunk)

            if batch.chunk_count >= self.chunk_threshold:
                batch_id = self._allocate_id()
                batch = self.store.seal_batch(batch, batch_id)
                sealed.append(batch)
                logger.info("Sealed batch %s with %d chunks", batch_id, batch.chunk_count)
                # Start a fresh open batch
                batch = Batch(id="open", conversation_id=conversation_id)

        # Save remaining open batch if it has chunks
        if batch.chunks:
            self.store.save_batch(batch)

        return sealed


class AutoScanEngine:
    """Coordinates scanning across all sessions."""

    def __init__(self, store: StateStore, paths: MossPaths, settings: Settings) -> None:
        self.scanner = SessionScanner(store)
        self.batch_manager = BatchManager(store, paths, settings)
        self.store = store
        self.paths = paths

    def catch_up(self, sessions_dir: Path | None = None) -> dict:
        """Scan all session JSONL files from their cursors."""
        if sessions_dir is None:
            sessions_dir = self.paths.open_batches_dir.parent

        total_sessions = 0
        total_chunks = 0

        # Look for session JSONL files
        for session_file in sorted(sessions_dir.glob("*.jsonl")):
            session_id = session_file.stem
            chunks = self.scanner.scan(session_file, session_id)
            if chunks:
                self.batch_manager.add_chunks(session_id, chunks)
                total_chunks += len(chunks)
            total_sessions += 1

        return {"sessions_scanned": total_sessions, "chunks_added": total_chunks}

    def flag(self, session_id: str, session_path: Path | None = None) -> dict:
        """Scan a single session from cursor to EOF."""
        if session_path is None:
            session_path = self.paths.open_batches_dir.parent / f"{session_id}.jsonl"

        chunks = self.scanner.scan(session_path, session_id)
        if chunks:
            self.batch_manager.add_chunks(session_id, chunks)

        return {"chunks_added": len(chunks)}
