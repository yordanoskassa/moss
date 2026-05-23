"""Tests for the auto-scan engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moss.daemon.auto_scan import AutoScanEngine, BatchManager, SessionScanner
from moss.models.batch import Batch, Chunk
from moss.models.config import Settings
from moss.state.paths import MossPaths
from moss.state.store import StateStore


@pytest.fixture
def session_file(tmp_path: Path) -> Path:
    """Create a sample session JSONL file."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    session_file = session_dir / "test-session.jsonl"

    lines = [
        {"role": "user", "content": "Do task A"},
        {"role": "assistant", "content": "Starting task A..."},
        {"role": "user", "content": "Also do task B"},
        {"role": "assistant", "content": "Now working on task B..."},
        {"role": "user", "content": "Check status"},
        {"role": "assistant", "content": "Status: both tasks in progress"},
    ]

    with open(session_file, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return session_file


class TestSessionScanner:
    def test_scan_from_zero(self, store: StateStore, session_file: Path) -> None:
        scanner = SessionScanner(store)
        chunks = scanner.scan(session_file, "test-session")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_scan_advances_cursor(self, store: StateStore, session_file: Path) -> None:
        scanner = SessionScanner(store)
        scanner.scan(session_file, "test-session")
        cursor = store.read_cursor("test-session")
        assert cursor > 0

    def test_scan_from_cursor(self, store: StateStore, session_file: Path) -> None:
        scanner = SessionScanner(store)

        # First scan
        chunks1 = scanner.scan(session_file, "test-session")

        # Second scan should return nothing (no new lines)
        chunks2 = scanner.scan(session_file, "test-session")
        assert len(chunks2) == 0

    def test_scan_nonexistent_file(self, store: StateStore) -> None:
        scanner = SessionScanner(store)
        chunks = scanner.scan(Path("/nonexistent.jsonl"), "fake")
        assert chunks == []

    def test_scan_with_new_lines_appended(
        self, store: StateStore, session_file: Path
    ) -> None:
        scanner = SessionScanner(store)

        # First scan
        scanner.scan(session_file, "test-session")

        # Append new lines
        with open(session_file, "a") as f:
            f.write(json.dumps({"role": "user", "content": "New message"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "Reply"}) + "\n")

        # Second scan should pick up new lines
        chunks = scanner.scan(session_file, "test-session")
        assert len(chunks) > 0


class TestBatchManager:
    def test_add_chunks_creates_batch(
        self,
        store: StateStore,
        paths: MossPaths,
        settings: Settings,
        sample_chunks: list[Chunk],
    ) -> None:
        manager = BatchManager(store, paths, settings)
        sealed = manager.add_chunks("conv-1", sample_chunks)
        # With default threshold of 8, 2 chunks shouldn't seal
        assert len(sealed) == 0

    def test_sealing_at_threshold(
        self,
        store: StateStore,
        paths: MossPaths,
    ) -> None:
        settings = Settings()
        settings.batch.chunk_threshold = 3

        manager = BatchManager(store, paths, settings)

        chunks = [
            Chunk(
                session_id=f"s{i}",
                cursor_start=i * 10,
                cursor_end=(i + 1) * 10,
                content=f"Content {i}",
            )
            for i in range(5)
        ]

        sealed = manager.add_chunks("conv-1", chunks)
        assert len(sealed) == 1
        assert sealed[0].chunk_count == 3
        assert sealed[0].is_sealed


class TestAutoScanEngine:
    def test_catch_up(
        self,
        store: StateStore,
        paths: MossPaths,
        settings: Settings,
        session_file: Path,
    ) -> None:
        engine = AutoScanEngine(store, paths, settings)
        result = engine.catch_up(session_file.parent)
        assert result["sessions_scanned"] >= 1
        assert result["chunks_added"] >= 0

    def test_flag_single_session(
        self,
        store: StateStore,
        paths: MossPaths,
        settings: Settings,
        session_file: Path,
    ) -> None:
        engine = AutoScanEngine(store, paths, settings)
        result = engine.flag("test-session", session_file)
        assert "chunks_added" in result
