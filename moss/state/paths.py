"""Standard directory layout for MOSS_HOME."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_HOME = Path.home() / ".moss"


class MossPaths:
    """Path helpers for the MOSS state directory."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = home or _DEFAULT_HOME

    # -- Top-level directories --

    @property
    def batches_dir(self) -> Path:
        return self.home / "batches"

    @property
    def open_batches_dir(self) -> Path:
        return self.batches_dir / "open"

    @property
    def cursors_dir(self) -> Path:
        return self.home / "cursors"

    @property
    def evolutions_dir(self) -> Path:
        return self.home / "evolutions"

    @property
    def swap_dir(self) -> Path:
        return self.home / "swap"

    @property
    def swap_requests_dir(self) -> Path:
        return self.swap_dir / "requests"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    # -- Specific files --

    @property
    def last_known_good(self) -> Path:
        return self.swap_dir / "last-known-good.json"

    @property
    def daemon_log(self) -> Path:
        return self.logs_dir / "daemon.log"

    @property
    def config_file(self) -> Path:
        return self.home / "moss.toml"

    # -- Batch paths --

    def batch_file(self, batch_id: str) -> Path:
        return self.batches_dir / f"{batch_id}.jsonl"

    def open_batch_file(self, session_id: str) -> Path:
        return self.open_batches_dir / f"{session_id}.jsonl"

    # -- Cursor paths --

    def cursor_file(self, session_id: str) -> Path:
        return self.cursors_dir / f"{session_id}.json"

    # -- Evolution paths --

    def evolution_dir(self, evo_id: str) -> Path:
        return self.evolutions_dir / evo_id

    def evolution_state_file(self, evo_id: str) -> Path:
        return self.evolution_dir(evo_id) / "state.json"

    def baseline_dir(self, evo_id: str) -> Path:
        return self.evolution_dir(evo_id) / "baseline"

    def iteration_dir(self, evo_id: str, iteration: int) -> Path:
        return self.evolution_dir(evo_id) / f"iter-{iteration}"

    # -- Ensure directories exist --

    def ensure_all(self) -> None:
        """Create all standard directories."""
        for d in [
            self.batches_dir,
            self.open_batches_dir,
            self.cursors_dir,
            self.evolutions_dir,
            self.swap_dir,
            self.swap_requests_dir,
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
