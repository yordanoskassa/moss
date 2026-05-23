"""Configuration models and depth dial."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class DepthDial(str, Enum):
    """Evolution depth dial controlling iteration budgets."""

    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def max_iterations(self) -> int:
        return {"light": 2, "standard": 5, "deep": 10}[self.value]

    @property
    def stage_round_budget(self) -> int:
        return {"light": 1, "standard": 3, "deep": 5}[self.value]

    @property
    def trials_per_task(self) -> int:
        return {"light": 1, "standard": 3, "deep": 5}[self.value]

    @property
    def plateau_threshold(self) -> int:
        return {"light": 1, "standard": 3, "deep": 5}[self.value]


class RunnerConfig(BaseModel):
    command: str = "claude"
    flags: list[str] = Field(default_factory=lambda: ["--print", "--dangerously-skip-permissions"])
    timeout: int = 300


class BatchConfig(BaseModel):
    chunk_threshold: int = 8


class SwapConfig(BaseModel):
    poll_interval: float = 2.0
    probe_window: float = 90.0
    probe_sample_interval: float = 5.0
    heartbeat_max_age: float = 30.0
    consecutive_passes_required: int = 3


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8420


class DaemonConfig(BaseModel):
    socket_path: str = "/tmp/moss-daemon.sock"
    auto_scan_interval: int = 3600


class DepthConfig(BaseModel):
    max_iterations: int
    stage_round_budget: int
    trials_per_task: int
    plateau_threshold: int


class Settings(BaseModel):
    """Top-level MOSS configuration loaded from moss.toml."""

    home: Path = Field(default_factory=lambda: Path.home() / ".moss")
    substrate: str = "openclaw"
    depth: dict[str, DepthConfig] = Field(default_factory=dict)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    runner_default: str = "claude-code"
    runner_claude_code: RunnerConfig = Field(default_factory=RunnerConfig)
    swap: SwapConfig = Field(default_factory=SwapConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)

    def depth_for(self, dial: DepthDial) -> DepthConfig:
        """Get depth config, falling back to dial's defaults."""
        if dial.value in self.depth:
            return self.depth[dial.value]
        return DepthConfig(
            max_iterations=dial.max_iterations,
            stage_round_budget=dial.stage_round_budget,
            trials_per_task=dial.trials_per_task,
            plateau_threshold=dial.plateau_threshold,
        )
