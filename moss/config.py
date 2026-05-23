"""Load and parse moss.toml configuration."""

from __future__ import annotations

from pathlib import Path

import tomli

from moss.models.config import (
    BatchConfig,
    DaemonConfig,
    DepthConfig,
    GatewayConfig,
    RunnerConfig,
    Settings,
    SwapConfig,
)

_DEFAULT_LOCATIONS = [
    Path("moss.toml"),
    Path.home() / ".moss" / "moss.toml",
]


def find_config_file(explicit: Path | None = None) -> Path | None:
    """Find the moss.toml config file."""
    if explicit and explicit.exists():
        return explicit
    for loc in _DEFAULT_LOCATIONS:
        if loc.exists():
            return loc
    return None


def load_config(path: Path | None = None) -> Settings:
    """Load settings from moss.toml, falling back to defaults."""
    config_path = find_config_file(path)
    if config_path is None:
        return Settings()

    with open(config_path, "rb") as f:
        raw = tomli.load(f)

    moss_section = raw.get("moss", {})
    home = Path(moss_section.get("home", "~/.moss")).expanduser()
    substrate = moss_section.get("substrate", "openclaw")

    depth: dict[str, DepthConfig] = {}
    for dial_name in ("light", "standard", "deep"):
        if dial_name in raw.get("depth", {}):
            d = raw["depth"][dial_name]
            depth[dial_name] = DepthConfig(**d)

    batch_raw = raw.get("batch", {})
    batch = BatchConfig(**batch_raw) if batch_raw else BatchConfig()

    runner_default = raw.get("runner", {}).get("default", "claude-code")
    runner_cc_raw = raw.get("runner", {}).get("claude-code", {})
    runner_claude_code = RunnerConfig(**runner_cc_raw) if runner_cc_raw else RunnerConfig()

    swap_raw = raw.get("swap", {})
    swap = SwapConfig(**swap_raw) if swap_raw else SwapConfig()

    gw_raw = raw.get("gateway", {})
    gateway = GatewayConfig(**gw_raw) if gw_raw else GatewayConfig()

    daemon_raw = raw.get("daemon", {})
    daemon = DaemonConfig(**daemon_raw) if daemon_raw else DaemonConfig()

    return Settings(
        home=home,
        substrate=substrate,
        depth=depth,
        batch=batch,
        runner_default=runner_default,
        runner_claude_code=runner_claude_code,
        swap=swap,
        gateway=gateway,
        daemon=daemon,
    )
