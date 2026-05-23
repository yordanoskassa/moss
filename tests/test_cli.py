"""Tests for CLI subcommand routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from moss.cli.main import app

runner = CliRunner()


class TestCLIRouting:
    """Test that CLI subcommands are properly registered and route correctly."""

    def test_app_has_evo_subgroup(self) -> None:
        """The 'evo' command group should exist."""
        result = runner.invoke(app, ["evo", "--help"])
        assert result.exit_code == 0
        assert "Evolution control commands" in result.output

    def test_status_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "status", "--help"])
        assert result.exit_code == 0

    def test_batches_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "batches", "--help"])
        assert result.exit_code == 0

    def test_batch_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "batch", "--help"])
        assert result.exit_code == 0

    def test_start_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "start", "--help"])
        assert result.exit_code == 0

    def test_stop_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "stop", "--help"])
        assert result.exit_code == 0

    def test_restart_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "restart", "--help"])
        assert result.exit_code == 0

    def test_apply_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "apply", "--help"])
        assert result.exit_code == 0

    def test_flag_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "flag", "--help"])
        assert result.exit_code == 0

    def test_catch_up_command_exists(self) -> None:
        result = runner.invoke(app, ["evo", "catch-up", "--help"])
        assert result.exit_code == 0

    def test_start_requires_batch_id(self) -> None:
        """Start should fail without a batch_id argument."""
        result = runner.invoke(app, ["evo", "start"])
        assert result.exit_code != 0

    def test_flag_requires_session_id(self) -> None:
        """Flag should fail without a session_id argument."""
        result = runner.invoke(app, ["evo", "flag"])
        assert result.exit_code != 0
