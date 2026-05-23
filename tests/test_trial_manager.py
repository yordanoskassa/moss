"""Tests for trial manager container lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from moss.daemon.trial_manager import TrialManager, TrialResult


class TestTrialManager:
    def test_initial_status(self) -> None:
        manager = TrialManager()
        status = manager.status()
        assert status["active_workers"] == 0
        assert status["workers"] == []

    def test_destroy_all_empty(self) -> None:
        manager = TrialManager()
        assert manager.destroy_all() == 0

    @pytest.mark.asyncio
    async def test_run_single_trial_docker_error(self) -> None:
        """Trial should return error result when Docker isn't available."""
        manager = TrialManager()
        mock_client = MagicMock()
        mock_client.containers.run.side_effect = Exception("Docker not available")
        mock_client.networks.get.side_effect = Exception("not found")
        mock_client.networks.create.return_value = MagicMock()
        manager._client = mock_client

        result = await manager._run_single_trial(
            image_tag="test:latest",
            task={"id": "task-1", "command": "echo hello"},
            trial_num=1,
            timeout=10,
        )

        assert isinstance(result, TrialResult)
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_run_single_trial_success(self) -> None:
        """Test successful trial execution with mock Docker."""
        manager = TrialManager()

        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"Task completed successfully"

        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.networks.get.return_value = MagicMock()
        manager._client = mock_client

        result = await manager._run_single_trial(
            image_tag="test:latest",
            task={"id": "task-1", "command": "echo hello"},
            trial_num=1,
            timeout=60,
        )

        assert result.success is True
        assert result.exit_code == 0
        assert "Task completed" in result.transcript

    def test_ensure_network_creates(self) -> None:
        manager = TrialManager()
        mock_client = MagicMock()

        from docker.errors import NotFound
        mock_client.networks.get.side_effect = NotFound("not found")
        mock_client.networks.create.return_value = MagicMock()
        manager._client = mock_client

        manager._ensure_network()
        mock_client.networks.create.assert_called_once()

    def test_ensure_network_exists(self) -> None:
        manager = TrialManager()
        mock_client = MagicMock()
        mock_client.networks.get.return_value = MagicMock()
        manager._client = mock_client

        manager._ensure_network()
        mock_client.networks.create.assert_not_called()
