"""Tests for swap supervisor health check and rollback logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from moss.daemon.swap_supervisor import HealthProbe, SwapSupervisor
from moss.models.config import SwapConfig
from moss.state.paths import MossPaths
from moss.state.store import StateStore


class TestHealthProbe:
    @pytest.mark.asyncio
    async def test_check_container_running_success(self) -> None:
        probe = HealthProbe()
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_client.containers.get.return_value = mock_container
        probe._client = mock_client

        result = await probe.check_container_running()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_container_running_stopped(self) -> None:
        probe = HealthProbe()
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_client.containers.get.return_value = mock_container
        probe._client = mock_client

        result = await probe.check_container_running()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_container_not_found(self) -> None:
        from docker.errors import NotFound

        probe = HealthProbe()
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = NotFound("not found")
        probe._client = mock_client

        result = await probe.check_container_running()
        assert result is False


class TestSwapSupervisor:
    def test_poll_for_request(
        self, store: StateStore, paths: MossPaths
    ) -> None:
        supervisor = SwapSupervisor(store=store, paths=paths)

        # No requests initially
        assert supervisor._poll_for_request() is None

        # Write a swap request
        store.write_swap_request("test-image:latest")

        # Should find it
        request = supervisor._poll_for_request()
        assert request is not None
        assert request["image_tag"] == "test-image:latest"

        # Should be consumed (file deleted)
        assert supervisor._poll_for_request() is None

    def test_last_known_good(self, store: StateStore, paths: MossPaths) -> None:
        # Initially no LKG
        assert store.read_last_known_good() is None

        # Write LKG
        store.write_last_known_good("moss-gateway:v1")
        assert store.read_last_known_good() == "moss-gateway:v1"

        # Update LKG
        store.write_last_known_good("moss-gateway:v2")
        assert store.read_last_known_good() == "moss-gateway:v2"

    @pytest.mark.asyncio
    async def test_commit_writes_lkg(
        self, store: StateStore, paths: MossPaths
    ) -> None:
        callback = AsyncMock()
        supervisor = SwapSupervisor(
            store=store, paths=paths, on_complete=callback
        )

        await supervisor._commit("new-image:v1")

        assert store.read_last_known_good() == "new-image:v1"
        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_rollback_with_lkg(
        self, store: StateStore, paths: MossPaths
    ) -> None:
        store.write_last_known_good("old-image:v1")

        callback = AsyncMock()
        mock_client = MagicMock()
        from docker.errors import DockerException
        mock_client.containers.get.side_effect = DockerException("not found")
        mock_client.containers.run.return_value = MagicMock()

        supervisor = SwapSupervisor(
            store=store, paths=paths, on_complete=callback
        )
        supervisor._client = mock_client

        await supervisor._rollback()

        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert result["status"] == "rolled-back"

    @pytest.mark.asyncio
    async def test_rollback_without_lkg(
        self, store: StateStore, paths: MossPaths
    ) -> None:
        callback = AsyncMock()
        supervisor = SwapSupervisor(
            store=store, paths=paths, on_complete=callback
        )

        await supervisor._rollback()

        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert result["status"] == "rollback-failed"
