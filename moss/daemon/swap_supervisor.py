"""Swap supervisor: file-poll → restart → health probe → commit/rollback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

import docker
from docker.errors import DockerException

from moss.models.config import SwapConfig
from moss.state.paths import MossPaths
from moss.state.store import StateStore

logger = logging.getLogger(__name__)


class HealthProbe:
    """Runs 4 health checks per probe sample."""

    def __init__(self, container_name: str = "moss-gateway") -> None:
        self._client: docker.DockerClient | None = None
        self.container_name = container_name

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def check_heartbeat_freshness(self, max_age: float = 30.0) -> bool:
        """Check that the heartbeat file is fresh (≤ max_age seconds old)."""
        try:
            container = self.client.containers.get(self.container_name)
            result = container.exec_run(
                ["cat", "/tmp/moss-heartbeat"],
                demux=True,
            )
            if result.exit_code != 0:
                return False
            stdout = result.output[0]
            if stdout is None:
                return False
            ts = float(stdout.decode().strip())
            return (time.time() - ts) <= max_age
        except Exception as e:
            logger.debug("Heartbeat check failed: %s", e)
            return False

    async def check_container_running(self) -> bool:
        """Check the container is in running state via docker inspect."""
        try:
            container = self.client.containers.get(self.container_name)
            return container.status == "running"
        except DockerException:
            return False

    async def check_substrate_status_1(self) -> bool:
        """Substrate CLI status probe #1."""
        try:
            container = self.client.containers.get(self.container_name)
            result = container.exec_run(
                ["curl", "-sf", "http://localhost:8420/health"],
                demux=True,
            )
            return result.exit_code == 0
        except Exception:
            return False

    async def check_substrate_status_2(self) -> bool:
        """Substrate CLI status probe #2."""
        try:
            container = self.client.containers.get(self.container_name)
            result = container.exec_run(
                ["curl", "-sf", "http://localhost:8420/evo/status"],
                demux=True,
            )
            return result.exit_code == 0
        except Exception:
            return False

    async def run_all(self, heartbeat_max_age: float = 30.0) -> bool:
        """Run all 4 health checks. All must pass."""
        checks = [
            self.check_heartbeat_freshness(heartbeat_max_age),
            self.check_container_running(),
            self.check_substrate_status_1(),
            self.check_substrate_status_2(),
        ]
        results = await asyncio.gather(*checks, return_exceptions=True)
        passed = all(r is True for r in results)
        if not passed:
            logger.debug("Health probe results: %s", results)
        return passed


class SwapSupervisor:
    """Polls for swap requests and manages container replacement."""

    def __init__(
        self,
        store: StateStore,
        paths: MossPaths,
        config: SwapConfig | None = None,
        container_name: str = "moss-gateway",
        on_complete: Callable[[dict[str, Any]], Coroutine] | None = None,
    ) -> None:
        self.store = store
        self.paths = paths
        self.config = config or SwapConfig()
        self.container_name = container_name
        self.probe = HealthProbe(container_name)
        self._on_complete = on_complete
        self._running = False
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def run(self) -> None:
        """Main polling loop — runs as an asyncio task."""
        self._running = True
        logger.info(
            "Swap supervisor started, polling every %.1fs", self.config.poll_interval
        )

        while self._running:
            try:
                request = self._poll_for_request()
                if request:
                    await self._handle_swap(request)
            except Exception:
                logger.exception("Error in swap supervisor poll")

            await asyncio.sleep(self.config.poll_interval)

    def stop(self) -> None:
        self._running = False

    def _poll_for_request(self) -> dict[str, Any] | None:
        """Check for swap-request files in the requests directory."""
        req_dir = self.paths.swap_requests_dir
        if not req_dir.exists():
            return None

        for path in sorted(req_dir.glob("swap-*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                # Remove the request file once read
                path.unlink()
                data["_request_file"] = str(path)
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Invalid swap request %s: %s", path, e)
                path.unlink(missing_ok=True)

        return None

    async def _handle_swap(self, request: dict[str, Any]) -> None:
        """Handle a swap request: stop → start → probe → commit/rollback."""
        image_tag = request.get("image_tag")
        if not image_tag:
            logger.error("Swap request missing image_tag")
            return

        logger.info("Processing swap request for image: %s", image_tag)

        # Step 1: Stop current container
        try:
            old_container = self.client.containers.get(self.container_name)
            old_container.stop(timeout=30)
            old_container.remove()
            logger.info("Stopped old container")
        except DockerException as e:
            logger.warning("Could not stop old container: %s", e)

        # Step 2: Start new container from candidate image
        try:
            new_container = self.client.containers.run(
                image_tag,
                name=self.container_name,
                detach=True,
                ports={"8420/tcp": 8420},
                labels={"moss.swap": "candidate"},
            )
            logger.info("Started candidate container: %s", new_container.short_id)
        except DockerException as e:
            logger.error("Could not start candidate container: %s", e)
            await self._rollback()
            return

        # Step 3: Enter probe window
        success = await self._probe_window()

        # Step 4: Commit or rollback
        if success:
            await self._commit(image_tag)
        else:
            await self._rollback()

    async def _probe_window(self) -> bool:
        """Run health probes during the probe window.

        Requires `consecutive_passes_required` consecutive passing probes.
        """
        start = time.time()
        consecutive_passes = 0

        while (time.time() - start) < self.config.probe_window:
            passed = await self.probe.run_all(self.config.heartbeat_max_age)

            if passed:
                consecutive_passes += 1
                logger.info(
                    "Health probe passed (%d/%d consecutive)",
                    consecutive_passes,
                    self.config.consecutive_passes_required,
                )
                if consecutive_passes >= self.config.consecutive_passes_required:
                    return True
            else:
                if consecutive_passes > 0:
                    logger.warning("Health probe failed, resetting consecutive count")
                consecutive_passes = 0

            await asyncio.sleep(self.config.probe_sample_interval)

        logger.error("Probe window expired without sufficient consecutive passes")
        return False

    async def _commit(self, image_tag: str) -> None:
        """Commit a successful swap."""
        self.store.write_last_known_good(image_tag)
        logger.info("Swap committed: %s is now last-known-good", image_tag)

        result = {"status": "success", "image_tag": image_tag}
        if self._on_complete:
            await self._on_complete(result)

    async def _rollback(self) -> None:
        """Roll back to last-known-good image."""
        lkg = self.store.read_last_known_good()
        if not lkg:
            logger.error("No last-known-good image to roll back to")
            result = {"status": "rollback-failed", "error": "No LKG image"}
            if self._on_complete:
                await self._on_complete(result)
            return

        logger.info("Rolling back to last-known-good image: %s", lkg)

        # Stop failed candidate
        try:
            container = self.client.containers.get(self.container_name)
            container.stop(timeout=10)
            container.remove()
        except DockerException:
            pass

        # Start LKG
        try:
            self.client.containers.run(
                lkg,
                name=self.container_name,
                detach=True,
                ports={"8420/tcp": 8420},
                labels={"moss.swap": "rollback"},
            )
            logger.info("Rollback container started")
            result = {"status": "rolled-back", "image_tag": lkg}
        except DockerException as e:
            logger.error("Rollback failed: %s", e)
            result = {"status": "rollback-failed", "error": str(e)}

        if self._on_complete:
            await self._on_complete(result)
