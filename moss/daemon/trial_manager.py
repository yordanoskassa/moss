"""Ephemeral trial worker container lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result from running a task in a trial container."""

    task_id: str
    trial_num: int
    transcript: str
    success: bool
    exit_code: int = 0
    error: str | None = None


@dataclass
class TrialWorker:
    """Represents an ephemeral trial container."""

    container_id: str
    image_tag: str
    task_id: str
    trial_num: int
    status: str = "created"


class TrialManager:
    """Manages ephemeral trial worker containers."""

    def __init__(self, network_name: str = "moss-trial-net") -> None:
        self._client: docker.DockerClient | None = None
        self.network_name = network_name
        self._workers: dict[str, TrialWorker] = {}

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def run_trials(
        self,
        image_tag: str,
        tasks: list[dict[str, Any]],
        trials_per_task: int = 3,
        timeout: int = 300,
    ) -> dict[str, list[TrialResult]]:
        """Run batch tasks inside ephemeral containers.

        Args:
            image_tag: Docker image to use for trial containers.
            tasks: List of task dicts with 'id' and 'command' keys.
            trials_per_task: Number of times to repeat each task.
            timeout: Per-trial timeout in seconds.

        Returns:
            Dict mapping task_id to list of TrialResults.
        """
        results: dict[str, list[TrialResult]] = {}

        # Ensure network exists
        self._ensure_network()

        for task in tasks:
            task_id = task["id"]
            task_results: list[TrialResult] = []

            for trial_num in range(1, trials_per_task + 1):
                logger.info(
                    "Running trial %d/%d for task %s",
                    trial_num,
                    trials_per_task,
                    task_id,
                )

                result = await self._run_single_trial(
                    image_tag=image_tag,
                    task=task,
                    trial_num=trial_num,
                    timeout=timeout,
                )
                task_results.append(result)

            results[task_id] = task_results

        return results

    async def _run_single_trial(
        self,
        image_tag: str,
        task: dict[str, Any],
        trial_num: int,
        timeout: int,
    ) -> TrialResult:
        """Spawn a single ephemeral container for one trial."""
        task_id = task["id"]
        command = task.get("command", "")
        container = None

        try:
            container = self.client.containers.run(
                image_tag,
                command=command,
                detach=True,
                network=self.network_name,
                # Network-isolated, no user state volume
                read_only=False,
                auto_remove=False,
                labels={
                    "moss.trial": "true",
                    "moss.task_id": task_id,
                    "moss.trial_num": str(trial_num),
                },
            )

            worker = TrialWorker(
                container_id=container.id,
                image_tag=image_tag,
                task_id=task_id,
                trial_num=trial_num,
                status="running",
            )
            self._workers[container.id] = worker

            # Wait for container to finish
            exit_info = container.wait(timeout=timeout)
            exit_code = exit_info.get("StatusCode", -1)

            # Capture logs as transcript
            logs = container.logs(stdout=True, stderr=True)
            transcript = logs.decode(errors="replace") if isinstance(logs, bytes) else str(logs)

            worker.status = "completed"

            return TrialResult(
                task_id=task_id,
                trial_num=trial_num,
                transcript=transcript,
                success=exit_code == 0,
                exit_code=exit_code,
            )

        except Exception as e:
            logger.error("Trial %d for task %s failed: %s", trial_num, task_id, e)
            return TrialResult(
                task_id=task_id,
                trial_num=trial_num,
                transcript="",
                success=False,
                exit_code=-1,
                error=str(e),
            )
        finally:
            if container:
                try:
                    container.remove(force=True)
                except DockerException:
                    pass
                self._workers.pop(container.id, None)

    def _ensure_network(self) -> None:
        """Ensure the trial network exists."""
        try:
            self.client.networks.get(self.network_name)
        except docker.errors.NotFound:
            try:
                self.client.networks.create(
                    self.network_name,
                    driver="bridge",
                    internal=True,  # Network-isolated
                )
                logger.info("Created trial network: %s", self.network_name)
            except DockerException as e:
                logger.warning("Could not create trial network: %s", e)

    def destroy_all(self) -> int:
        """Destroy all active trial containers. Returns count destroyed."""
        destroyed = 0
        for cid in list(self._workers.keys()):
            try:
                container = self.client.containers.get(cid)
                container.remove(force=True)
                destroyed += 1
            except DockerException:
                pass
            self._workers.pop(cid, None)
        return destroyed

    def status(self) -> dict[str, Any]:
        """Return status of all active trial workers."""
        return {
            "active_workers": len(self._workers),
            "workers": [
                {
                    "container_id": w.container_id[:12],
                    "task_id": w.task_id,
                    "trial_num": w.trial_num,
                    "status": w.status,
                }
                for w in self._workers.values()
            ],
        }
