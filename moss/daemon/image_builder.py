"""Docker image builder for candidate substrate images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import docker
from docker.errors import BuildError, DockerException

from moss.state.store import StateStore

logger = logging.getLogger(__name__)


class ImageBuilder:
    """Builds candidate Docker images from a modified substrate repo."""

    def __init__(self, store: StateStore, evo_id: str) -> None:
        self.store = store
        self.evo_id = evo_id
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def build(
        self,
        context_path: Path,
        dockerfile: str = "Dockerfile.gateway",
        iteration: int = 0,
        extra_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a candidate Docker image.

        Args:
            context_path: Path to the build context (repo root).
            dockerfile: Dockerfile to use.
            iteration: Iteration number for tagging.
            extra_tags: Additional tags to apply.

        Returns:
            Dict with image_id, tag, and build log path.
        """
        import time

        tag = f"moss-candidate:{self.evo_id}-iter{iteration}"
        timestamp_tag = f"moss-candidate:{int(time.time())}"

        logger.info("Building image %s from %s", tag, context_path)

        build_log_lines: list[str] = []

        try:
            image, build_logs = self.client.images.build(
                path=str(context_path),
                dockerfile=dockerfile,
                tag=tag,
                rm=True,
                forcerm=True,
            )

            for log_entry in build_logs:
                if "stream" in log_entry:
                    line = log_entry["stream"].rstrip()
                    if line:
                        build_log_lines.append(line)
                        logger.debug("BUILD: %s", line)
                elif "error" in log_entry:
                    build_log_lines.append(f"ERROR: {log_entry['error']}")

            # Apply additional tags
            image.tag(tag)
            image.tag(timestamp_tag)
            if extra_tags:
                for t in extra_tags:
                    image.tag(t)

            # Save build log
            log_content = "\n".join(build_log_lines)
            self.store.save_stage_artifact(
                self.evo_id, iteration, "build.log", log_content
            )

            result = {
                "image_id": image.id,
                "tag": tag,
                "tags": [tag, timestamp_tag] + (extra_tags or []),
                "success": True,
            }

            logger.info("Image built successfully: %s (%s)", tag, image.short_id)
            return result

        except BuildError as e:
            for log_entry in e.build_log:
                if "stream" in log_entry:
                    build_log_lines.append(log_entry["stream"].rstrip())
                elif "error" in log_entry:
                    build_log_lines.append(f"ERROR: {log_entry['error']}")

            log_content = "\n".join(build_log_lines)
            self.store.save_stage_artifact(
                self.evo_id, iteration, "build.log", log_content
            )

            logger.error("Image build failed: %s", e.msg)
            return {"success": False, "error": e.msg, "tag": tag}

        except DockerException as e:
            logger.error("Docker error during build: %s", e)
            return {"success": False, "error": str(e), "tag": tag}
