"""Daemon entry point: event loop, RPC server, swap supervisor, auto-scan."""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

from moss.config import load_config
from moss.daemon.auto_scan import AutoScanEngine
from moss.daemon.server import RPCServer
from moss.daemon.swap_supervisor import SwapSupervisor
from moss.daemon.trial_manager import TrialManager
from moss.daemon.image_builder import ImageBuilder
from moss.models.config import Settings
from moss.runner.claude_code import ClaudeCodeRunner
from moss.state.paths import MossPaths
from moss.state.store import StateStore

logger = logging.getLogger(__name__)


class MossDaemon:
    """Host-resident daemon coordinating all MOSS background services."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.settings = load_config(config_path)
        self.paths = MossPaths(self.settings.home)
        self.store = StateStore(self.paths)

        self.rpc_server = RPCServer(self.settings.daemon.socket_path)
        self.swap_supervisor = SwapSupervisor(
            store=self.store,
            paths=self.paths,
            config=self.settings.swap,
        )
        self.scan_engine = AutoScanEngine(
            store=self.store,
            paths=self.paths,
            settings=self.settings,
        )
        self.trial_manager = TrialManager()
        self.runner = ClaudeCodeRunner(self.settings.runner_claude_code)

        self._shutdown_event = asyncio.Event()

    def _register_handlers(self) -> None:
        """Register all RPC method handlers."""

        # -- scan.* --
        async def scan_catch_up(params: dict[str, Any]) -> dict[str, Any]:
            return self.scan_engine.catch_up()

        async def scan_flag(params: dict[str, Any]) -> dict[str, Any]:
            session_id = params.get("session_id", "")
            if not session_id:
                return {"error": "session_id is required"}
            return self.scan_engine.flag(session_id)

        self.rpc_server.register("scan.catch_up", scan_catch_up)
        self.rpc_server.register("scan.flag", scan_flag)

        # -- coding_agent.invoke --
        async def coding_agent_invoke(params: dict[str, Any]) -> dict[str, Any]:
            stage = params.get("stage", "")
            prompt = params.get("prompt", "")
            workdir = Path(params.get("workdir", "."))
            context = params.get("context", {})
            result = await self.runner.invoke(stage, prompt, workdir, context)
            return {
                "output": result.output,
                "success": result.success,
                "exit_code": result.exit_code,
                "error": result.error,
            }

        self.rpc_server.register("coding_agent.invoke", coding_agent_invoke)

        # -- trial.* --
        async def trial_create(params: dict[str, Any]) -> dict[str, Any]:
            image_tag = params.get("image_tag", "")
            tasks = params.get("tasks", [])
            trials_per = params.get("trials_per_task", 3)
            results = await self.trial_manager.run_trials(image_tag, tasks, trials_per)
            return {
                "task_count": len(results),
                "results": {
                    tid: [
                        {"trial": r.trial_num, "success": r.success, "exit_code": r.exit_code}
                        for r in rs
                    ]
                    for tid, rs in results.items()
                },
            }

        async def trial_status(params: dict[str, Any]) -> dict[str, Any]:
            return self.trial_manager.status()

        async def trial_destroy(params: dict[str, Any]) -> dict[str, Any]:
            count = self.trial_manager.destroy_all()
            return {"destroyed": count}

        self.rpc_server.register("trial.create", trial_create)
        self.rpc_server.register("trial.status", trial_status)
        self.rpc_server.register("trial.destroy", trial_destroy)

        # -- image.build --
        async def image_build(params: dict[str, Any]) -> dict[str, Any]:
            evo_id = params.get("evo_id", "unknown")
            context_path = Path(params.get("context_path", "."))
            iteration = params.get("iteration", 0)
            builder = ImageBuilder(self.store, evo_id)
            return await builder.build(context_path, iteration=iteration)

        self.rpc_server.register("image.build", image_build)

        # -- swap.* --
        async def swap_request(params: dict[str, Any]) -> dict[str, Any]:
            image_tag = params.get("image_tag", "")
            if not image_tag:
                return {"error": "image_tag required"}
            path = self.store.write_swap_request(image_tag)
            return {"request_path": str(path)}

        async def swap_status(params: dict[str, Any]) -> dict[str, Any]:
            lkg = self.store.read_last_known_good()
            return {"last_known_good": lkg}

        self.rpc_server.register("swap.request", swap_request)
        self.rpc_server.register("swap.status", swap_status)

    async def _periodic_scan(self) -> None:
        """Periodically run auto-scan."""
        interval = self.settings.daemon.auto_scan_interval
        while not self._shutdown_event.is_set():
            try:
                result = self.scan_engine.catch_up()
                if result["chunks_added"] > 0:
                    logger.info("Auto-scan: %s", result)
            except Exception:
                logger.exception("Auto-scan error")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Interval elapsed, loop again

    async def run(self) -> None:
        """Start all daemon services."""
        self.paths.ensure_all()
        self._register_handlers()

        # Start RPC server
        await self.rpc_server.start()

        # Start background tasks
        swap_task = asyncio.create_task(self.swap_supervisor.run())
        scan_task = asyncio.create_task(self._periodic_scan())

        logger.info("MOSS daemon started")

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Clean up
        self.swap_supervisor.stop()
        swap_task.cancel()
        scan_task.cancel()

        try:
            await asyncio.gather(swap_task, scan_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        await self.rpc_server.stop()
        logger.info("MOSS daemon stopped")

    def shutdown(self) -> None:
        self._shutdown_event.set()


def main() -> None:
    """Entry point for moss-daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    daemon = MossDaemon()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, daemon.shutdown)

    try:
        loop.run_until_complete(daemon.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
