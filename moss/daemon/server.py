"""Unix-socket RPC server for the MOSS daemon."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type for RPC handlers
RPCHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class RPCServer:
    """Asyncio Unix-socket server with JSON-RPC-style protocol.

    Supports 4 operation families:
    1. coding_agent.invoke — spawn Claude Code per stage
    2. trial.* — trial worker lifecycle
    3. scan.* — auto-scan engine
    4. image.build, swap.* — build + swap
    """

    def __init__(self, socket_path: str = "/tmp/moss-daemon.sock") -> None:
        self.socket_path = socket_path
        self._handlers: dict[str, RPCHandler] = {}
        self._server: asyncio.AbstractServer | None = None

    def register(self, method: str, handler: RPCHandler) -> None:
        """Register an RPC method handler."""
        self._handlers[method] = handler

    async def start(self) -> None:
        """Start the Unix socket server."""
        # Remove stale socket
        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self.socket_path,
        )

        logger.info("RPC server listening on %s", self.socket_path)

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()

        logger.info("RPC server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        try:
            data = await reader.readline()
            if not data:
                return

            try:
                request = json.loads(data.decode())
            except json.JSONDecodeError as e:
                response = {"error": f"Invalid JSON: {e}"}
                writer.write(json.dumps(response).encode())
                await writer.drain()
                return

            method = request.get("method", "")
            params = request.get("params", {})

            logger.debug("RPC call: %s(%s)", method, params)

            handler = self._handlers.get(method)
            if handler is None:
                response = {"error": f"Unknown method: {method}"}
            else:
                try:
                    response = await handler(params)
                except Exception as e:
                    logger.exception("RPC handler error for %s", method)
                    response = {"error": str(e)}

            writer.write(json.dumps(response, default=str).encode())
            await writer.drain()

        except Exception:
            logger.exception("Connection handler error")
        finally:
            writer.close()
            await writer.wait_closed()
