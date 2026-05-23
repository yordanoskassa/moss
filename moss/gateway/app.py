"""FastAPI app with evolution-control endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from moss.config import load_config
from moss.models.config import DepthDial
from moss.models.evolution import EvolutionStatus
from moss.pipeline.orchestrator import Orchestrator
from moss.runner.claude_code import ClaudeCodeRunner
from moss.state.paths import MossPaths
from moss.state.store import StateStore

logger = logging.getLogger(__name__)

from moss.gateway.webhooks import router as webhooks_router

app = FastAPI(title="MOSS Evolution Gateway", version="0.1.0")
app.include_router(webhooks_router)

# Global state
_settings = load_config()
_paths = MossPaths(_settings.home)
_store = StateStore(_paths)
_current_evolution_task: asyncio.Task | None = None
_current_evo_id: str | None = None


# --- Request/Response models ---


class StartRequest(BaseModel):
    batch_id: str
    depth: str = "standard"


class RestartRequest(BaseModel):
    evo_id: str


class ApplyRequest(BaseModel):
    image_tag: str


# --- Health ---


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# --- Evolution endpoints ---


@app.get("/evo/status")
async def evo_status() -> dict[str, Any]:
    """Current evolution state."""
    evolutions = _store.list_evolutions()
    if not evolutions:
        return {"status": "idle"}

    latest = evolutions[-1]
    state = _store.load_evolution_state(latest)
    if state is None:
        return {"status": "idle"}

    return {
        "evo_id": latest,
        "status": state.status.value,
        "batch_id": state.batch_id,
        "current_iteration": state.current_iteration,
        "max_iterations": state.max_iterations,
        "depth": state.depth,
        "verdict": state.verdict.value if state.verdict else None,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "error": state.error,
    }


@app.get("/evo/batches")
async def evo_batches() -> dict[str, Any]:
    """List all batches."""
    batches = _store.list_batches()
    return {
        "batches": [
            {
                "id": b.id,
                "status": b.status.value,
                "chunk_count": b.chunk_count,
                "created_at": b.created_at.isoformat(),
            }
            for b in batches
        ]
    }


@app.get("/evo/batch/{batch_id}")
async def evo_batch(batch_id: str) -> dict[str, Any]:
    """Batch details."""
    batch = _store.load_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    return {
        "id": batch.id,
        "status": batch.status.value,
        "conversation_id": batch.conversation_id,
        "chunk_count": batch.chunk_count,
        "created_at": batch.created_at.isoformat(),
        "chunks": [
            {
                "session_id": c.session_id,
                "cursor_start": c.cursor_start,
                "cursor_end": c.cursor_end,
                "keypoints": c.keypoints,
                "content_length": len(c.content),
            }
            for c in batch.chunks
        ],
    }


@app.post("/evo/start")
async def evo_start(request: StartRequest) -> dict[str, Any]:
    """Trigger an evolution run."""
    global _current_evolution_task, _current_evo_id

    if _current_evolution_task and not _current_evolution_task.done():
        raise HTTPException(status_code=409, detail="Evolution already running")

    batch = _store.load_batch(request.batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch {request.batch_id} not found")

    try:
        depth = DepthDial(request.depth)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid depth: {request.depth}")

    runner = ClaudeCodeRunner(_settings.runner_claude_code)
    orchestrator = Orchestrator(runner, _store, _paths, _settings)

    async def _run() -> None:
        await orchestrator.run_evolution(batch, depth)

    _current_evolution_task = asyncio.create_task(_run())
    _current_evo_id = f"evo-{int(time.time())}"

    return {"evo_id": _current_evo_id, "status": "started"}


@app.post("/evo/stop")
async def evo_stop() -> dict[str, Any]:
    """Halt the current evolution run."""
    global _current_evolution_task

    if _current_evolution_task is None or _current_evolution_task.done():
        return {"message": "No evolution running"}

    _current_evolution_task.cancel()
    try:
        await _current_evolution_task
    except asyncio.CancelledError:
        pass

    # Update state
    evolutions = _store.list_evolutions()
    if evolutions:
        state = _store.load_evolution_state(evolutions[-1])
        if state and state.status == EvolutionStatus.RUNNING:
            state.status = EvolutionStatus.STOPPED
            _store.save_evolution_state(evolutions[-1], state)

    return {"message": "Evolution stopped"}


@app.post("/evo/restart")
async def evo_restart(request: RestartRequest) -> dict[str, Any]:
    """Restart a failed evolution run."""
    state = _store.load_evolution_state(request.evo_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Evolution {request.evo_id} not found")

    if state.status not in (EvolutionStatus.FAILED, EvolutionStatus.STOPPED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot restart evolution in state {state.status.value}",
        )

    batch = _store.load_batch(state.batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch {state.batch_id} not found")

    depth = DepthDial(state.depth)
    runner = ClaudeCodeRunner(_settings.runner_claude_code)
    orchestrator = Orchestrator(runner, _store, _paths, _settings)

    global _current_evolution_task
    _current_evolution_task = asyncio.create_task(
        orchestrator.run_evolution(batch, depth)
    )

    return {"evo_id": request.evo_id, "status": "restarted"}


@app.post("/evo/apply")
async def evo_apply(request: ApplyRequest) -> dict[str, Any]:
    """Write a swap-request file to apply a candidate image."""
    path = _store.write_swap_request(request.image_tag)
    return {"request_path": str(path), "image_tag": request.image_tag}
