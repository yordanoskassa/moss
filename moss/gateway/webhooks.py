"""Webhook receivers for evolution lifecycle events."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evo/webhooks", tags=["webhooks"])


class WebhookPayload(BaseModel):
    """Generic webhook payload."""

    event: str
    data: dict[str, Any] = {}


# In-memory queue for system messages to inject into the agent
_pending_messages: list[dict[str, Any]] = []


def get_pending_messages() -> list[dict[str, Any]]:
    """Drain and return all pending system messages."""
    messages = list(_pending_messages)
    _pending_messages.clear()
    return messages


def _enqueue_message(event: str, content: str) -> None:
    """Add a system message to the pending queue."""
    _pending_messages.append({"event": event, "content": content})
    logger.info("Enqueued system message for event: %s", event)


@router.post("/evolution-converged")
async def evolution_converged(payload: WebhookPayload) -> dict[str, str]:
    """Fired when the evolution loop converges."""
    evo_id = payload.data.get("evo_id", "unknown")
    score = payload.data.get("final_score", "N/A")
    iterations = payload.data.get("iterations", "N/A")

    _enqueue_message(
        "evolution-converged",
        f"MOSS evolution {evo_id} has converged after {iterations} iterations. "
        f"Final aggregate score: {score}. "
        "The candidate image is ready for swap. Use `moss evo apply <image-tag>` to deploy.",
    )

    return {"status": "accepted"}


@router.post("/evolution-failed")
async def evolution_failed(payload: WebhookPayload) -> dict[str, str]:
    """Fired on terminal evolution failure."""
    evo_id = payload.data.get("evo_id", "unknown")
    error = payload.data.get("error", "unknown error")
    verdict = payload.data.get("verdict", "")

    message = f"MOSS evolution {evo_id} has failed: {error}."
    if verdict:
        message += f" Verdict: {verdict}."

    _enqueue_message("evolution-failed", message)

    return {"status": "accepted"}


@router.post("/apply-complete")
async def apply_complete(payload: WebhookPayload) -> dict[str, str]:
    """Fired after swap settles (commit or rollback)."""
    status = payload.data.get("status", "unknown")
    image_tag = payload.data.get("image_tag", "unknown")

    if status == "success":
        content = (
            f"MOSS swap completed successfully. Image {image_tag} is now active "
            "and recorded as last-known-good."
        )
    elif status == "rolled-back":
        content = (
            f"MOSS swap for {image_tag} failed health probes. "
            "Rolled back to last-known-good image."
        )
    else:
        content = f"MOSS swap result: {status} for image {image_tag}."

    _enqueue_message("apply-complete", content)

    return {"status": "accepted"}
