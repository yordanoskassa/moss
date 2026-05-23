"""Batch and chunk models for session trace grouping."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class BatchStatus(str, Enum):
    OPEN = "open"
    SEALED = "sealed"


class Chunk(BaseModel):
    """A slice of a session transcript with scored keypoints."""

    session_id: str
    cursor_start: int
    cursor_end: int
    content: str
    keypoints: list[str] = Field(default_factory=list)


class Batch(BaseModel):
    """A group of chunks from one or more sessions, forming an evolution input."""

    id: str
    chunks: list[Chunk] = Field(default_factory=list)
    status: BatchStatus = BatchStatus.OPEN
    conversation_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def seal(self) -> None:
        self.status = BatchStatus.SEALED

    @property
    def is_sealed(self) -> bool:
        return self.status == BatchStatus.SEALED

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
