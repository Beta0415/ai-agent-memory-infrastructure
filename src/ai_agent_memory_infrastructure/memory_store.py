from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    INSTRUCTION = "instruction"
    EPISODE = "episode"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"
    CONFLICTED = "conflicted"


RETRIEVAL_MEMORY_STATUSES = (MemoryStatus.ACTIVE,)

RELATED_MEMORY_STATUSES = (
    MemoryStatus.ACTIVE,
    MemoryStatus.CONFLICTED,
)


class MemoryLinkRelation(StrEnum):
    REFINES = "REFINES"
    SUPERSEDES = "SUPERSEDES"
    CORRECTS = "CORRECTS"
    CONFLICTS = "CONFLICTS"


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    memory_type: MemoryType
    role: str
    source: str
    session_id: str
    importance: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object]
    status: MemoryStatus = MemoryStatus.ACTIVE
    token_count: int = 0
    last_accessed_at: datetime | None = None
    reinforced_at: datetime | None = None
    expired: bool = False
    subject: str = ""
    predicate: str = ""
    object: str = ""


@dataclass(frozen=True)
class MemoryLink:
    id: str
    from_memory_id: str
    relation: MemoryLinkRelation
    to_memory_id: str


@dataclass(frozen=True)
class MemoryMatch:
    memory: Memory
    similarity_score: float


@dataclass(frozen=True)
class MemoryFilters:
    memory_types: tuple[MemoryType, ...] = ()
    roles: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    min_importance: float | None = None
    statuses: tuple[MemoryStatus, ...] = ()
    expired: bool | None = False


def validate_memory(
    *,
    content: str,
    role: str,
    source: str,
    session_id: str,
    importance: float,
    metadata: Mapping[str, object],
) -> None:
    if not content.strip():
        raise ValueError("Memory content cannot be empty.")
    if not role.strip():
        raise ValueError("Memory role cannot be empty.")
    if not source.strip():
        raise ValueError("Memory source cannot be empty.")
    if not session_id.strip():
        raise ValueError("Memory session ID cannot be empty.")
    if not 0.0 <= importance <= 1.0:
        raise ValueError("Memory importance must be between 0 and 1.")
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("Memory metadata must be JSON serializable.") from exc
