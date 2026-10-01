from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_agent_memory_infrastructure.memory_api import (
    MemoryAddResult,
    MemoryAPI,
    ReadPathSettings,
)
from ai_agent_memory_infrastructure.memory_context import (
    DuplicateCluster,
    format_context_memories,
)
from ai_agent_memory_infrastructure.memory_retrieval import (
    DEFAULT_RESULT_LIMIT,
    RetrievedMemory,
)
from ai_agent_memory_infrastructure.memory_store import MemoryFilters


@dataclass(frozen=True)
class MemoryContext:
    text: str
    matches: tuple[RetrievedMemory, ...]
    candidates: tuple[RetrievedMemory, ...]
    deduplicated: bool = False
    dropped_memories: tuple[RetrievedMemory, ...] = ()
    duplicate_clusters: tuple[DuplicateCluster, ...] = ()


class MemoryRuntime:
    def __init__(self, memory_api: MemoryAPI) -> None:
        self._memory_api = memory_api

    def runtime_build_memory_context(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        settings: ReadPathSettings | None = None,
    ) -> MemoryContext:
        result = self._memory_api.memory_context(
            query,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters,
            now=now,
            settings=settings,
        )
        matches = result.memories
        text = format_context_memories(matches)
        return MemoryContext(
            text=text,
            matches=matches,
            candidates=result.candidates,
            deduplicated=result.deduplicated,
            dropped_memories=result.dropped_memories,
            duplicate_clusters=result.duplicate_clusters,
        )

    def runtime_write_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemoryAddResult:
        return self._memory_api.memory_add(conversation)
