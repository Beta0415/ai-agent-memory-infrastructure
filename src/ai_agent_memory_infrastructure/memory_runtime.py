from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_agent_memory_infrastructure.memory_api import (
    MemoryAddResult,
    MemoryAPI,
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
    ) -> MemoryContext:
        result = self._memory_api.memory_search(
            query,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters,
            now=now,
        )
        matches = result.matches
        text = "\n".join(
            (
                f"- [{match.memory.memory_type.value}; "
                f"{match.memory.role}; "
                f"final={match.final_score:.3f}; "
                f"semantic={match.semantic_score:.3f}] "
                f"{match.memory.content}"
            )
            for match in matches
        )
        return MemoryContext(
            text=text,
            matches=matches,
            candidates=result.candidates,
        )

    def runtime_write_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemoryAddResult:
        return self._memory_api.memory_add(conversation)
