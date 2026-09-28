from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ai_agent_memory_infrastructure.memory_api import (
    MemoryAddResult,
    MemoryAPI,
)
from ai_agent_memory_infrastructure.memory_store import (
    MemoryFilters,
    MemoryMatch,
)


@dataclass(frozen=True)
class MemoryContext:
    text: str
    matches: tuple[MemoryMatch, ...]


class MemoryRuntime:
    def __init__(self, memory_api: MemoryAPI) -> None:
        self._memory_api = memory_api

    def runtime_build_memory_context(
        self,
        query: str,
        *,
        limit: int = 4,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
    ) -> MemoryContext:
        matches = self._memory_api.memory_search(
            query,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters,
        )
        text = "\n".join(
            (
                f"- [{match.memory.memory_type.value}; "
                f"{match.memory.role}; "
                f"similarity={match.similarity_score:.3f}] "
                f"{match.memory.content}"
            )
            for match in matches
        )
        return MemoryContext(text=text, matches=tuple(matches))

    def runtime_write_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemoryAddResult:
        return self._memory_api.memory_add(conversation)
