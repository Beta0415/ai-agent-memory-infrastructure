from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from ai_agent_memory_infrastructure.memory_store import (
    Memory,
    MemoryAPI,
    MemoryFilters,
    MemoryMatch,
    MemoryType,
)


@dataclass(frozen=True)
class MemoryContext:
    text: str
    matches: tuple[MemoryMatch, ...]


class MemoryRuntime:
    def __init__(
        self,
        memory_api: MemoryAPI,
        *,
        source: str,
        session_id: str,
    ) -> None:
        self._memory_api = memory_api
        self._source = source
        self._session_id = session_id

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

    def runtime_save_exchange_to_memory(
        self,
        user_message: str,
        assistant_message: str,
        *,
        importance: float = 0.5,
        metadata: dict[str, object] | None = None,
    ) -> tuple[Memory, Memory]:
        exchange_id = uuid4().hex
        shared_metadata = {
            **(metadata or {}),
            "exchange_id": exchange_id,
        }
        user_memory = self._memory_api.memory_add(
            user_message,
            memory_type=MemoryType.CONVERSATION,
            role="user",
            source=self._source,
            session_id=self._session_id,
            importance=importance,
            metadata=shared_metadata,
        )
        assistant_memory = self._memory_api.memory_add(
            assistant_message,
            memory_type=MemoryType.CONVERSATION,
            role="assistant",
            source=self._source,
            session_id=self._session_id,
            importance=importance,
            metadata=shared_metadata,
        )
        return user_memory, assistant_memory
