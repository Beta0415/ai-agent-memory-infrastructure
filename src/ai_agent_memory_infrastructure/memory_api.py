from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from ai_agent_memory_infrastructure.memory_selector import (
    MemorySelectionOutput,
    MemorySelector,
)
from ai_agent_memory_infrastructure.memory_store import (
    ChromaMemoryStore,
    Memory,
    MemoryFilters,
    MemoryMatch,
    MemoryType,
)


@dataclass(frozen=True)
class MemoryAddResult:
    selection: MemorySelectionOutput
    stored_memories: tuple[Memory, ...]


class MemoryAPI(Protocol):
    def memory_add(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemoryAddResult: ...

    def memory_search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
    ) -> list[MemoryMatch]: ...

    def memory_get(self, memory_id: str) -> Memory | None: ...

    def memory_update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        memory_type: MemoryType | None = None,
        role: str | None = None,
        source: str | None = None,
        session_id: str | None = None,
        importance: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Memory: ...

    def memory_delete(self, memory_id: str) -> bool: ...


class AgentMemory:
    def __init__(
        self,
        store: ChromaMemoryStore,
        selector: MemorySelector,
        *,
        source: str,
        session_id: str,
    ) -> None:
        self._store = store
        self._selector = selector
        self._source = source
        self._session_id = session_id

    def memory_add(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemoryAddResult:
        selection = self._selector.extract_memory(conversation)
        exchange_id = uuid4().hex
        stored_memories: list[Memory] = []

        for candidate in selection.candidates:
            if not candidate.should_store:
                continue
            stored_memories.append(
                self._store.write_memory(
                    candidate.content,
                    memory_type=candidate.type,
                    role=candidate.role,
                    source=self._source,
                    session_id=self._session_id,
                    importance=candidate.importance,
                    metadata={
                        "confidence": candidate.confidence,
                        "debug_source_quote": (
                            candidate.debug_source_quote
                        ),
                        "exchange_id": exchange_id,
                        "selector_model": self._selector.model_name,
                    },
                )
            )

        return MemoryAddResult(
            selection=selection,
            stored_memories=tuple(stored_memories),
        )

    def memory_search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
    ) -> list[MemoryMatch]:
        return self._store.memory_search(
            query,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters,
        )

    def memory_get(self, memory_id: str) -> Memory | None:
        return self._store.memory_get(memory_id)

    def memory_update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        memory_type: MemoryType | None = None,
        role: str | None = None,
        source: str | None = None,
        session_id: str | None = None,
        importance: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Memory:
        return self._store.memory_update(
            memory_id,
            content=content,
            memory_type=memory_type,
            role=role,
            source=source,
            session_id=session_id,
            importance=importance,
            metadata=metadata,
        )

    def memory_delete(self, memory_id: str) -> bool:
        return self._store.memory_delete(memory_id)

    def count(self) -> int:
        return self._store.count()

    def clear(self) -> None:
        self._store.clear()
