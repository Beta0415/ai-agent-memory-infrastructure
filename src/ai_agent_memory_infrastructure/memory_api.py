from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from ai_agent_memory_infrastructure.memory_compressor import (
    MemoryCompressor,
)
from ai_agent_memory_infrastructure.memory_context import (
    CONTEXT_TOKEN_BUDGET,
    DuplicateCluster,
    apply_context_compression,
    cluster_duplicate_memories,
    context_token_count,
    order_context_memories,
    resolve_duplicate_group,
)
from ai_agent_memory_infrastructure.memory_relation import (
    MemoryRelationClassifier,
)
from ai_agent_memory_infrastructure.memory_retrieval import (
    CANDIDATE_RETRIEVAL_LIMIT,
    DEFAULT_RESULT_LIMIT,
    WEIGHTED_CANDIDATE_LIMIT,
    MemoryReranker,
    RetrievedMemory,
    apply_rerank_scores,
    rank_by_retrieval_score,
)
from ai_agent_memory_infrastructure.memory_selector import (
    MemorySelectionOutput,
    MemorySelector,
)
from ai_agent_memory_infrastructure.memory_store import (
    ChromaMemoryStore,
    Memory,
    MemoryFilters,
    MemoryType,
)


@dataclass(frozen=True)
class MemorySearchResult:
    matches: tuple[RetrievedMemory, ...]
    candidates: tuple[RetrievedMemory, ...]


@dataclass(frozen=True)
class MemoryContextResult:
    memories: tuple[RetrievedMemory, ...]
    matches: tuple[RetrievedMemory, ...]
    candidates: tuple[RetrievedMemory, ...]
    deduplicated: bool = False
    dropped_memories: tuple[RetrievedMemory, ...] = ()
    duplicate_clusters: tuple[DuplicateCluster, ...] = ()


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
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
    ) -> MemorySearchResult: ...

    def memory_context(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        token_budget: int = CONTEXT_TOKEN_BUDGET,
    ) -> MemoryContextResult: ...

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
        reranker: MemoryReranker | None = None,
        compressor: MemoryCompressor | None = None,
        relation_classifier: MemoryRelationClassifier | None = None,
    ) -> None:
        self._store = store
        self._selector = selector
        self._reranker = reranker
        self._compressor = compressor
        self._relation_classifier = relation_classifier
        self._source = source
        self._session_id = session_id

    def memory_add(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemoryAddResult:
        selection = self._selector.extract_memory(conversation)
        exchange_id = uuid4().hex
        stored_memories: list[Memory] = []

        for candidate in selection.selected:
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
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
    ) -> MemorySearchResult:
        if limit < 1:
            raise ValueError("Memory result limit must be at least 1.")
        semantic_matches = self._store.memory_search(
            query,
            limit=max(limit, CANDIDATE_RETRIEVAL_LIMIT),
            min_similarity=min_similarity,
            filters=filters,
        )
        scored_at = now or datetime.now(UTC)
        scored_candidates = rank_by_retrieval_score(
            semantic_matches,
            limit=max(len(semantic_matches), 1),
            now=scored_at,
        )
        rerank_pool = scored_candidates[:WEIGHTED_CANDIDATE_LIMIT]
        if self._reranker is None or not rerank_pool:
            return MemorySearchResult(
                matches=tuple(scored_candidates[:limit]),
                candidates=scored_candidates,
            )
        scores = self._reranker.rerank(query, rerank_pool)
        reranked = apply_rerank_scores(rerank_pool, scores)
        rerank_scores = {
            item.memory.id: item.rerank_score for item in reranked
        }
        candidates = tuple(
            replace(item, rerank_score=rerank_scores.get(item.memory.id))
            for item in scored_candidates
        )
        return MemorySearchResult(
            matches=tuple(reranked[:limit]),
            candidates=candidates,
        )

    def memory_context(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        token_budget: int = CONTEXT_TOKEN_BUDGET,
    ) -> MemoryContextResult:
        if token_budget < 1:
            raise ValueError("Memory token budget must be at least 1.")
        search = self.memory_search(
            query,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters,
            now=now,
        )
        if not search.matches:
            return MemoryContextResult(
                memories=(),
                matches=(),
                candidates=search.candidates,
            )
        embeddings = self._store.memory_embeddings(
            [item.memory.id for item in search.matches]
        )
        groups = cluster_duplicate_memories(
            search.matches,
            embeddings,
        )
        kept: list[RetrievedMemory] = []
        duplicate_clusters: list[DuplicateCluster] = []
        for group in groups:
            if len(group) < 2:
                kept.extend(group)
                continue
            if self._relation_classifier is None:
                raise RuntimeError(
                    "Near-duplicate memories need a relation classifier."
                )
            relation = self._relation_classifier.classify(group)
            resolved = resolve_duplicate_group(group, relation)
            kept.extend(resolved)
            label = relation.strip().lower()
            duplicate_clusters.append(
                DuplicateCluster(
                    memories=tuple(group),
                    relation=label,
                    selected=resolved[0] if label == "equivalent" else None,
                )
            )
        memories = tuple(kept)
        kept_ids = {item.memory.id for item in memories}
        dropped_memories = tuple(
            item
            for item in search.matches
            if item.memory.id not in kept_ids
        )
        deduplicated = len(dropped_memories) > 0
        if context_token_count(memories) > token_budget:
            if self._compressor is None:
                raise RuntimeError(
                    "Memory context exceeds the token budget and no "
                    "compressor is configured."
                )
            compressed = self._compressor.compress(
                memories,
                token_budget=token_budget,
            )
            memories = apply_context_compression(memories, compressed)
        memories = order_context_memories(memories)
        return MemoryContextResult(
            memories=memories,
            matches=search.matches,
            candidates=search.candidates,
            deduplicated=deduplicated,
            dropped_memories=dropped_memories,
            duplicate_clusters=tuple(duplicate_clusters),
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
