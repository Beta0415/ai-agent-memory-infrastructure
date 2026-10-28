from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from ai_agent_memory_infrastructure.memory_atomic_checker import (
    ATOMIC_CHECK_VERDICTS,
    MemoryAtomicChecker,
)
from ai_agent_memory_infrastructure.memory_atomic_rewriter import (
    AtomicRewritePiece,
    MemoryAtomicRewriter,
)
from ai_agent_memory_infrastructure.memory_atomic_score import (
    ATOMIC_SCORE_REVIEW_THRESHOLD,
    AtomicScore,
    AtomicScoreWeights,
    needs_atomic_review,
    score_atomic_memory,
)
from ai_agent_memory_infrastructure.memory_candidate_relation import (
    CANDIDATE_RELATION_EFFECTS,
    CANDIDATE_RELATIONS,
    WRITE_RELATED_LIMIT,
    WRITE_RELATED_SIMILARITY,
    MemoryCandidateRelationClassifier,
    similarity_above,
)
from ai_agent_memory_infrastructure.memory_dedup import (
    MemoryDedupClassifier,
)
from ai_agent_memory_infrastructure.memory_entity_judge import EntityJudge
from ai_agent_memory_infrastructure.memory_entity_selector import (
    EntitySelector,
)
from ai_agent_memory_infrastructure.memory_entity_similarity import (
    EntityResolutionSettings,
)
from ai_agent_memory_infrastructure.memory_graph import (
    GraphBuildResult,
    MemoryGraph,
    MemoryGraphExtractor,
)
from ai_agent_memory_infrastructure.memory_compressor import (
    MemoryCompressor,
)
from ai_agent_memory_infrastructure.memory_maintenance import (
    MaintenanceResult,
    MaintenanceSettings,
    MemoryMaintenance,
)
from ai_agent_memory_infrastructure.memory_context import (
    CONTEXT_DUPLICATE_SIMILARITY,
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
    IMPORTANCE_WEIGHT,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_WEIGHT,
    SEMANTIC_WEIGHT,
    WEIGHTED_CANDIDATE_LIMIT,
    MemoryReranker,
    RetrievedMemory,
    apply_rerank_scores,
    rank_by_retrieval_score,
)
from ai_agent_memory_infrastructure.memory_selector import (
    MemoryCandidate,
    MemorySelectionOutput,
    MemorySelector,
)
from ai_agent_memory_infrastructure.memory_store import (
    RELATED_MEMORY_STATUSES,
    RETRIEVAL_MEMORY_STATUSES,
    Memory,
    MemoryFilters,
    MemoryLink,
    MemoryMatch,
    MemoryStatus,
    MemoryType,
)
from ai_agent_memory_infrastructure.postgres_memory_store import (
    PostgresMemoryStore,
)


@dataclass(frozen=True)
class ReadPathSettings:
    semantic_weight: float = SEMANTIC_WEIGHT
    recency_weight: float = RECENCY_WEIGHT
    importance_weight: float = IMPORTANCE_WEIGHT
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    candidate_limit: int = CANDIDATE_RETRIEVAL_LIMIT
    weighted_limit: int = WEIGHTED_CANDIDATE_LIMIT
    result_limit: int = DEFAULT_RESULT_LIMIT
    duplicate_similarity: float = CONTEXT_DUPLICATE_SIMILARITY
    token_budget: int = CONTEXT_TOKEN_BUDGET

    def __post_init__(self) -> None:
        weights = (
            self.semantic_weight,
            self.recency_weight,
            self.importance_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Retrieval weights must be zero or greater.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Retrieval weights must sum to 1.")
        if self.recency_half_life_days <= 0:
            raise ValueError("Recency half-life must be greater than zero.")
        if self.result_limit < 1:
            raise ValueError("Returned memory count must be at least 1.")
        if not (
            self.candidate_limit >= self.weighted_limit >= self.result_limit
        ):
            raise ValueError(
                "Retrieval cuts must stay in order: semantic candidates, "
                "then the weighted pool, then the number returned."
            )
        if not 0.0 <= self.duplicate_similarity <= 1.0:
            raise ValueError(
                "Duplicate similarity must be between 0 and 1."
            )
        if self.token_budget < 1:
            raise ValueError("Token budget must be at least 1.")

    @property
    def recency_lambda_per_day(self) -> float:
        return math.log(2) / self.recency_half_life_days


_DEFAULT_ATOMIC_WEIGHTS = AtomicScoreWeights()


@dataclass(frozen=True)
class WritePathSettings:
    predicate_weight: float = _DEFAULT_ATOMIC_WEIGHTS.predicate
    list_weight: float = _DEFAULT_ATOMIC_WEIGHTS.coordinated_items
    clause_weight: float = _DEFAULT_ATOMIC_WEIGHTS.clause
    coordination_weight: float = _DEFAULT_ATOMIC_WEIGHTS.coordination
    value_weight: float = _DEFAULT_ATOMIC_WEIGHTS.value
    length_weight: float = _DEFAULT_ATOMIC_WEIGHTS.tokens
    related_similarity: float = WRITE_RELATED_SIMILARITY
    related_limit: int = WRITE_RELATED_LIMIT
    atomic_score_threshold: float = ATOMIC_SCORE_REVIEW_THRESHOLD

    def __post_init__(self) -> None:
        self.atomic_weights()
        if not 0.0 <= self.atomic_score_threshold <= 1.0:
            raise ValueError(
                "Atomic score threshold must be between 0 and 1."
            )
        if not 0.0 <= self.related_similarity <= 1.0:
            raise ValueError(
                "Related similarity must be between 0 and 1."
            )
        if self.related_limit < 1:
            raise ValueError("Related memory count must be at least 1.")

    def atomic_weights(self) -> AtomicScoreWeights:
        return AtomicScoreWeights(
            predicate=self.predicate_weight,
            coordinated_items=self.list_weight,
            clause=self.clause_weight,
            coordination=self.coordination_weight,
            value=self.value_weight,
            tokens=self.length_weight,
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
class CandidateMemoryDecision:
    candidate: MemoryCandidate
    related: tuple[MemoryMatch, ...]
    relation: str | None
    stored: Memory | None
    links: tuple[MemoryLink, ...] = ()
    relations: tuple[tuple[str, str], ...] = ()
    atomic_score: AtomicScore | None = None
    atomic_verdict: str | None = None
    rewritten: tuple[str, ...] = ()
    from_rewrite: bool = False


@dataclass(frozen=True)
class MemoryAddResult:
    selection: MemorySelectionOutput
    stored_memories: tuple[Memory, ...]
    decisions: tuple[CandidateMemoryDecision, ...] = ()
    not_selected_scores: tuple[AtomicScore, ...] = ()


class MemoryAPI(Protocol):
    def memory_add(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        settings: WritePathSettings | None = None,
    ) -> MemoryAddResult: ...

    def memory_search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        settings: ReadPathSettings | None = None,
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
        settings: ReadPathSettings | None = None,
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

    def memory_maintain(
        self,
        *,
        now: datetime | None = None,
        settings: MaintenanceSettings | None = None,
    ) -> MaintenanceResult: ...

    def memory_total_tokens(self) -> int: ...

    def memory_graph_build(self) -> GraphBuildResult: ...


def _shared_relation(relations: tuple[tuple[str, str], ...]) -> str | None:
    labels = {relation for _memory_id, relation in relations}
    if len(labels) == 1:
        return next(iter(labels))
    return None


def _proposition_key(content: str) -> str:
    return " ".join(content.split()).casefold()


def _candidate_from_rewrite(
    candidate: MemoryCandidate,
    piece: AtomicRewritePiece,
) -> MemoryCandidate | None:
    content = piece.content.strip()
    subject = piece.subject.strip()
    predicate = piece.predicate.strip()
    relation_object = piece.object.strip()
    if not all((content, subject, predicate, relation_object)):
        return None
    return MemoryCandidate(
        content=content,
        subject=subject,
        predicate=predicate,
        object=relation_object,
        type=candidate.type,
        importance=candidate.importance,
        confidence=candidate.confidence,
        role=candidate.role,
        debug_source_quote=candidate.debug_source_quote,
        debug_reason=candidate.debug_reason,
    )


def _with_current_statuses(
    filters: MemoryFilters | None,
) -> MemoryFilters:
    if filters is not None and filters.statuses:
        return filters
    if filters is None:
        return MemoryFilters(statuses=RETRIEVAL_MEMORY_STATUSES)
    return replace(filters, statuses=RETRIEVAL_MEMORY_STATUSES)


class AgentMemory:
    def __init__(
        self,
        store: PostgresMemoryStore,
        selector: MemorySelector,
        *,
        source: str,
        session_id: str,
        reranker: MemoryReranker | None = None,
        compressor: MemoryCompressor | None = None,
        relation_classifier: MemoryRelationClassifier | None = None,
        candidate_relation_classifier: (
            MemoryCandidateRelationClassifier | None
        ) = None,
        atomic_checker: MemoryAtomicChecker | None = None,
        atomic_rewriter: MemoryAtomicRewriter | None = None,
        dedup_classifier: MemoryDedupClassifier | None = None,
        graph_extractor: MemoryGraphExtractor | None = None,
        entity_judge: EntityJudge | None = None,
        entity_selector: EntitySelector | None = None,
    ) -> None:
        self._store = store
        self._selector = selector
        self._reranker = reranker
        self._compressor = compressor
        self._relation_classifier = relation_classifier
        self._candidate_relation_classifier = (
            candidate_relation_classifier
        )
        self._atomic_checker = atomic_checker
        self._atomic_rewriter = atomic_rewriter
        self._dedup_classifier = dedup_classifier
        self._graph_extractor = graph_extractor
        self._entity_judge = entity_judge
        self._entity_selector = entity_selector
        self._source = source
        self._session_id = session_id

    def memory_add(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        settings: WritePathSettings | None = None,
    ) -> MemoryAddResult:
        active = settings or WritePathSettings()
        weights = active.atomic_weights()
        selection = self._selector.extract_memory(conversation)
        exchange_id = uuid4().hex
        stored_memories: list[Memory] = []
        decisions: list[CandidateMemoryDecision] = []

        for candidate in selection.selected:
            self._process_selected_candidate(
                candidate,
                active=active,
                weights=weights,
                exchange_id=exchange_id,
                stored_memories=stored_memories,
                decisions=decisions,
                allow_rewrite=True,
            )

        return MemoryAddResult(
            selection=selection,
            stored_memories=tuple(stored_memories),
            decisions=tuple(decisions),
            not_selected_scores=tuple(
                score_atomic_memory(candidate.content, weights=weights)
                for candidate in selection.not_selected
            ),
        )

    def _process_selected_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        active: WritePathSettings,
        weights: AtomicScoreWeights,
        exchange_id: str,
        stored_memories: list[Memory],
        decisions: list[CandidateMemoryDecision],
        allow_rewrite: bool,
    ) -> None:
        atomic_score = score_atomic_memory(
            candidate.content,
            weights=weights,
        )
        verdict = None
        if needs_atomic_review(
            atomic_score.score,
            active.atomic_score_threshold,
        ):
            verdict = self._check_atomic_candidate(candidate)
            if verdict == "invalid":
                self._reject_or_rewrite_candidate(
                    candidate,
                    atomic_score=atomic_score,
                    active=active,
                    weights=weights,
                    exchange_id=exchange_id,
                    stored_memories=stored_memories,
                    decisions=decisions,
                    allow_rewrite=allow_rewrite,
                )
                return
        related = self._related_memories(candidate.content, active)
        if not related:
            stored = self._write_candidate(candidate, exchange_id)
            stored_memories.append(stored)
            decisions.append(
                CandidateMemoryDecision(
                    candidate=candidate,
                    related=(),
                    relation=None,
                    stored=stored,
                    atomic_score=atomic_score,
                    atomic_verdict=verdict,
                    from_rewrite=not allow_rewrite,
                )
            )
            return
        relations = self._classify_candidate(candidate.content, related)
        stored, links = self._apply_candidate_relation(
            candidate,
            related,
            relations,
            exchange_id,
        )
        if stored is not None:
            stored_memories.append(stored)
        decisions.append(
            CandidateMemoryDecision(
                candidate=candidate,
                related=tuple(related),
                relation=_shared_relation(relations),
                relations=relations,
                stored=stored,
                links=links,
                atomic_score=atomic_score,
                atomic_verdict=verdict,
                from_rewrite=not allow_rewrite,
            )
        )

    def _reject_or_rewrite_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        atomic_score: AtomicScore,
        active: WritePathSettings,
        weights: AtomicScoreWeights,
        exchange_id: str,
        stored_memories: list[Memory],
        decisions: list[CandidateMemoryDecision],
        allow_rewrite: bool,
    ) -> None:
        replacements: tuple[MemoryCandidate, ...] = ()
        if allow_rewrite:
            replacements = self._rewrite_atomic_candidate(candidate)
        decisions.append(
            CandidateMemoryDecision(
                candidate=candidate,
                related=(),
                relation=None,
                stored=None,
                atomic_score=atomic_score,
                atomic_verdict="invalid",
                rewritten=tuple(
                    replacement.content for replacement in replacements
                ),
            )
        )
        for replacement in replacements:
            self._process_selected_candidate(
                replacement,
                active=active,
                weights=weights,
                exchange_id=exchange_id,
                stored_memories=stored_memories,
                decisions=decisions,
                allow_rewrite=False,
            )

    def _rewrite_atomic_candidate(
        self,
        candidate: MemoryCandidate,
    ) -> tuple[MemoryCandidate, ...]:
        if self._atomic_rewriter is None:
            raise RuntimeError(
                "An invalid atomic memory needs an atomic rewriter."
            )
        pieces = self._atomic_rewriter.rewrite(candidate)
        replacements: list[MemoryCandidate] = []
        seen = {_proposition_key(candidate.content)}
        for piece in pieces:
            replacement = _candidate_from_rewrite(candidate, piece)
            if replacement is None:
                continue
            key = _proposition_key(replacement.content)
            if key in seen:
                continue
            seen.add(key)
            replacements.append(replacement)
        return tuple(replacements)

    def _check_atomic_candidate(self, candidate: MemoryCandidate) -> str:
        if self._atomic_checker is None:
            raise RuntimeError(
                "A memory at the atomic-score threshold needs an "
                "atomic checker."
            )
        verdict = self._atomic_checker.check(candidate)
        label = verdict.strip().lower()
        if label not in ATOMIC_CHECK_VERDICTS:
            raise RuntimeError(
                "The atomic checker returned an unknown verdict: "
                f"{verdict}"
            )
        return label

    def _related_memories(
        self,
        content: str,
        settings: WritePathSettings,
    ) -> list[MemoryMatch]:
        matches = self._store.memory_search(
            content,
            limit=settings.related_limit,
            min_similarity=settings.related_similarity,
            filters=MemoryFilters(
                statuses=RELATED_MEMORY_STATUSES,
                expired=None,
            ),
            above_min_similarity=True,
        )
        return [
            match
            for match in matches
            if similarity_above(
                match.similarity_score,
                settings.related_similarity,
            )
        ]

    def _classify_candidate(
        self,
        content: str,
        related: Sequence[MemoryMatch],
    ) -> tuple[tuple[str, str], ...]:
        if self._candidate_relation_classifier is None:
            raise RuntimeError(
                "Related memories need a candidate relation classifier."
            )
        labeled = self._candidate_relation_classifier.classify(
            content,
            related,
        )
        relations = {
            memory_id: relation.strip().lower()
            for memory_id, relation in labeled
        }
        expected = {match.memory.id for match in related}
        if set(relations) != expected:
            raise RuntimeError(
                "The candidate relation classifier did not label every "
                "related memory."
            )
        unknown = [
            relation
            for relation in relations.values()
            if relation not in CANDIDATE_RELATIONS
        ]
        if unknown:
            raise RuntimeError(
                "The candidate relation classifier returned an unknown "
                f"relation: {unknown[0]}"
            )
        return tuple(
            (match.memory.id, relations[match.memory.id]) for match in related
        )

    def _apply_candidate_relation(
        self,
        candidate: MemoryCandidate,
        related: Sequence[MemoryMatch],
        relations: tuple[tuple[str, str], ...],
        exchange_id: str,
    ) -> tuple[Memory | None, tuple[MemoryLink, ...]]:
        labels = dict(relations)
        equivalent_ids = [
            memory_id
            for memory_id, relation in relations
            if relation == "equivalent"
        ]
        if equivalent_ids:
            self._store.reinforce_memories(
                equivalent_ids,
                reinforced_at=datetime.now(UTC),
            )
        changing = [
            match
            for match in related
            if labels[match.memory.id]
            not in ("equivalent", "related_distinct")
        ]
        distinct = any(
            relation == "related_distinct" for _memory_id, relation in relations
        )
        if not changing and equivalent_ids:
            return None, ()
        if not changing and distinct:
            stored = self._write_candidate(candidate, exchange_id)
            return stored, ()
        if not changing:
            return None, ()
        new_status = (
            MemoryStatus.CONFLICTED
            if any(relation == "contradiction" for _memory_id, relation in relations)
            else MemoryStatus.ACTIVE
        )
        stored = self._write_candidate(
            candidate,
            exchange_id,
            status=new_status,
        )
        links = []
        for match in changing:
            effect = CANDIDATE_RELATION_EFFECTS[labels[match.memory.id]]
            if effect.link is None or effect.previous_status is None:
                continue
            links.append(
                self._store.record_memory_link(
                    from_memory_id=stored.id,
                    to_memory_id=match.memory.id,
                    relation=effect.link,
                    to_status=effect.previous_status,
                )
            )
        return stored, tuple(links)

    def _write_candidate(
        self,
        candidate: MemoryCandidate,
        exchange_id: str,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> Memory:
        return self._store.write_memory(
            candidate.content,
            memory_type=candidate.type,
            role=candidate.role,
            source=self._source,
            session_id=self._session_id,
            importance=candidate.importance,
            metadata={
                "confidence": candidate.confidence,
                "debug_source_quote": candidate.debug_source_quote,
                "exchange_id": exchange_id,
                "selector_model": self._selector.model_name,
            },
            status=status,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
        )

    def memory_search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        settings: ReadPathSettings | None = None,
    ) -> MemorySearchResult:
        active = (
            settings
            if settings is not None
            else ReadPathSettings(result_limit=limit)
        )
        semantic_matches = self._store.memory_search(
            query,
            limit=active.candidate_limit,
            min_similarity=min_similarity,
            filters=_with_current_statuses(filters),
        )
        scored_at = now or datetime.now(UTC)
        self._store.memory_mark_accessed(
            [match.memory.id for match in semantic_matches],
            accessed_at=scored_at,
        )
        scored_candidates = rank_by_retrieval_score(
            semantic_matches,
            limit=max(len(semantic_matches), 1),
            now=scored_at,
            semantic_weight=active.semantic_weight,
            recency_weight=active.recency_weight,
            importance_weight=active.importance_weight,
            lambda_per_day=active.recency_lambda_per_day,
        )
        rerank_pool = scored_candidates[: active.weighted_limit]
        if self._reranker is None or not rerank_pool:
            return MemorySearchResult(
                matches=tuple(scored_candidates[: active.result_limit]),
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
            matches=tuple(reranked[: active.result_limit]),
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
        settings: ReadPathSettings | None = None,
    ) -> MemoryContextResult:
        active = (
            settings
            if settings is not None
            else ReadPathSettings(
                result_limit=limit,
                token_budget=token_budget,
            )
        )
        search = self.memory_search(
            query,
            min_similarity=min_similarity,
            filters=filters,
            now=now,
            settings=active,
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
            duplicate_similarity=active.duplicate_similarity,
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
        if context_token_count(memories) > active.token_budget:
            if self._compressor is None:
                raise RuntimeError(
                    "Memory context exceeds the token budget and no "
                    "compressor is configured."
                )
            compressed = self._compressor.compress(
                memories,
                token_budget=active.token_budget,
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

    def memory_total_tokens(self) -> int:
        return self._store.total_token_count()

    def memory_maintain(
        self,
        *,
        now: datetime | None = None,
        settings: MaintenanceSettings | None = None,
    ) -> MaintenanceResult:
        return MemoryMaintenance(
            self._store,
            dedup_classifier=self._dedup_classifier,
        ).run(
            now=now or datetime.now(UTC),
            settings=settings,
        )

    def memory_graph_build(
        self,
        settings: EntityResolutionSettings | None = None,
    ) -> GraphBuildResult:
        if self._graph_extractor is None:
            return GraphBuildResult(0, 0, 0)
        return MemoryGraph(
            self._store,
            self._graph_extractor,
            self._entity_judge,
            self._entity_selector,
        ).build(settings)

    def clear(self) -> None:
        self._store.clear()
