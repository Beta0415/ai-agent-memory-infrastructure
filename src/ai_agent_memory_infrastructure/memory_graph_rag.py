from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from openai import OpenAI

from ai_agent_memory_infrastructure.memory_entity_similarity import (
    EntityComparison,
    EntityResolutionSettings,
    embedding_cosine_similarity,
    entity_name_key,
    jaccard_token_similarity,
    normalized_levenshtein_similarity,
    passes_entity_threshold,
)
from ai_agent_memory_infrastructure.memory_episode import prevalence_threshold
from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory
from ai_agent_memory_infrastructure.memory_store import (
    Memory,
    MemoryStatus,
    MemoryType,
)

GRAPH_RAG_MAX_HOPS = 3
GRAPH_RAG_FINAL_MEMORY_TOP_K = 10
GRAPH_RAG_MAX_EDGES_PER_SEED = 30
GRAPH_RAG_FRONTIER_CAP = 5
GRAPH_RAG_MIN_HOP_SIMILARITY = 0.50

GRAPH_RAG_ANSWER_INSTRUCTIONS = """
Answer the user's question using only the provided memory graph evidence.
Do not invent or infer unsupported personal facts.
If the graph evidence is insufficient, say so.
""".strip()


@dataclass(frozen=True)
class GraphRagSettings:
    final_top_k: int = GRAPH_RAG_FINAL_MEMORY_TOP_K
    max_edges_per_seed: int = GRAPH_RAG_MAX_EDGES_PER_SEED
    frontier_cap: int = GRAPH_RAG_FRONTIER_CAP
    min_hop_similarity: float = GRAPH_RAG_MIN_HOP_SIMILARITY
    max_hops: int = GRAPH_RAG_MAX_HOPS
    prevalence_baseline: float = 0.20
    prevalence_decay_scale: int = 100

    def __post_init__(self) -> None:
        if self.final_top_k < 1:
            raise ValueError("Graph RAG final top K must be at least 1.")
        if self.max_edges_per_seed < 1:
            raise ValueError("Graph RAG edge cap must be at least 1.")
        if self.frontier_cap < 1:
            raise ValueError("Graph RAG frontier cap must be at least 1.")
        if not 0.0 <= self.min_hop_similarity <= 1.0:
            raise ValueError("Graph RAG hop similarity must be between 0 and 1.")
        if not 1 <= self.max_hops <= GRAPH_RAG_MAX_HOPS:
            raise ValueError("Graph RAG hop count must be from 1 to 3.")
        if self.prevalence_decay_scale < 1:
            raise ValueError("Prevalence decay scale must be at least 1.")


@dataclass(frozen=True)
class GraphRagSeed:
    entity_id: str
    name: str
    explicit: bool
    broad: bool


@dataclass(frozen=True)
class GraphRagPath:
    seed_entity_id: str
    edge_id: str
    source_entity_id: str
    source_name: str
    relation: str
    target_entity_id: str
    target_name: str
    memory_id: str
    hop: int = 1


@dataclass(frozen=True)
class GraphRagMemory:
    memory_id: str
    content: str
    semantic_similarity: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class GraphRagResult:
    query: str
    seed_entities: tuple[GraphRagSeed, ...]
    paths: tuple[GraphRagPath, ...]
    retrieved_memories: tuple[GraphRagMemory, ...]
    answer: str

    def as_record(self) -> dict[str, object]:
        return {
            "query": self.query,
            "seed_entities": [
                {
                    "entity_id": seed.entity_id,
                    "name": seed.name,
                    "explicit": seed.explicit,
                    "broad": seed.broad,
                }
                for seed in self.seed_entities
            ],
            "paths": [
                {
                    "seed_entity_id": path.seed_entity_id,
                    "edge_id": path.edge_id,
                    "source_entity_id": path.source_entity_id,
                    "source_name": path.source_name,
                    "relation": path.relation,
                    "target_entity_id": path.target_entity_id,
                    "target_name": path.target_name,
                    "memory_id": path.memory_id,
                    "hop": path.hop,
                }
                for path in self.paths
            ],
            "retrieved_memories": [
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "semantic_similarity": memory.semantic_similarity,
                    "rerank_score": memory.rerank_score,
                }
                for memory in self.retrieved_memories
            ],
            "answer": self.answer,
        }


class GraphRagAnswerer(Protocol):
    def answer(
        self,
        query: str,
        paths: Sequence[GraphRagPath],
        memories: Sequence[GraphRagMemory],
    ) -> str: ...


class OpenAIGraphRagAnswerer:
    def __init__(self, *, api_key: str, model: str = "gpt-5.6-sol") -> None:
        self._model = model
        self._openai = OpenAI(api_key=api_key)

    def answer(
        self,
        query: str,
        paths: Sequence[GraphRagPath],
        memories: Sequence[GraphRagMemory],
    ) -> str:
        response = self._openai.responses.create(
            model=self._model,
            instructions=GRAPH_RAG_ANSWER_INSTRUCTIONS,
            input=json.dumps(
                {
                    "question": query,
                    "graph_paths": [
                        {
                            "source": path.source_name,
                            "relation": path.relation,
                            "target": path.target_name,
                            "memory_id": path.memory_id,
                        }
                        for path in paths
                        if path.memory_id in {memory.memory_id for memory in memories}
                    ],
                    "memories": [
                        {
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                        }
                        for memory in memories
                    ],
                },
                ensure_ascii=False,
            ),
            store=False,
        )
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("Graph RAG returned an empty answer.")
        return text


def run_graph_rag(
    store: object,
    query: str,
    *,
    reranker: object | None,
    answerer: GraphRagAnswerer | None,
    settings: GraphRagSettings | None = None,
    resolution: EntityResolutionSettings | None = None,
) -> GraphRagResult:
    """Answer one question from the current graph without writing anything."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Graph RAG needs a question.")
    active = settings or GraphRagSettings()
    entities = store.graph_rag_entities()
    aliases = store.graph_rag_aliases()
    degrees = store.graph_rag_degrees()
    total = store.active_memory_count()
    bound = prevalence_threshold(
        total,
        baseline=active.prevalence_baseline,
        decay_scale=active.prevalence_decay_scale,
    )
    explicit = _explicit_seeds(cleaned, entities, aliases, degrees, total, bound)
    query_vector: Sequence[float] | None = None
    if explicit:
        seeds = explicit
    else:
        query_vector = tuple(store.embed_texts([cleaned])[0])
        seeds = _semantic_seed(
            cleaned,
            query_vector,
            entities,
            degrees,
            total,
            bound,
            resolution or EntityResolutionSettings(),
        )
    walk_from = tuple(seed for seed in seeds if not seed.broad)
    if not seeds:
        return _empty(cleaned, (), (), "No stored entity matched this question.")
    if not walk_from:
        return _empty(
            cleaned,
            seeds,
            (),
            "The graph evidence is insufficient to answer this.",
        )
    if query_vector is None:
        query_vector = tuple(store.embed_texts([cleaned])[0])
    names = {entity_id: name for entity_id, name, _embedding in entities}
    paths = _walk(
        store,
        walk_from,
        names,
        degrees,
        total,
        bound,
        query_vector,
        active,
    )
    memory_ids = list(dict.fromkeys(path.memory_id for path in paths))
    ranked = _rank_memories(
        store.graph_rag_memory_similarities(memory_ids, query_vector)
    )
    if not ranked:
        return _empty(
            cleaned,
            seeds,
            paths,
            "The graph evidence is insufficient to answer this.",
        )
    if reranker is None or not hasattr(reranker, "rerank"):
        raise RuntimeError("Graph RAG needs a reranker.")
    scores = reranker.rerank(cleaned, [_as_retrieved(memory) for memory in ranked])
    ranked = tuple(
        GraphRagMemory(
            memory_id=memory.memory_id,
            content=memory.content,
            semantic_similarity=memory.semantic_similarity,
            rerank_score=scores[memory.memory_id],
        )
        for memory in ranked
    )
    ranked = tuple(
        sorted(
            ranked,
            key=lambda memory: (
                -(memory.rerank_score or 0.0),
                -memory.semantic_similarity,
                memory.memory_id,
            ),
        )
    )
    selected = ranked[: active.final_top_k]
    if not selected:
        return _empty(
            cleaned,
            seeds,
            paths,
            "The graph evidence is insufficient to answer this.",
        )
    if answerer is None:
        raise RuntimeError("Graph RAG needs an answerer.")
    answer = answerer.answer(cleaned, paths, selected)
    return GraphRagResult(
        query=cleaned,
        seed_entities=seeds,
        paths=paths,
        retrieved_memories=selected,
        answer=answer,
    )


def _empty(
    query: str,
    seeds: Sequence[GraphRagSeed],
    paths: Sequence[GraphRagPath],
    answer: str,
) -> GraphRagResult:
    return GraphRagResult(
        query=query,
        seed_entities=tuple(seeds),
        paths=tuple(paths),
        retrieved_memories=(),
        answer=answer,
    )


def _match_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in entity_name_key(text).split():
        token = "".join(character for character in raw if character.isalnum())
        if token:
            tokens.append(token)
    return tokens


def _explicit_seeds(
    query: str,
    entities: Sequence[tuple[str, str, tuple[float, ...] | None]],
    aliases: Sequence[tuple[str, str, str]],
    degrees: Mapping[str, int],
    total: int,
    bound: float,
) -> tuple[GraphRagSeed, ...]:
    query_tokens = _match_tokens(query)
    spans: list[tuple[int, int, str, str]] = []
    for entity_id, name, _embedding in entities:
        spans.extend(_token_spans(query_tokens, name, entity_id, name))
    for alias_name, entity_id, canonical in aliases:
        spans.extend(_token_spans(query_tokens, alias_name, entity_id, canonical))
    spans.sort(key=lambda span: (span[0] - span[1], span[0], span[3]))
    chosen: list[tuple[int, int, str, str]] = []
    for span in spans:
        if any(_overlaps(span, kept) for kept in chosen):
            continue
        chosen.append(span)
    seeds: dict[str, GraphRagSeed] = {}
    for _start, _end, entity_id, name in chosen:
        degree = degrees.get(entity_id, 0)
        seeds.setdefault(
            entity_id,
            GraphRagSeed(
                entity_id=entity_id,
                name=name,
                explicit=True,
                broad=_is_broad(degree, total, bound),
            ),
        )
    return tuple(seeds.values())


def _semantic_seed(
    query: str,
    query_vector: Sequence[float],
    entities: Sequence[tuple[str, str, tuple[float, ...] | None]],
    degrees: Mapping[str, int],
    total: int,
    bound: float,
    resolution: EntityResolutionSettings,
) -> tuple[GraphRagSeed, ...]:
    best: EntityComparison | None = None
    best_score = -1.0
    for entity_id, name, embedding in entities:
        if embedding is None:
            embedding_score = 0.0
        else:
            embedding_score = embedding_cosine_similarity(query_vector, embedding)
        comparison = EntityComparison(
            mention_name=query,
            existing_entity_id=entity_id,
            existing_name=name,
            levenshtein=normalized_levenshtein_similarity(query, name),
            jaccard=jaccard_token_similarity(query, name),
            embedding=embedding_score,
        )
        if not passes_entity_threshold(comparison, resolution):
            continue
        score = max(
            comparison.levenshtein,
            comparison.jaccard,
            comparison.embedding,
        )
        if score > best_score:
            best = comparison
            best_score = score
    if best is None:
        return ()
    degree = degrees.get(best.existing_entity_id, 0)
    broad = _is_broad(degree, total, bound)
    if broad:
        return ()
    return (
        GraphRagSeed(
            entity_id=best.existing_entity_id,
            name=best.existing_name,
            explicit=False,
            broad=False,
        ),
    )


def _walk(
    store: object,
    seeds: Sequence[GraphRagSeed],
    names: Mapping[str, str],
    degrees: Mapping[str, int],
    total: int,
    bound: float,
    query_vector: Sequence[float],
    settings: GraphRagSettings,
) -> tuple[GraphRagPath, ...]:
    """Walk up to 3 hops through specific entities.

    An edge continues the walk only when its source memory is similar enough
    to the question. Only the best of those edges are followed. A broad
    entity is never entered, and an entity is entered only once.
    """
    queue: list[tuple[str, int, str]] = [
        (seed.entity_id, 1, seed.entity_id) for seed in seeds
    ]
    entered = {seed.entity_id for seed in seeds}
    expanded: set[str] = set()
    seen_edges: set[str] = set()
    paths: list[GraphRagPath] = []
    while queue:
        entity_id, hop, origin = queue.pop(0)
        if entity_id in expanded or hop > settings.max_hops:
            continue
        if _is_broad(degrees.get(entity_id, 0), total, bound):
            expanded.add(entity_id)
            continue
        expanded.add(entity_id)
        edges = store.graph_rag_edges([entity_id])
        fresh = [edge for edge in edges if edge[0] not in seen_edges]
        if not fresh:
            continue
        similarities = {
            memory_id: similarity
            for memory_id, _content, similarity in store.graph_rag_memory_similarities(
                [edge[1] for edge in fresh],
                query_vector,
            )
        }
        ranked_edges = sorted(
            fresh,
            key=lambda edge: (-similarities.get(edge[1], 0.0), edge[0]),
        )
        passing = [
            edge
            for edge in ranked_edges
            if similarities.get(edge[1], 0.0) >= settings.min_hop_similarity
        ][: settings.max_edges_per_seed]
        for edge in passing:
            edge_id, memory_id, relation, from_id, from_name, to_id, to_name = edge
            seen_edges.add(edge_id)
            paths.append(
                GraphRagPath(
                    seed_entity_id=origin,
                    edge_id=edge_id,
                    source_entity_id=from_id,
                    source_name=from_name,
                    relation=relation,
                    target_entity_id=to_id,
                    target_name=to_name,
                    memory_id=memory_id,
                    hop=hop,
                )
            )
        if hop >= settings.max_hops:
            continue
        followed = 0
        for edge in passing:
            if followed >= settings.frontier_cap:
                break
            followed += 1
            _enqueue_neighbor(
                edge,
                entity_id,
                hop,
                origin,
                names,
                degrees,
                total,
                bound,
                entered,
                queue,
            )
    return tuple(paths)


def _enqueue_neighbor(
    edge: tuple[str, str, str, str, str, str, str],
    entity_id: str,
    hop: int,
    origin: str,
    names: Mapping[str, str],
    degrees: Mapping[str, int],
    total: int,
    bound: float,
    entered: set[str],
    queue: list[tuple[str, int, str]],
) -> None:
    _edge_id, _memory_id, _relation, from_id, _from_name, to_id, _to_name = edge
    neighbor = to_id if from_id == entity_id else from_id
    if neighbor == entity_id or neighbor in entered:
        return
    if neighbor not in names:
        return
    if _is_broad(degrees.get(neighbor, 0), total, bound):
        return
    entered.add(neighbor)
    queue.append((neighbor, hop + 1, origin))


def _rank_memories(
    rows: Sequence[tuple[str, str, float]],
) -> tuple[GraphRagMemory, ...]:
    ordered = sorted(rows, key=lambda row: (-row[2], row[0]))
    return tuple(
        GraphRagMemory(
            memory_id=memory_id,
            content=content,
            semantic_similarity=similarity,
        )
        for memory_id, content, similarity in ordered
    )


def _as_retrieved(memory: GraphRagMemory) -> RetrievedMemory:
    moment = datetime.now(UTC)
    return RetrievedMemory(
        memory=Memory(
            id=memory.memory_id,
            content=memory.content,
            memory_type=MemoryType.FACT,
            role="user",
            source="graph_rag",
            session_id="graph_rag",
            importance=0.0,
            created_at=moment,
            updated_at=moment,
            metadata={},
            status=MemoryStatus.ACTIVE,
        ),
        semantic_score=memory.semantic_similarity,
        recency_score=0.0,
        retrieval_score=memory.semantic_similarity,
        final_score=memory.semantic_similarity,
    )


def _is_broad(degree: int, total: int, bound: float) -> bool:
    if total < 1 or degree < 1:
        return False
    return (degree / total) + 1e-12 > bound


def _token_spans(
    query_tokens: Sequence[str],
    name: str,
    entity_id: str,
    canonical_name: str,
) -> list[tuple[int, int, str, str]]:
    name_tokens = _match_tokens(name)
    if not name_tokens or len(name_tokens) > len(query_tokens):
        return []
    spans: list[tuple[int, int, str, str]] = []
    width = len(name_tokens)
    for start in range(len(query_tokens) - width + 1):
        if list(query_tokens[start : start + width]) == name_tokens:
            spans.append((start, start + width, entity_id, canonical_name))
    return spans


def _overlaps(
    left: tuple[int, int, str, str],
    right: tuple[int, int, str, str],
) -> bool:
    return left[0] < right[1] and right[0] < left[1]
