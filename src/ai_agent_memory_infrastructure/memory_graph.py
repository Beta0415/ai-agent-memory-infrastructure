from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from ai_agent_memory_infrastructure.memory_timing import runtime_span, timed
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_entity_judge import (
    ENTITY_JUDGE_VERDICTS,
    EntityJudge,
)
from ai_agent_memory_infrastructure.memory_entity_selector import (
    EntitySelector,
)
from ai_agent_memory_infrastructure.memory_entity_similarity import (
    EntityComparison,
    EntityResolutionSettings,
    entity_name_key,
    passes_entity_threshold,
    score_new_mentions,
)
from ai_agent_memory_infrastructure.memory_store import Memory
from ai_agent_memory_infrastructure.postgres_memory_store import (
    PostgresMemoryStore,
)


class GraphEdgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_entity: str = Field(min_length=1, max_length=300)
    relation: str = Field(min_length=1, max_length=300)
    to_entity: str = Field(min_length=1, max_length=300)


class GraphExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edges: list[GraphEdgeOutput]


MEMORY_GRAPH_INSTRUCTIONS = """
You turn one stored atomic memory into graph edges.

Read only that memory. Do not add a fact it does not state.

An entity is a person, organization, place, time, subject, tool, or
other thing named in the memory. Copy the entity text from the memory.
Do not rename it. If the memory says "NYU", the entity is "NYU". If it
says "New York University", the entity is "New York University".
Do not expand an abbreviation that is not written in the memory.

When a longer name contains an organization that is written inside it,
emit both entities and a "part of" edge from the longer name to that
organization. "WeBank's Risk Division" includes the entity "WeBank".
"the WeBank AI assistant" includes the entity "WeBank". Do not split a
place out of a longer place name when the shorter words are not the
organization, such as "York" from "New York".

Return one edge for each connection the memory states. The endpoints
must be entities copied from the memory. A qualifier such as a school,
employer, place, time, or purpose is its own entity when the memory
states it.

Example memory: "The user is majoring in Economics at New York University."
Edges:
- The user | majoring in | Economics
- The user | majoring at | New York University

Example memory: "Beta Chen built an AI assistant for WeBank's Risk Division that supported calculations."
Edges:
- Beta Chen | built | an AI assistant
- an AI assistant | for | WeBank's Risk Division
- an AI assistant | supported | calculations
- WeBank's Risk Division | part of | WeBank
""".strip()

GRAPH_EXTRACTION_VERSION = "2"


@dataclass(frozen=True)
class GraphEdgeDraft:
    from_entity: str
    relation: str
    to_entity: str


@dataclass(frozen=True)
class GraphBuildResult:
    entity_count: int
    edge_count: int
    built_count: int


class MemoryGraphExtractor(Protocol):
    def extract(self, memory: Memory) -> Sequence[GraphEdgeDraft]: ...


class OpenAIMemoryGraphExtractor:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.6-luna",
    ) -> None:
        self._model = model
        self._openai = OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @timed("Extractor")
    def extract(self, memory: Memory) -> Sequence[GraphEdgeDraft]:
        if not memory.content.strip():
            raise ValueError("Graph extraction needs memory content.")
        payload = {
            "content": memory.content,
            "subject": memory.subject,
            "predicate": memory.predicate,
            "object": memory.object,
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_GRAPH_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=GraphExtractionOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The graph extractor returned no structured output."
            )
        return [
            GraphEdgeDraft(
                from_entity=edge.from_entity.strip(),
                relation=edge.relation.strip(),
                to_entity=edge.to_entity.strip(),
            )
            for edge in response.output_parsed.edges
            if edge.from_entity.strip()
            and edge.relation.strip()
            and edge.to_entity.strip()
        ]


@dataclass(frozen=True)
class MemoryConnectionView:
    memories: tuple[tuple[str, str], ...]
    links: tuple[tuple[str, str, tuple[str, ...]], ...]
    broad_entities: tuple[tuple[str, int], ...]
    groups: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    hubs: tuple[tuple[str, str, tuple[str, ...]], ...] = ()


def memory_connections(
    mentions: Sequence[tuple[str, str, str, str]],
    *,
    broad_limit: int = 20,
) -> MemoryConnectionView:
    contents: dict[str, str] = {}
    members_by_entity: dict[str, set[str]] = {}
    name_by_entity: dict[str, str] = {}
    for memory_id, content, entity_id, name in mentions:
        contents[memory_id] = content
        members_by_entity.setdefault(entity_id, set()).add(memory_id)
        name_by_entity[entity_id] = name
    shared: dict[tuple[str, str], list[str]] = {}
    groups: list[tuple[str, str, tuple[str, ...]]] = []
    hubs: list[tuple[str, str, tuple[str, ...]]] = []
    for entity_id, members in members_by_entity.items():
        if len(members) < 2:
            continue
        record = (
            entity_id,
            name_by_entity[entity_id],
            tuple(sorted(members)),
        )
        groups.append(record)
        if len(members) > broad_limit:
            hubs.append(record)
            continue
        ordered = sorted(members)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                shared.setdefault((left, right), []).append(
                    name_by_entity[entity_id]
                )
    ordered_groups = tuple(
        sorted(groups, key=lambda item: (-len(item[2]), item[1], item[0]))
    )
    ordered_hubs = tuple(
        sorted(hubs, key=lambda item: (-len(item[2]), item[1], item[0]))
    )
    return MemoryConnectionView(
        memories=tuple(sorted(contents.items())),
        links=tuple(
            (left, right, tuple(sorted(names)))
            for (left, right), names in sorted(shared.items())
        ),
        broad_entities=tuple(
            (name, len(memory_ids)) for _entity_id, name, memory_ids in ordered_hubs
        ),
        groups=ordered_groups,
        hubs=ordered_hubs,
    )


def graph_source_key(memory: Memory) -> str:
    payload = "\n".join(
        (
            GRAPH_EXTRACTION_VERSION,
            memory.content,
            memory.subject,
            memory.predicate,
            memory.object,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class MemoryGraph:
    def __init__(
        self,
        store: PostgresMemoryStore,
        extractor: MemoryGraphExtractor,
        judge: EntityJudge | None = None,
        selector: EntitySelector | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._judge = judge
        self._selector = selector
        self._resolution = EntityResolutionSettings()

    def build(
        self,
        settings: EntityResolutionSettings | None = None,
    ) -> GraphBuildResult:
        self._resolution = settings or EntityResolutionSettings()
        with runtime_span("Load Memories"):
            current = self._store.graph_source_memories()
            self._store.delete_graph_outside(
                [memory.id for memory in current]
            )
        built_count = 0
        stored_keys = self._store.graph_source_keys()
        for memory in current:
            source_key = graph_source_key(memory)
            if stored_keys.get(memory.id) == source_key:
                continue
            with runtime_span("Memory"):
                resolved, embeddings = self._resolve_mentions(
                    _unique_edges(self._extractor.extract(memory))
                )
                with runtime_span("Store Graph"):
                    self._store.replace_memory_graph(
                        memory.id,
                        source_key,
                        [
                            (edge.from_entity, edge.relation, edge.to_entity)
                            for edge in resolved
                        ],
                        embeddings,
                    )
            built_count += 1
        with runtime_span("Drop Orphan Entities"):
            self._store.delete_orphan_graph_entities()
        return GraphBuildResult(
            entity_count=self._store.graph_entity_count(),
            edge_count=self._store.graph_edge_count(),
            built_count=built_count,
        )

    def _resolve_mentions(
        self,
        edges: Sequence[GraphEdgeDraft],
    ) -> tuple[tuple[GraphEdgeDraft, ...], dict[str, list[float]]]:
        with runtime_span("Resolve Names"):
            return self._resolve_mentions_timed(edges)

    def _resolve_mentions_timed(
        self,
        edges: Sequence[GraphEdgeDraft],
    ) -> tuple[tuple[GraphEdgeDraft, ...], dict[str, list[float]]]:
        existing = self._store.graph_entity_names()
        existing_by_key = {
            entity_name_key(name): name for _entity_id, name, _embedding in existing
        }
        aliases = self._store.graph_entity_alias_names()
        replacements: dict[str, str] = {}
        new_mentions: list[str] = []
        for mention in _mention_names(edges):
            key = entity_name_key(mention)
            if key in existing_by_key:
                continue
            if key in aliases:
                replacements[key] = aliases[key]
                continue
            new_mentions.append(mention)
        embeddings: dict[str, list[float]] = {}
        if new_mentions and existing:
            comparisons, embeddings = self._score_mentions(new_mentions, existing)
            grouped: dict[str, list[EntityComparison]] = {}
            for comparison in comparisons:
                grouped.setdefault(
                    entity_name_key(comparison.mention_name),
                    [],
                ).append(comparison)
            for mention in new_mentions:
                key = entity_name_key(mention)
                chosen, absorbed = self._chosen_entity(
                    mention,
                    grouped.get(key, []),
                )
                if chosen is None:
                    continue
                replacements[key] = chosen.existing_name
                for name in absorbed:
                    replacements[entity_name_key(name)] = chosen.existing_name
                self._store.record_graph_entity_alias(
                    mention,
                    chosen.existing_entity_id,
                )
        elif new_mentions:
            embeddings = self._embed_names(new_mentions)
        return _rewrite_edges(edges, replacements), embeddings

    def _score_mentions(
        self,
        new_mentions: Sequence[str],
        existing: Sequence[tuple[str, str, list[float] | None]],
    ) -> tuple[tuple[EntityComparison, ...], dict[str, list[float]]]:
        vectors_by_key: dict[str, list[float]] = {}
        missing: list[tuple[str | None, str]] = []
        for entity_id, name, embedding in existing:
            key = entity_name_key(name)
            if embedding is None:
                missing.append((entity_id, name.strip()))
            else:
                vectors_by_key[key] = embedding
        for name in new_mentions:
            key = entity_name_key(name)
            if key not in vectors_by_key:
                missing.append((None, name.strip()))
        if missing:
            texts: list[str] = []
            seen: set[str] = set()
            for _entity_id, name in missing:
                key = entity_name_key(name)
                if key in seen:
                    continue
                seen.add(key)
                texts.append(name)
            vectors = self._store.embed_texts(texts)
            embedded = {
                entity_name_key(text): vector
                for text, vector in zip(texts, vectors, strict=True)
            }
            vectors_by_key.update(embedded)
            self._store.save_graph_entity_embeddings(
                [
                    (entity_id, embedded[entity_name_key(name)])
                    for entity_id, name in missing
                    if entity_id is not None
                ]
            )
        return (
            score_new_mentions(
                new_mentions,
                [(entity_id, name) for entity_id, name, _embedding in existing],
                vectors_by_key,
            ),
            {
                entity_name_key(name): vectors_by_key[entity_name_key(name)]
                for name in new_mentions
            },
        )

    def _embed_names(self, names: Sequence[str]) -> dict[str, list[float]]:
        texts: list[str] = []
        seen: set[str] = set()
        for name in names:
            key = entity_name_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            texts.append(name.strip())
        vectors = self._store.embed_texts(texts)
        return {
            entity_name_key(text): vector
            for text, vector in zip(texts, vectors, strict=True)
        }

    def _chosen_entity(
        self,
        mention: str,
        comparisons: Sequence[EntityComparison],
    ) -> tuple[EntityComparison | None, tuple[str, ...]]:
        passing = [
            comparison
            for comparison in comparisons
            if passes_entity_threshold(comparison, self._resolution)
        ]
        if not passing:
            return None, ()
        if self._judge is None:
            raise RuntimeError(
                "An entity pair above the similarity threshold needs an "
                "entity judge."
            )
        matched = [
            comparison
            for comparison in passing
            if self._judge_pair(mention, comparison) == "same_entity"
        ]
        if not matched:
            return None, ()
        if len(matched) == 1:
            return matched[0], ()
        chosen = self._select_entity(matched)
        absorbed = tuple(
            comparison.existing_name
            for comparison in matched
            if comparison.existing_entity_id != chosen.existing_entity_id
        )
        self._store.merge_graph_entities(
            chosen.existing_entity_id,
            [
                comparison.existing_entity_id
                for comparison in matched
                if comparison.existing_entity_id != chosen.existing_entity_id
            ],
        )
        return chosen, absorbed

    def _select_entity(
        self,
        matched: Sequence[EntityComparison],
    ) -> EntityComparison:
        if self._selector is None:
            raise RuntimeError(
                "Several existing entities judged the same need an entity "
                "selector."
            )
        chosen_name = self._selector.select(
            [comparison.existing_name for comparison in matched]
        )
        chosen_key = entity_name_key(chosen_name)
        for comparison in matched:
            if entity_name_key(comparison.existing_name) == chosen_key:
                return comparison
        raise RuntimeError(
            "The entity selector returned a name that was not one of the "
            f"entities: {chosen_name}"
        )

    def _judge_pair(self, mention: str, comparison: EntityComparison) -> str:
        assert self._judge is not None
        verdict = self._judge.judge(mention, comparison.existing_name)
        if verdict not in ENTITY_JUDGE_VERDICTS:
            raise RuntimeError(
                "The entity judge returned an unknown verdict: "
                f"{verdict}"
            )
        return verdict


def _rewrite_edges(
    edges: Sequence[GraphEdgeDraft],
    replacements: dict[str, str],
) -> tuple[GraphEdgeDraft, ...]:
    rewritten: list[GraphEdgeDraft] = []
    for edge in edges:
        rewritten.append(
            GraphEdgeDraft(
                from_entity=replacements.get(
                    entity_name_key(edge.from_entity),
                    edge.from_entity,
                ),
                relation=edge.relation,
                to_entity=replacements.get(
                    entity_name_key(edge.to_entity),
                    edge.to_entity,
                ),
            )
        )
    return tuple(rewritten)


def _mention_names(edges: Sequence[GraphEdgeDraft]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        for name in (edge.from_entity, edge.to_entity):
            key = entity_name_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            names.append(name.strip())
    return tuple(names)


def _unique_edges(
    edges: Sequence[GraphEdgeDraft],
) -> tuple[GraphEdgeDraft, ...]:
    unique: list[GraphEdgeDraft] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (
            entity_name_key(edge.from_entity),
            " ".join(edge.relation.split()).casefold(),
            entity_name_key(edge.to_entity),
        )
        if key in seen or not all(key):
            continue
        seen.add(key)
        unique.append(edge)
    return tuple(unique)
