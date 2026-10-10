from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

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


def entity_name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


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
    ) -> None:
        self._store = store
        self._extractor = extractor

    def build(self) -> GraphBuildResult:
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
            edges = _unique_edges(self._extractor.extract(memory))
            self._store.replace_memory_graph(
                memory.id,
                source_key,
                [
                    (edge.from_entity, edge.relation, edge.to_entity)
                    for edge in edges
                ],
            )
            built_count += 1
        self._store.delete_orphan_graph_entities()
        return GraphBuildResult(
            entity_count=self._store.graph_entity_count(),
            edge_count=self._store.graph_edge_count(),
            built_count=built_count,
        )


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
