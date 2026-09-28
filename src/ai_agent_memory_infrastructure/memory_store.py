from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from openai import OpenAI


class MemoryType(StrEnum):
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    INSTRUCTION = "instruction"


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    memory_type: MemoryType
    role: str
    source: str
    session_id: str
    importance: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object]


@dataclass(frozen=True)
class MemoryMatch:
    memory: Memory
    similarity_score: float


@dataclass(frozen=True)
class MemoryFilters:
    memory_types: tuple[MemoryType, ...] = ()
    roles: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    min_importance: float | None = None


class MemoryAPI(Protocol):
    def memory_add(
        self,
        content: str,
        *,
        memory_type: MemoryType,
        role: str,
        source: str,
        session_id: str,
        importance: float = 0.5,
        metadata: Mapping[str, object] | None = None,
    ) -> Memory: ...

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


class ChromaMemoryStore:
    def __init__(
        self,
        *,
        path: str | Path,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "agent_memory",
    ) -> None:
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._openai = OpenAI(api_key=api_key)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=list(texts),
        )
        return [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]

    @staticmethod
    def _validate_memory(
        *,
        content: str,
        role: str,
        source: str,
        session_id: str,
        importance: float,
        metadata: Mapping[str, object],
    ) -> None:
        if not content.strip():
            raise ValueError("Memory content cannot be empty.")
        if not role.strip():
            raise ValueError("Memory role cannot be empty.")
        if not source.strip():
            raise ValueError("Memory source cannot be empty.")
        if not session_id.strip():
            raise ValueError("Memory session ID cannot be empty.")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("Memory importance must be between 0 and 1.")
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("Memory metadata must be JSON serializable.") from exc

    @staticmethod
    def _encode_metadata(memory: Memory) -> dict[str, str | float]:
        return {
            "memory_type": memory.memory_type.value,
            "role": memory.role,
            "source": memory.source,
            "session_id": memory.session_id,
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "metadata_json": json.dumps(
                memory.metadata, separators=(",", ":"), sort_keys=True
            ),
        }

    @staticmethod
    def _parse_timestamp(value: object, fallback: datetime) -> datetime:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return fallback

    @classmethod
    def _decode_memory(
        cls,
        memory_id: str,
        content: str,
        raw_metadata: Mapping[str, object],
    ) -> Memory:
        now = datetime.now(UTC)
        created_at = cls._parse_timestamp(
            raw_metadata.get("created_at"), now
        )
        updated_at = cls._parse_timestamp(
            raw_metadata.get("updated_at"), created_at
        )
        try:
            memory_type = MemoryType(
                str(raw_metadata.get("memory_type", "conversation"))
            )
        except ValueError:
            memory_type = MemoryType.CONVERSATION

        stored_metadata: dict[str, object] = {}
        metadata_json = raw_metadata.get("metadata_json")
        if isinstance(metadata_json, str):
            try:
                decoded = json.loads(metadata_json)
                if isinstance(decoded, dict):
                    stored_metadata = decoded
            except json.JSONDecodeError:
                pass

        try:
            importance = float(raw_metadata.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5

        return Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            role=str(raw_metadata.get("role", "unknown")),
            source=str(raw_metadata.get("source", "legacy")),
            session_id=str(raw_metadata.get("session_id", "legacy")),
            importance=importance,
            created_at=created_at,
            updated_at=updated_at,
            metadata=stored_metadata,
        )

    def memory_add(
        self,
        content: str,
        *,
        memory_type: MemoryType,
        role: str,
        source: str,
        session_id: str,
        importance: float = 0.5,
        metadata: Mapping[str, object] | None = None,
    ) -> Memory:
        stored_metadata = dict(metadata or {})
        self._validate_memory(
            content=content,
            role=role,
            source=source,
            session_id=session_id,
            importance=importance,
            metadata=stored_metadata,
        )
        now = datetime.now(UTC)
        memory = Memory(
            id=uuid4().hex,
            content=content,
            memory_type=memory_type,
            role=role,
            source=source,
            session_id=session_id,
            importance=importance,
            created_at=now,
            updated_at=now,
            metadata=stored_metadata,
        )
        self._collection.add(
            ids=[memory.id],
            embeddings=self._embed([memory.content]),
            documents=[memory.content],
            metadatas=[self._encode_metadata(memory)],
        )
        return memory

    @staticmethod
    def _build_where(
        filters: MemoryFilters | None,
    ) -> dict[str, object] | None:
        if filters is None:
            return None
        if (
            filters.min_importance is not None
            and not 0.0 <= filters.min_importance <= 1.0
        ):
            raise ValueError("Minimum importance must be between 0 and 1.")

        conditions: list[dict[str, object]] = []

        def add_values(field: str, values: Sequence[str]) -> None:
            if len(values) == 1:
                conditions.append({field: values[0]})
            elif values:
                conditions.append({field: {"$in": list(values)}})

        add_values(
            "memory_type",
            [memory_type.value for memory_type in filters.memory_types],
        )
        add_values("role", filters.roles)
        add_values("source", filters.sources)
        add_values("session_id", filters.session_ids)
        if filters.min_importance is not None:
            conditions.append(
                {"importance": {"$gte": filters.min_importance}}
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def memory_search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
    ) -> list[MemoryMatch]:
        if not query.strip():
            raise ValueError("Memory search query cannot be empty.")
        if limit < 1:
            raise ValueError("Memory search limit must be at least 1.")
        if min_similarity is not None and not 0.0 <= min_similarity <= 1.0:
            raise ValueError("Minimum similarity must be between 0 and 1.")

        count = self._collection.count()
        if count == 0:
            return []

        query_options = {
            "query_embeddings": self._embed([query]),
            "n_results": min(limit, count),
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._build_where(filters)
        if where is not None:
            query_options["where"] = where

        result = self._collection.query(**query_options)
        ids = (result["ids"] or [[]])[0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]

        matches: list[MemoryMatch] = []
        for memory_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if document is None or metadata is None:
                continue
            similarity_score = 1.0 - float(distance)
            if (
                min_similarity is not None
                and similarity_score < min_similarity
            ):
                continue
            matches.append(
                MemoryMatch(
                    memory=self._decode_memory(
                        memory_id, document, metadata
                    ),
                    similarity_score=similarity_score,
                )
            )
        return matches

    def memory_get(self, memory_id: str) -> Memory | None:
        result = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        if not documents or not metadatas:
            return None
        document = documents[0]
        metadata = metadatas[0]
        if document is None or metadata is None:
            return None
        return self._decode_memory(result["ids"][0], document, metadata)

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
        existing = self.memory_get(memory_id)
        if existing is None:
            raise KeyError(f"Memory not found: {memory_id}")

        updated = Memory(
            id=existing.id,
            content=content if content is not None else existing.content,
            memory_type=memory_type or existing.memory_type,
            role=role if role is not None else existing.role,
            source=source if source is not None else existing.source,
            session_id=(
                session_id
                if session_id is not None
                else existing.session_id
            ),
            importance=(
                importance if importance is not None else existing.importance
            ),
            created_at=existing.created_at,
            updated_at=datetime.now(UTC),
            metadata=(
                dict(metadata)
                if metadata is not None
                else dict(existing.metadata)
            ),
        )
        self._validate_memory(
            content=updated.content,
            role=updated.role,
            source=updated.source,
            session_id=updated.session_id,
            importance=updated.importance,
            metadata=updated.metadata,
        )

        update_options = {
            "ids": [updated.id],
            "documents": [updated.content],
            "metadatas": [self._encode_metadata(updated)],
        }
        if updated.content != existing.content:
            update_options["embeddings"] = self._embed([updated.content])
        self._collection.update(**update_options)
        return updated

    def memory_delete(self, memory_id: str) -> bool:
        if self.memory_get(memory_id) is None:
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._get_collection()
