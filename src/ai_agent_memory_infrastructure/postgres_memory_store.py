from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ai_agent_memory_infrastructure.memory_context import count_tokens
from ai_agent_memory_infrastructure.memory_store import (
    Memory,
    MemoryFilters,
    MemoryLink,
    MemoryLinkRelation,
    MemoryMatch,
    MemoryStatus,
    MemoryType,
    validate_memory,
)

EMBEDDING_DIMENSIONS = 1536


class PostgresMemoryStore:
    def __init__(
        self,
        *,
        database_url: str,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be at least 1.")
        self._database_url = database_url
        self._dimensions = dimensions
        self._embedding_model = embedding_model
        self._openai = OpenAI(api_key=api_key)
        ensure_database(database_url)
        self._connection = psycopg.connect(
            database_url,
            row_factory=dict_row,
            connect_timeout=5,
        )
        with self._connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._connection.commit()
        register_vector(self._connection)
        self._prepare_schema()

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=list(texts),
        )
        vectors = [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise RuntimeError(
                    "Embedding length does not match the memory table."
                )
        return vectors

    def _prepare_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    importance DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self._dimensions}) NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TIMESTAMPTZ,
                    reinforced_at TIMESTAMPTZ,
                    expired BOOLEAN NOT NULL DEFAULT false,
                    subject TEXT NOT NULL DEFAULT '',
                    predicate TEXT NOT NULL DEFAULT '',
                    object TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS token_count INTEGER
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS reinforced_at TIMESTAMPTZ
                """
            )
            cursor.execute(
                """
                UPDATE memories
                SET reinforced_at = created_at
                WHERE reinforced_at IS NULL
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS expired BOOLEAN NOT NULL DEFAULT false
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS subject TEXT NOT NULL DEFAULT ''
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS predicate TEXT NOT NULL DEFAULT ''
                """
            )
            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS object TEXT NOT NULL DEFAULT ''
                """
            )
            cursor.execute(
                """
                UPDATE memories
                SET expired = true, status = 'active'
                WHERE status = 'expired'
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    from_memory_id TEXT NOT NULL REFERENCES memories(id),
                    relation TEXT NOT NULL,
                    to_memory_id TEXT NOT NULL REFERENCES memories(id),
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_bin (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    importance DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self._dimensions}) NOT NULL,
                    status TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    last_accessed_at TIMESTAMPTZ,
                    reinforced_at TIMESTAMPTZ,
                    expired BOOLEAN NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    predicate TEXT NOT NULL DEFAULT '',
                    object TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cursor.execute(
                """
                ALTER TABLE memory_bin
                DROP COLUMN IF EXISTS removed_at,
                DROP COLUMN IF EXISTS kept_memory_id,
                DROP COLUMN IF EXISTS reason
                """
            )
            cursor.execute(
                """
                ALTER TABLE memory_bin
                ADD COLUMN IF NOT EXISTS subject TEXT NOT NULL DEFAULT ''
                """
            )
            cursor.execute(
                """
                ALTER TABLE memory_bin
                ADD COLUMN IF NOT EXISTS predicate TEXT NOT NULL DEFAULT ''
                """
            )
            cursor.execute(
                """
                ALTER TABLE memory_bin
                ADD COLUMN IF NOT EXISTS object TEXT NOT NULL DEFAULT ''
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS memories_embedding_cosine
                ON memories USING hnsw (embedding vector_cosine_ops)
                """
            )
        self._connection.commit()
        self._backfill_token_counts()

    def write_memory(
        self,
        content: str,
        *,
        memory_type: MemoryType,
        role: str,
        source: str,
        session_id: str,
        importance: float = 0.5,
        metadata: Mapping[str, object] | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        subject: str = "",
        predicate: str = "",
        object: str = "",
    ) -> Memory:
        stored_metadata = dict(metadata or {})
        validate_memory(
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
            status=status,
            token_count=count_tokens(content),
            reinforced_at=now,
            subject=subject,
            predicate=predicate,
            object=object,
        )
        self._insert(memory, self._embed([memory.content])[0])
        return memory

    def import_record(
        self,
        *,
        memory_id: str,
        content: str,
        memory_type: str,
        role: str,
        source: str,
        session_id: str,
        importance: float,
        created_at: datetime,
        updated_at: datetime,
        metadata: Mapping[str, object],
        embedding: Sequence[float],
        status: MemoryStatus = MemoryStatus.ACTIVE,
        token_count: int | None = None,
        last_accessed_at: datetime | None = None,
        reinforced_at: datetime | None = None,
        expired: bool = False,
        subject: str = "",
        predicate: str = "",
        object: str = "",
    ) -> bool:
        vector = [float(value) for value in embedding]
        if len(vector) != self._dimensions:
            raise RuntimeError(
                "Embedding length does not match the memory table."
            )
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memories (
                    id, content, memory_type, role, source, session_id,
                    importance, created_at, updated_at, metadata, embedding,
                    status, token_count, last_accessed_at, reinforced_at,
                    expired, subject, predicate, object
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    memory_id,
                    content,
                    memory_type,
                    role,
                    source,
                    session_id,
                    importance,
                    created_at,
                    updated_at,
                    Jsonb(dict(metadata)),
                    vector,
                    status.value,
                    (
                        count_tokens(content)
                        if token_count is None
                        else token_count
                    ),
                    last_accessed_at,
                    reinforced_at or created_at,
                    expired,
                    subject,
                    predicate,
                    object,
                ),
            )
            inserted = cursor.rowcount == 1
        self._connection.commit()
        return inserted

    def memory_search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        above_min_similarity: bool = False,
    ) -> list[MemoryMatch]:
        if not query.strip():
            raise ValueError("Memory search query cannot be empty.")
        if limit < 1:
            raise ValueError("Memory search limit must be at least 1.")
        if above_min_similarity and min_similarity is None:
            raise ValueError(
                "A strict similarity search needs a minimum similarity."
            )
        if min_similarity is not None and not 0.0 <= min_similarity <= 1.0:
            raise ValueError("Minimum similarity must be between 0 and 1.")
        if self.count() == 0:
            return []

        vector = self._embed([query])[0]
        clauses, params = _filter_clauses(filters)
        if min_similarity is not None:
            comparator = "<" if above_min_similarity else "<="
            clauses.append(
                f"(embedding <=> %s::vector) {comparator} %s"
            )
            params.extend([vector, 1.0 - min_similarity])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, content, memory_type, role, source, session_id,
                       importance, created_at, updated_at, metadata, status,
                       token_count, last_accessed_at, reinforced_at, expired,
                       subject, predicate, object,
                       (embedding <=> %s::vector) AS distance
                FROM memories
                {where}
                ORDER BY distance
                LIMIT %s
                """,
                (vector, *params, limit),
            )
            rows = cursor.fetchall()
        return [
            MemoryMatch(
                memory=_memory_from_row(row),
                similarity_score=1.0 - float(row["distance"]),
            )
            for row in rows
        ]

    def memory_embeddings(
        self,
        memory_ids: Sequence[str],
    ) -> dict[str, tuple[float, ...]]:
        if not memory_ids:
            return {}
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, embedding FROM memories WHERE id = ANY(%s)",
                (list(memory_ids),),
            )
            rows = cursor.fetchall()
        found = {
            row["id"]: _vector_values(row["embedding"]) for row in rows
        }
        missing = [
            memory_id for memory_id in memory_ids if memory_id not in found
        ]
        if missing:
            raise RuntimeError(
                "Stored memory embeddings are missing: " + ", ".join(missing)
            )
        return found

    def memory_get(self, memory_id: str) -> Memory | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, content, memory_type, role, source, session_id,
                       importance, created_at, updated_at, metadata, status,
                       token_count, last_accessed_at, reinforced_at, expired,
                       subject, predicate, object
                FROM memories
                WHERE id = %s
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _memory_from_row(row)

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
                session_id if session_id is not None else existing.session_id
            ),
            importance=(
                importance if importance is not None else existing.importance
            ),
            created_at=existing.created_at,
            updated_at=datetime.now(UTC),
            metadata=(
                dict(metadata) if metadata is not None else dict(existing.metadata)
            ),
            status=existing.status,
            token_count=(
                count_tokens(content)
                if content is not None and content != existing.content
                else existing.token_count
            ),
            last_accessed_at=existing.last_accessed_at,
            reinforced_at=existing.reinforced_at,
            expired=existing.expired,
            subject=existing.subject,
            predicate=existing.predicate,
            object=existing.object,
        )
        validate_memory(
            content=updated.content,
            role=updated.role,
            source=updated.source,
            session_id=updated.session_id,
            importance=updated.importance,
            metadata=updated.metadata,
        )
        embedding = None
        if updated.content != existing.content:
            embedding = self._embed([updated.content])[0]
        with self._connection.cursor() as cursor:
            if embedding is None:
                cursor.execute(
                    """
                    UPDATE memories
                    SET content = %s, memory_type = %s, role = %s,
                        source = %s, session_id = %s, importance = %s,
                        updated_at = %s, metadata = %s, token_count = %s
                    WHERE id = %s
                    """,
                    (
                        updated.content,
                        updated.memory_type.value,
                        updated.role,
                        updated.source,
                        updated.session_id,
                        updated.importance,
                        updated.updated_at,
                        Jsonb(updated.metadata),
                        updated.token_count,
                        updated.id,
                    ),
                )
            else:
                cursor.execute(
                    """
                    UPDATE memories
                    SET content = %s, memory_type = %s, role = %s,
                        source = %s, session_id = %s, importance = %s,
                        updated_at = %s, metadata = %s, token_count = %s,
                        embedding = %s::vector
                    WHERE id = %s
                    """,
                    (
                        updated.content,
                        updated.memory_type.value,
                        updated.role,
                        updated.source,
                        updated.session_id,
                        updated.importance,
                        updated.updated_at,
                        Jsonb(updated.metadata),
                        updated.token_count,
                        embedding,
                        updated.id,
                    ),
                )
        self._connection.commit()
        return updated

    def total_token_count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(SUM(token_count), 0) AS tokens FROM memories"
            )
            row = cursor.fetchone()
        return int(row["tokens"])

    def memory_mark_accessed(
        self,
        memory_ids: Sequence[str],
        *,
        accessed_at: datetime,
    ) -> None:
        if not memory_ids:
            return
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE memories
                SET last_accessed_at = %s
                WHERE id = ANY(%s)
                """,
                (accessed_at, list(memory_ids)),
            )
        self._connection.commit()

    def expire_inactive_memories(
        self,
        *,
        now: datetime,
        inactivity_days: int,
    ) -> int:
        if inactivity_days < 1:
            raise ValueError("Inactivity period must be at least 1 day.")
        cutoff = now - timedelta(days=inactivity_days)
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE memories
                SET expired = true
                WHERE expired = false
                AND GREATEST(
                    created_at,
                    updated_at,
                    COALESCE(last_accessed_at, created_at),
                    COALESCE(reinforced_at, created_at)
                ) < %s
                """,
                (cutoff,),
            )
            expired = cursor.rowcount
        self._connection.commit()
        return expired

    def memory_ids(self, filters: MemoryFilters | None = None) -> list[str]:
        clauses, params = _filter_clauses(filters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection.cursor() as cursor:
            cursor.execute(f"SELECT id FROM memories {where}", params)
            rows = cursor.fetchall()
        return [str(row["id"]) for row in rows]

    def similar_memories(
        self,
        memory_id: str,
        *,
        limit: int,
        min_similarity: float,
        filters: MemoryFilters | None = None,
    ) -> list[MemoryMatch]:
        if limit < 1:
            raise ValueError("Memory search limit must be at least 1.")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("Minimum similarity must be between 0 and 1.")
        clauses, params = _filter_clauses(filters, alias="other")
        clauses.append("other.id <> source.id")
        clauses.append("(other.embedding <=> source.embedding) < %s")
        params.append(1.0 - min_similarity)
        where = " AND ".join(clauses)
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT other.id, other.content, other.memory_type, other.role,
                       other.source, other.session_id, other.importance,
                       other.created_at, other.updated_at, other.metadata,
                       other.status, other.token_count, other.last_accessed_at,
                       other.reinforced_at, other.expired,
                       other.subject, other.predicate, other.object,
                       (other.embedding <=> source.embedding) AS distance
                FROM memories AS source
                JOIN memories AS other ON true
                WHERE source.id = %s AND {where}
                ORDER BY distance
                LIMIT %s
                """,
                (memory_id, *params, limit),
            )
            rows = cursor.fetchall()
        return [
            MemoryMatch(
                memory=_memory_from_row(row),
                similarity_score=1.0 - float(row["distance"]),
            )
            for row in rows
        ]

    def linked_memory_pairs(self) -> set[tuple[str, str]]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT from_memory_id, to_memory_id FROM memory_links"
            )
            rows = cursor.fetchall()
        pairs: set[tuple[str, str]] = set()
        for row in rows:
            left = str(row["from_memory_id"])
            right = str(row["to_memory_id"])
            pairs.add((left, right) if left < right else (right, left))
        return pairs

    def move_memory_to_bin(self, memory_id: str) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memory_bin (
                    id, content, memory_type, role, source, session_id,
                    importance, created_at, updated_at, metadata, embedding,
                    status, token_count, last_accessed_at, reinforced_at,
                    expired, subject, predicate, object
                )
                SELECT id, content, memory_type, role, source, session_id,
                       importance, created_at, updated_at, metadata, embedding,
                       status, token_count, last_accessed_at, reinforced_at,
                       expired, subject, predicate, object
                FROM memories
                WHERE id = %s
                """,
                (memory_id,),
            )
            if cursor.rowcount != 1:
                self._connection.rollback()
                raise KeyError(f"Memory not found: {memory_id}")
            cursor.execute(
                """
                DELETE FROM memory_links
                WHERE from_memory_id = %s OR to_memory_id = %s
                """,
                (memory_id, memory_id),
            )
            cursor.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
        self._connection.commit()

    def bin_count(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM memory_bin")
            row = cursor.fetchone()
        return int(row["count"])

    def bin_content(self, memory_id: str) -> str | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT content FROM memory_bin WHERE id = %s",
                (memory_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return str(row["content"])

    def reinforce_memories(
        self,
        memory_ids: Sequence[str],
        *,
        reinforced_at: datetime,
    ) -> None:
        if not memory_ids:
            return
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE memories
                SET reinforced_at = %s, expired = false
                WHERE id = ANY(%s)
                """,
                (reinforced_at, list(memory_ids)),
            )
        self._connection.commit()

    def memory_delete(self, memory_id: str) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM memory_links
                WHERE from_memory_id = %s OR to_memory_id = %s
                """,
                (memory_id, memory_id),
            )
            cursor.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            deleted = cursor.rowcount == 1
        self._connection.commit()
        return deleted

    def record_memory_link(
        self,
        *,
        from_memory_id: str,
        to_memory_id: str,
        relation: MemoryLinkRelation,
        to_status: MemoryStatus,
    ) -> MemoryLink:
        if from_memory_id == to_memory_id:
            raise ValueError("A memory link needs two different memories.")
        now = datetime.now(UTC)
        link = MemoryLink(
            id=uuid4().hex,
            from_memory_id=from_memory_id,
            relation=relation,
            to_memory_id=to_memory_id,
        )
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE memories
                SET status = %s, updated_at = %s
                WHERE id = %s
                """,
                (to_status.value, now, to_memory_id),
            )
            if cursor.rowcount != 1:
                self._connection.rollback()
                raise KeyError(f"Memory not found: {to_memory_id}")
            cursor.execute(
                """
                INSERT INTO memory_links (
                    id, from_memory_id, relation, to_memory_id, created_at
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    link.id,
                    link.from_memory_id,
                    link.relation.value,
                    link.to_memory_id,
                    now,
                ),
            )
        self._connection.commit()
        return link

    def memory_links_from(self, memory_id: str) -> list[MemoryLink]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, from_memory_id, relation, to_memory_id
                FROM memory_links
                WHERE from_memory_id = %s
                ORDER BY created_at, id
                """,
                (memory_id,),
            )
            rows = cursor.fetchall()
        return [_link_from_row(row) for row in rows]

    def _rollback_if_aborted(self) -> None:
        status = self._connection.info.transaction_status
        if status == psycopg.pq.TransactionStatus.INERROR:
            self._connection.rollback()

    def count(self) -> int:
        self._rollback_if_aborted()
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM memories")
            row = cursor.fetchone()
        return int(row["count"])

    def clear(self) -> None:
        self._rollback_if_aborted()
        with self._connection.cursor() as cursor:
            cursor.execute("DELETE FROM memory_links")
            cursor.execute("DELETE FROM memory_bin")
            cursor.execute("DELETE FROM memories")
        self._connection.commit()

    def _insert(self, memory: Memory, embedding: Sequence[float]) -> None:
        self.import_record(
            memory_id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type.value,
            role=memory.role,
            source=memory.source,
            session_id=memory.session_id,
            importance=memory.importance,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            metadata=memory.metadata,
            embedding=embedding,
            status=memory.status,
            token_count=memory.token_count,
            last_accessed_at=memory.last_accessed_at,
            reinforced_at=memory.reinforced_at or memory.created_at,
            expired=memory.expired,
            subject=memory.subject,
            predicate=memory.predicate,
            object=memory.object,
        )

    def _backfill_token_counts(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, content
                FROM memories
                WHERE token_count IS NULL
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                cursor.execute(
                    "UPDATE memories SET token_count = %s WHERE id = %s",
                    (count_tokens(str(row["content"])), row["id"]),
                )
        self._connection.commit()


def ensure_database(database_url: str) -> None:
    parsed = urlparse(database_url)
    database_name = parsed.path.lstrip("/")
    if not database_name:
        raise ValueError("DATABASE_URL must include a database name.")
    admin_url = urlunparse(parsed._replace(path="/postgres"))
    connection = psycopg.connect(
        admin_url, autocommit=True, connect_timeout=5
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database_name,),
            )
            if cursor.fetchone() is None:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(database_name)
                    )
                )
    finally:
        connection.close()


def _filter_clauses(
    filters: MemoryFilters | None,
    alias: str = "",
) -> tuple[list[str], list[object]]:
    prefix = f"{alias}." if alias else ""
    if filters is None:
        return [], []
    if (
        filters.min_importance is not None
        and not 0.0 <= filters.min_importance <= 1.0
    ):
        raise ValueError("Minimum importance must be between 0 and 1.")

    clauses: list[str] = []
    params: list[object] = []
    memory_type_values: list[str] = []
    for memory_type in filters.memory_types:
        memory_type_values.append(memory_type.value)
        if memory_type is MemoryType.EPISODE:
            memory_type_values.append("conversation")
    if memory_type_values:
        clauses.append(f"{prefix}memory_type = ANY(%s)")
        params.append(memory_type_values)
    if filters.roles:
        clauses.append(f"{prefix}role = ANY(%s)")
        params.append(list(filters.roles))
    if filters.sources:
        clauses.append(f"{prefix}source = ANY(%s)")
        params.append(list(filters.sources))
    if filters.session_ids:
        clauses.append(f"{prefix}session_id = ANY(%s)")
        params.append(list(filters.session_ids))
    if filters.min_importance is not None:
        clauses.append(f"{prefix}importance >= %s")
        params.append(filters.min_importance)
    if filters.statuses:
        clauses.append(f"{prefix}status = ANY(%s)")
        params.append([status.value for status in filters.statuses])
    if filters.expired is not None:
        clauses.append(f"{prefix}expired = %s")
        params.append(filters.expired)
    return clauses, params


def _vector_values(embedding: object) -> tuple[float, ...]:
    if hasattr(embedding, "to_list"):
        values = embedding.to_list()
    elif hasattr(embedding, "tolist"):
        values = embedding.tolist()
    else:
        values = list(embedding)
    return tuple(float(value) for value in values)


def _memory_from_row(row: Mapping[str, object]) -> Memory:
    raw_type = str(row["memory_type"])
    if raw_type == "conversation":
        memory_type = MemoryType.EPISODE
    else:
        try:
            memory_type = MemoryType(raw_type)
        except ValueError:
            memory_type = MemoryType.EPISODE
    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        metadata = {}
    created_at = row["created_at"]
    updated_at = row["updated_at"]
    if not isinstance(created_at, datetime) or not isinstance(updated_at, datetime):
        raise RuntimeError("Stored memory timestamps are invalid.")
    return Memory(
        id=str(row["id"]),
        content=str(row["content"]),
        memory_type=memory_type,
        role=str(row["role"]),
        source=str(row["source"]),
        session_id=str(row["session_id"]),
        importance=float(row["importance"]),
        created_at=created_at,
        updated_at=updated_at,
        metadata=metadata,
        status=_memory_status(row.get("status", MemoryStatus.ACTIVE)),
        token_count=_token_count(row.get("token_count")),
        last_accessed_at=_optional_datetime(row.get("last_accessed_at")),
        reinforced_at=_optional_datetime(row.get("reinforced_at")) or created_at,
        expired=bool(row.get("expired", False)),
        subject=str(row.get("subject") or ""),
        predicate=str(row.get("predicate") or ""),
        object=str(row.get("object") or ""),
    )


def _token_count(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    raise RuntimeError("Stored memory access time is invalid.")


def _memory_status(value: object) -> MemoryStatus:
    try:
        return MemoryStatus(str(value))
    except ValueError as exc:
        raise RuntimeError(f"Stored memory status is invalid: {value}") from exc


def _link_from_row(row: Mapping[str, object]) -> MemoryLink:
    try:
        relation = MemoryLinkRelation(str(row["relation"]))
    except ValueError as exc:
        raise RuntimeError(
            f"Stored memory link is invalid: {row['relation']}"
        ) from exc
    return MemoryLink(
        id=str(row["id"]),
        from_memory_id=str(row["from_memory_id"]),
        relation=relation,
        to_memory_id=str(row["to_memory_id"]),
    )
