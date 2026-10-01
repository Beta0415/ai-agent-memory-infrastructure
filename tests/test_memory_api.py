from __future__ import annotations

import math
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from ai_agent_memory_infrastructure.memory_api import (
    AgentMemory,
    ReadPathSettings,
)
from ai_agent_memory_infrastructure.memory_context import (
    CONTEXT_DUPLICATE_SIMILARITY,
    CONTEXT_TOKEN_BUDGET,
    apply_context_compression,
    cluster_duplicate_memories,
    context_token_count,
    count_tokens,
    embedding_similarity,
    format_context_memories,
    order_context_memories,
    resolve_duplicate_group,
)
from ai_agent_memory_infrastructure.memory_retrieval import (
    CANDIDATE_RETRIEVAL_LIMIT,
    IMPORTANCE_WEIGHT,
    RECENCY_LAMBDA_PER_DAY,
    RECENCY_WEIGHT,
    SEMANTIC_WEIGHT,
    WEIGHTED_CANDIDATE_LIMIT,
    RetrievedMemory,
    calculate_recency_score,
)
from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_selector import (
    MemoryCandidate,
    MemorySelectionOutput,
)
from ai_agent_memory_infrastructure.memory_store import (
    ChromaMemoryStore,
    Memory,
    MemoryFilters,
    MemoryMatch,
    MemoryType,
)


class FakeMemorySelector:
    model_name = "fake-memory-selector"

    def extract_memory(self, conversation):
        return MemorySelectionOutput(
            selected=[
                MemoryCandidate(
                    content="Preference: The user prefers concise answers.",
                    type=MemoryType.PREFERENCE,
                    importance=0.8,
                    confidence=0.95,
                    role="user",
                    debug_source_quote="Please keep answers concise.",
                    debug_reason="This is a durable response preference.",
                )
            ],
            not_selected=[
                MemoryCandidate(
                    content="Conversation: The assistant acknowledged.",
                    type=MemoryType.EPISODE,
                    importance=0.1,
                    confidence=0.9,
                    role="assistant",
                    debug_source_quote="I will keep them concise.",
                    debug_reason=(
                        "This acknowledgement is not useful long-term."
                    ),
                )
            ],
        )


class MemoryAPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = ChromaMemoryStore(
            path=self.directory.name,
            api_key="test-key",
        )
        self.store._embed = self._fake_embed

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def _fake_embed(texts):
        embeddings = []
        for text in texts:
            lowered = text.lower()
            if "python" in lowered:
                embeddings.append([1.0, 0.0, 0.0])
            elif "concise" in lowered:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings

    def test_core_memory_lifecycle(self) -> None:
        memory = self.store.write_memory(
            "The user prefers Python.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-1",
            importance=0.8,
            metadata={"topic": "language"},
        )

        loaded = self.store.memory_get(memory.id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.content, memory.content)
        self.assertEqual(loaded.memory_type, MemoryType.PREFERENCE)
        self.assertEqual(loaded.metadata, {"topic": "language"})

        matches = self.store.memory_search(
            "Python",
            filters=MemoryFilters(
                memory_types=(MemoryType.PREFERENCE,),
                min_importance=0.7,
            ),
        )
        self.assertEqual(matches[0].memory.id, memory.id)
        self.assertAlmostEqual(matches[0].similarity_score, 1.0)

        updated = self.store.memory_update(
            memory.id,
            content="The user strongly prefers Python.",
            importance=0.9,
            metadata={"topic": "language", "confirmed": True},
        )
        self.assertEqual(updated.created_at, memory.created_at)
        self.assertGreaterEqual(updated.updated_at, memory.updated_at)
        self.assertEqual(updated.importance, 0.9)

        self.assertTrue(self.store.memory_delete(memory.id))
        self.assertFalse(self.store.memory_delete(memory.id))

    def test_legacy_conversation_memories_map_to_episodes(self) -> None:
        self.store._collection.add(
            ids=["legacy-memory"],
            embeddings=self._fake_embed(["A past event occurred."]),
            documents=["A past event occurred."],
            metadatas=[
                {
                    "memory_type": "conversation",
                    "role": "user",
                    "source": "legacy",
                    "session_id": "legacy-session",
                    "importance": 0.5,
                }
            ],
        )

        loaded = self.store.memory_get("legacy-memory")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.memory_type, MemoryType.EPISODE)

        matches = self.store.memory_search(
            "A past event occurred.",
            filters=MemoryFilters(memory_types=(MemoryType.EPISODE,)),
        )
        self.assertEqual(matches[0].memory.id, "legacy-memory")

    def test_runtime_selects_and_saves_approved_candidates(self) -> None:
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="memory_selector_v1",
            session_id="session-2",
        )
        runtime = MemoryRuntime(memory_api)
        result = runtime.runtime_write_memory(
            [
                {
                    "role": "user",
                    "content": "Please keep answers concise.",
                },
                {
                    "role": "assistant",
                    "content": "I will keep them concise.",
                },
            ]
        )

        self.assertEqual(len(result.selection.selected), 1)
        self.assertEqual(len(result.selection.not_selected), 1)
        self.assertEqual(len(result.stored_memories), 1)
        self.assertEqual(self.store.count(), 1)
        stored_memory = result.stored_memories[0]
        self.assertEqual(stored_memory.memory_type, MemoryType.PREFERENCE)
        self.assertEqual(stored_memory.role, "user")
        self.assertEqual(stored_memory.metadata["confidence"], 0.95)
        self.assertEqual(
            stored_memory.metadata["debug_source_quote"],
            "Please keep answers concise.",
        )
        self.assertNotIn("debug_reason", stored_memory.metadata)
        self.assertEqual(
            stored_memory.metadata["selector_model"],
            "fake-memory-selector",
        )

        context = runtime.runtime_build_memory_context("concise")
        self.assertTrue(context.matches)
        self.assertIn("preference\n- [user; relevance=", context.text)
        self.assertIn("concise", context.text)
        self.assertNotIn("semantic=", context.text)
        self.assertNotIn("semantic_score", context.matches[0].memory.metadata)

    def test_memory_context_keeps_best_equivalent_embedding(
        self,
    ) -> None:
        classifier = ScriptedRelationClassifier("equivalent")
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-3",
            relation_classifier=classifier,
        )
        first = self.store.write_memory(
            "The user prefers Python.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-3",
            importance=0.6,
        )
        second = self.store.write_memory(
            "The user likes Python a lot.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-3",
            importance=0.6,
        )
        self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-3",
            importance=0.6,
        )

        result = memory_api.memory_context("Python")

        self.assertEqual(self.store.count(), 3)
        self.assertEqual(len(result.matches), 3)
        self.assertEqual(len(classifier.groups), 1)
        self.assertEqual(
            set(classifier.groups[0]),
            {first.id, second.id},
        )
        self.assertEqual(len(result.memories), 2)
        kept = next(
            item
            for item in result.memories
            if "Python" in item.memory.content
        )
        self.assertIn(
            kept.memory.content,
            {
                "The user prefers Python.",
                "The user likes Python a lot.",
            },
        )
        self.assertEqual(kept.merged_ids, ())
        concise = next(
            item
            for item in result.memories
            if "concise" in item.memory.content
        )
        self.assertEqual(concise.memory.content, "The user prefers concise answers.")
        self.assertEqual(
            self.store.memory_get(first.id).content,
            "The user prefers Python.",
        )


class RecordingStore:
    def __init__(
        self,
        matches: list[MemoryMatch],
        embeddings: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        self.matches = matches
        self.embeddings = embeddings
        self.search_limit = None
        self.embedded_ids: list[str] = []

    def memory_search(self, query, *, limit=5, min_similarity=None, filters=None):
        self.search_limit = limit
        return self.matches[:limit]

    def memory_embeddings(self, memory_ids):
        self.embedded_ids = list(memory_ids)
        if self.embeddings is not None:
            return {
                memory_id: self.embeddings[memory_id]
                for memory_id in memory_ids
            }
        vectors = {}
        for index, memory_id in enumerate(memory_ids):
            vector = [0.0] * len(memory_ids)
            vector[index] = 1.0
            vectors[memory_id] = tuple(vector)
        return vectors


def _read_memory(
    store: RecordingStore,
    reranker=None,
    compressor=None,
    relation_classifier=None,
) -> AgentMemory:
    return AgentMemory(
        store,
        FakeMemorySelector(),
        source="test",
        session_id="session",
        reranker=reranker,
        compressor=compressor,
        relation_classifier=relation_classifier,
    )


def _memory(
    *,
    memory_id: str,
    content: str,
    importance: float,
    created_at: datetime,
    memory_type: MemoryType = MemoryType.FACT,
    role: str = "user",
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        role=role,
        source="test",
        session_id="session",
        importance=importance,
        created_at=created_at,
        updated_at=created_at,
        metadata={},
    )


class ReadPathRankingTest(unittest.TestCase):
    def test_recency_decays_exponentially_from_created_at(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        created_at = now - timedelta(days=30)
        score = calculate_recency_score(created_at, now=now)
        self.assertAlmostEqual(score, math.exp(-RECENCY_LAMBDA_PER_DAY * 30))
        self.assertAlmostEqual(score, 0.5)

    def test_runtime_retrieves_30_then_returns_top_4_by_final_score(
        self,
    ) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id="old-similar",
                    content="An old but highly similar fact.",
                    importance=0.1,
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.95,
            ),
            MemoryMatch(
                memory=_memory(
                    memory_id="recent-important",
                    content="A recent important fact.",
                    importance=0.9,
                    created_at=now,
                ),
                similarity_score=0.70,
            ),
        ]
        for index in range(4):
            matches.append(
                MemoryMatch(
                    memory=_memory(
                        memory_id=f"filler-{index}",
                        content=f"Filler fact {index}.",
                        importance=0.2,
                        created_at=now - timedelta(days=10),
                    ),
                    similarity_score=0.40,
                )
            )
        store = RecordingStore(matches)
        runtime = MemoryRuntime(_read_memory(store))

        context = runtime.runtime_build_memory_context(
            "fact",
            limit=4,
            now=now,
        )

        self.assertEqual(store.search_limit, CANDIDATE_RETRIEVAL_LIMIT)
        self.assertFalse(context.deduplicated)
        self.assertEqual(
            [match.memory.id for match in context.matches],
            ["recent-important", "old-similar", "filler-0", "filler-1"],
        )

        selected = context.matches[0]
        expected_recency = 1.0
        expected_final = (
            SEMANTIC_WEIGHT * 0.70
            + RECENCY_WEIGHT * expected_recency
            + IMPORTANCE_WEIGHT * 0.9
        )
        self.assertAlmostEqual(selected.semantic_score, 0.70)
        self.assertAlmostEqual(selected.recency_score, expected_recency)
        self.assertAlmostEqual(selected.retrieval_score, expected_final)
        self.assertAlmostEqual(selected.final_score, expected_final)
        self.assertNotIn("final_score", selected.memory.metadata)
        self.assertIn(
            "fact\n- [user; relevance=0.800; ] A recent important fact.",
            context.text,
        )
        self.assertIn("A recent important fact.", context.text)

    def test_reranker_selects_top_4_from_weighted_top_10(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id=f"rank-{index:02d}",
                    content=f"Candidate {index}.",
                    importance=0.5,
                    created_at=now,
                ),
                similarity_score=1.0 - (index * 0.04),
            )
            for index in range(12)
        ]
        rerank_scores = {
            f"rank-{index:02d}": 0.1 for index in range(10)
        }
        rerank_scores.update(
            {
                "rank-09": 0.99,
                "rank-08": 0.90,
                "rank-07": 0.80,
                "rank-06": 0.70,
            }
        )
        store = RecordingStore(matches)
        reranker = ScriptedReranker(rerank_scores)
        runtime = MemoryRuntime(_read_memory(store, reranker))

        context = runtime.runtime_build_memory_context(
            "candidate",
            limit=4,
            now=now,
        )

        self.assertEqual(
            reranker.seen_ids,
            [f"rank-{index:02d}" for index in range(WEIGHTED_CANDIDATE_LIMIT)],
        )
        self.assertEqual(
            [match.memory.id for match in context.matches],
            ["rank-09", "rank-08", "rank-07", "rank-06"],
        )
        selected = context.matches[0]
        self.assertAlmostEqual(selected.final_score, 0.99)
        self.assertGreater(selected.retrieval_score, 0.0)
        self.assertNotEqual(selected.final_score, selected.retrieval_score)
        self.assertNotIn("rank-11", reranker.seen_ids)
        self.assertEqual(len(context.candidates), 12)
        self.assertEqual(
            [candidate.memory.id for candidate in context.candidates],
            [f"rank-{index:02d}" for index in range(12)],
        )
        self.assertTrue(
            all(
                candidate.rerank_score is not None
                for candidate in context.candidates[:WEIGHTED_CANDIDATE_LIMIT]
            )
        )
        self.assertTrue(
            all(
                candidate.rerank_score is None
                for candidate in context.candidates[WEIGHTED_CANDIDATE_LIMIT:]
            )
        )


def _retrieved(memory: Memory, *, final_score: float) -> RetrievedMemory:
    return RetrievedMemory(
        memory=memory,
        semantic_score=final_score,
        recency_score=1.0,
        retrieval_score=final_score,
        final_score=final_score,
    )


class ContextDedupTest(unittest.TestCase):
    def test_similarity_above_threshold_merges_and_equal_does_not(
        self,
    ) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        left = _retrieved(
            _memory(
                memory_id="left",
                content="Alpha fact.",
                importance=0.5,
                created_at=now,
            ),
            final_score=0.9,
        )
        right = _retrieved(
            _memory(
                memory_id="right",
                content="Beta fact.",
                importance=0.5,
                created_at=now,
            ),
            final_score=0.8,
        )
        above = (
            0.91,
            math.sqrt(1 - 0.91**2),
        )
        at_threshold = (CONTEXT_DUPLICATE_SIMILARITY, math.sqrt(0.19))

        self.assertGreater(
            embedding_similarity((1.0, 0.0), above),
            CONTEXT_DUPLICATE_SIMILARITY,
        )
        self.assertAlmostEqual(
            embedding_similarity((1.0, 0.0), at_threshold),
            CONTEXT_DUPLICATE_SIMILARITY,
        )

        clustered = cluster_duplicate_memories(
            [left, right],
            {"left": (1.0, 0.0), "right": above},
        )
        self.assertEqual(len(clustered), 1)
        self.assertEqual(
            [item.memory.id for item in clustered[0]],
            ["left", "right"],
        )
        kept = resolve_duplicate_group(clustered[0], "equivalent")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].memory.id, "left")
        self.assertEqual(kept[0].final_score, 0.9)
        self.assertEqual(kept[0].memory.content, "Alpha fact.")

        both = resolve_duplicate_group(clustered[0], "contradictory")
        self.assertEqual(
            [item.memory.content for item in both],
            ["Alpha fact.", "Beta fact."],
        )

        separate = cluster_duplicate_memories(
            [left, right],
            {"left": (1.0, 0.0), "right": at_threshold},
        )
        self.assertEqual(len(separate), 2)

    def test_identical_content_is_kept_once(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        memories = [
            _retrieved(
                _memory(
                    memory_id=memory_id,
                    content="User likes tea.",
                    importance=0.5,
                    created_at=now,
                ),
                final_score=0.9 - index / 10,
            )
            for index, memory_id in enumerate(("one", "two"))
        ]
        clustered = cluster_duplicate_memories(
            memories,
            {"one": (1.0, 0.0), "two": (1.0, 0.0)},
        )
        self.assertEqual(len(clustered), 1)
        kept = resolve_duplicate_group(clustered[0], "equivalent")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].memory.id, "one")
        self.assertEqual(kept[0].memory.content, "User likes tea.")

    def test_linked_duplicates_merge_as_one_group(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        angle = math.acos(0.95)
        memories = [
            _retrieved(
                _memory(
                    memory_id=memory_id,
                    content=f"Fact {memory_id}.",
                    importance=0.5,
                    created_at=now,
                ),
                final_score=0.9 - index / 10,
            )
            for index, memory_id in enumerate(("a", "b", "c"))
        ]
        embeddings = {
            "a": (1.0, 0.0),
            "b": (math.cos(angle), math.sin(angle)),
            "c": (math.cos(angle * 2), math.sin(angle * 2)),
        }
        self.assertLessEqual(
            embedding_similarity(embeddings["a"], embeddings["c"]),
            CONTEXT_DUPLICATE_SIMILARITY,
        )
        clustered = cluster_duplicate_memories(memories, embeddings)
        self.assertEqual(len(clustered), 1)
        self.assertEqual(
            [item.memory.id for item in clustered[0]],
            ["a", "b", "c"],
        )
        kept = resolve_duplicate_group(clustered[0], "equivalent")
        self.assertEqual([item.memory.id for item in kept], ["a"])
        self.assertEqual(kept[0].memory.content, "Fact a.")
        both = resolve_duplicate_group(clustered[0], "contradictory")
        self.assertEqual(
            [item.memory.content for item in both],
            ["Fact a.", "Fact b.", "Fact c."],
        )

    def test_memory_context_dedupes_only_search_matches(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id=f"rank-{index:02d}",
                    content=f"Candidate {index}.",
                    importance=0.5,
                    created_at=now,
                ),
                similarity_score=1.0 - (index * 0.05),
            )
            for index in range(5)
        ]
        near = (0.96, math.sqrt(1 - 0.96**2), 0.0)
        embeddings = {
            "rank-00": (1.0, 0.0, 0.0),
            "rank-01": (0.0, 1.0, 0.0),
            "rank-02": (0.0, 0.0, 1.0),
            "rank-03": near,
            "rank-04": (1.0, 0.0, 0.0),
        }
        store = RecordingStore(matches, embeddings)
        classifier = ScriptedRelationClassifier("equivalent")
        memory_api = _read_memory(
            store,
            relation_classifier=classifier,
        )

        result = memory_api.memory_context("candidate", limit=4, now=now)

        self.assertEqual(
            store.embedded_ids,
            [f"rank-{index:02d}" for index in range(4)],
        )
        self.assertEqual(classifier.groups, [["rank-00", "rank-03"]])
        self.assertEqual(len(result.matches), 4)
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(
            [item.memory.id for item in result.memories],
            ["rank-00", "rank-01", "rank-02"],
        )
        self.assertTrue(result.deduplicated)
        self.assertEqual(
            [item.memory.id for item in result.dropped_memories],
            ["rank-03"],
        )
        self.assertEqual(result.memories[0].memory.content, "Candidate 0.")
        self.assertEqual(len(result.duplicate_clusters), 1)
        cluster = result.duplicate_clusters[0]
        self.assertEqual(cluster.relation, "equivalent")
        self.assertEqual(
            [item.memory.id for item in cluster.memories],
            ["rank-00", "rank-03"],
        )
        self.assertIsNotNone(cluster.selected)
        self.assertEqual(cluster.selected.memory.id, "rank-00")
        self.assertEqual(cluster.selected.memory.content, "Candidate 0.")

    def test_contradictory_cluster_keeps_every_memory(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id=memory_id,
                    content=content,
                    importance=0.5,
                    created_at=now,
                ),
                similarity_score=0.9,
            )
            for memory_id, content in (
                ("name", "His name is Beta."),
                ("negation", "His name is not Beta."),
            )
        ]
        embeddings = {
            "name": (1.0, 0.0),
            "negation": (1.0, 0.0),
        }
        classifier = ScriptedRelationClassifier("contradictory")
        memory_api = _read_memory(
            RecordingStore(matches, embeddings),
            relation_classifier=classifier,
        )

        result = memory_api.memory_context("name", now=now)

        self.assertFalse(result.deduplicated)
        self.assertEqual(len(result.memories), 2)
        self.assertEqual(len(result.duplicate_clusters), 1)
        cluster = result.duplicate_clusters[0]
        self.assertEqual(cluster.relation, "contradictory")
        self.assertEqual(
            [item.memory.content for item in cluster.memories],
            ["His name is Beta.", "His name is not Beta."],
        )
        self.assertIsNone(cluster.selected)


class ContextOrderingTest(unittest.TestCase):
    def test_groups_by_type_then_relevance(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        memories = [
            _retrieved(
                _memory(
                    memory_id="pref-high",
                    content="User wants short answers.",
                    importance=0.5,
                    created_at=now,
                    memory_type=MemoryType.PREFERENCE,
                ),
                final_score=0.910,
            ),
            _retrieved(
                _memory(
                    memory_id="fact-low",
                    content="Beta lives in New York.",
                    importance=0.5,
                    created_at=now,
                    memory_type=MemoryType.FACT,
                    role="assistant",
                ),
                final_score=0.400,
            ),
            _retrieved(
                _memory(
                    memory_id="fact-high",
                    content="The user's name is Beta.",
                    importance=0.5,
                    created_at=now,
                    memory_type=MemoryType.FACT,
                ),
                final_score=0.860,
            ),
        ]

        ordered = order_context_memories(memories)
        self.assertEqual(
            [item.memory.id for item in ordered],
            ["fact-high", "fact-low", "pref-high"],
        )
        self.assertEqual(
            format_context_memories(memories),
            "\n".join(
                [
                    "fact",
                    "- [user; relevance=0.860; ] The user's name is Beta.",
                    "- [assistant; relevance=0.400; ] Beta lives in New York.",
                    "preference",
                    "- [user; relevance=0.910; ] User wants short answers.",
                ]
            ),
        )


class ScriptedRelationClassifier:
    def __init__(self, relation: str) -> None:
        self.relation = relation
        self.groups: list[list[str]] = []

    def classify(self, memories):
        self.groups.append([memory.memory.id for memory in memories])
        return self.relation


class ScriptedCompressor:
    def __init__(self, contents: dict[str, str]) -> None:
        self.contents = contents
        self.seen_ids: list[str] = []
        self.token_budget = None

    def compress(self, memories, *, token_budget):
        self.seen_ids = [memory.memory.id for memory in memories]
        self.token_budget = token_budget
        return {
            memory.memory.id: self.contents[memory.memory.id]
            for memory in memories
        }


class ContextCompressionTest(unittest.TestCase):
    def test_short_context_skips_compression(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id="short",
                    content="User likes tea.",
                    importance=0.5,
                    created_at=now,
                ),
                similarity_score=0.9,
            )
        ]
        compressor = ScriptedCompressor({"short": "Likes tea."})
        memory_api = _read_memory(RecordingStore(matches), compressor=compressor)

        result = memory_api.memory_context("tea", now=now)

        self.assertLessEqual(
            context_token_count(result.memories),
            CONTEXT_TOKEN_BUDGET,
        )
        self.assertEqual(compressor.seen_ids, [])
        self.assertEqual(result.memories[0].memory.content, "User likes tea.")
        self.assertFalse(result.memories[0].compressed)

    def test_over_budget_shortens_after_dedup(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        long_content = (
            "The user has stated a durable preference for receiving "
            "detailed written explanations about Python packaging."
        )
        duplicate_content = (
            "The user wants detailed written explanations about "
            "Python packaging."
        )
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id="primary",
                    content=long_content,
                    importance=0.8,
                    created_at=now,
                ),
                similarity_score=0.95,
            ),
            MemoryMatch(
                memory=_memory(
                    memory_id="duplicate",
                    content=duplicate_content,
                    importance=0.7,
                    created_at=now,
                ),
                similarity_score=0.90,
            ),
            MemoryMatch(
                memory=_memory(
                    memory_id="outside",
                    content="This candidate is outside the returned matches.",
                    importance=0.1,
                    created_at=now - timedelta(days=10),
                ),
                similarity_score=0.10,
            ),
        ]
        embeddings = {
            "primary": (1.0, 0.0),
            "duplicate": (1.0, 0.0),
            "outside": (0.0, 1.0),
        }
        shortened = "User wants detailed Python packaging explanations."
        compressor = ScriptedCompressor({"primary": shortened})
        classifier = ScriptedRelationClassifier("equivalent")
        store = RecordingStore(matches, embeddings)
        memory_api = _read_memory(
            store,
            compressor=compressor,
            relation_classifier=classifier,
        )
        token_budget = count_tokens(long_content) - 1

        result = memory_api.memory_context(
            "python packaging",
            limit=2,
            now=now,
            token_budget=token_budget,
        )

        self.assertEqual(classifier.groups, [["primary", "duplicate"]])
        self.assertEqual(compressor.seen_ids, ["primary"])
        self.assertEqual(compressor.token_budget, token_budget)
        self.assertEqual(len(result.memories), 1)
        self.assertEqual(result.memories[0].memory.content, shortened)
        self.assertTrue(result.memories[0].compressed)
        self.assertEqual(result.memories[0].memory.id, "primary")
        self.assertEqual(len(result.duplicate_clusters), 1)
        cluster = result.duplicate_clusters[0]
        self.assertEqual(cluster.relation, "equivalent")
        self.assertEqual(cluster.selected.memory.content, long_content)

    def test_longer_compression_is_discarded(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        original = "User likes tea."
        memory = _retrieved(
            _memory(
                memory_id="tea",
                content=original,
                importance=0.5,
                created_at=now,
            ),
            final_score=0.8,
        )
        expanded = apply_context_compression(
            [memory],
            {
                "tea": (
                    "The user has a lasting personal preference for drinking tea "
                    "every afternoon."
                )
            },
        )
        self.assertEqual(expanded[0].memory.content, original)
        self.assertFalse(expanded[0].compressed)

        shortened = apply_context_compression(
            [memory],
            {"tea": "Likes tea."},
        )
        self.assertEqual(shortened[0].memory.content, "Likes tea.")
        self.assertTrue(shortened[0].compressed)
        self.assertLess(
            count_tokens("Likes tea."),
            count_tokens(original),
        )


class ScriptedReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen_ids: list[str] = []

    def rerank(self, query, candidates):
        self.seen_ids = [candidate.memory.id for candidate in candidates]
        return {
            candidate.memory.id: self.scores[candidate.memory.id]
            for candidate in candidates
        }


class ReadPathSettingsTest(unittest.TestCase):
    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ReadPathSettings(
                semantic_weight=0.5,
                recency_weight=0.5,
                importance_weight=0.5,
            )

    def test_rejects_cuts_out_of_order(self) -> None:
        with self.assertRaises(ValueError):
            ReadPathSettings(
                candidate_limit=5,
                weighted_limit=8,
                result_limit=4,
            )

    def test_custom_settings_change_scoring_and_cuts(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        created_at = now - timedelta(days=10)
        matches = [
            MemoryMatch(
                memory=_memory(
                    memory_id="older",
                    content="An older fact.",
                    importance=0.9,
                    created_at=created_at,
                ),
                similarity_score=0.2,
            ),
            MemoryMatch(
                memory=_memory(
                    memory_id="closer",
                    content="A closer fact.",
                    importance=0.1,
                    created_at=now,
                ),
                similarity_score=0.9,
            ),
        ]
        store = RecordingStore(matches)
        settings = ReadPathSettings(
            semantic_weight=0.0,
            recency_weight=1.0,
            importance_weight=0.0,
            recency_half_life_days=10,
            candidate_limit=2,
            weighted_limit=2,
            result_limit=1,
        )

        result = _read_memory(store).memory_search(
            "fact",
            now=now,
            settings=settings,
        )

        self.assertEqual(store.search_limit, 2)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].memory.id, "closer")
        self.assertAlmostEqual(result.matches[0].recency_score, 1.0)
        older = next(
            item for item in result.candidates if item.memory.id == "older"
        )
        self.assertAlmostEqual(older.recency_score, 0.5)
        self.assertAlmostEqual(older.retrieval_score, 0.5)


if __name__ == "__main__":
    unittest.main()
