from __future__ import annotations

import math
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from ai_agent_memory_infrastructure.memory_api import AgentMemory
from ai_agent_memory_infrastructure.memory_retrieval import (
    CANDIDATE_RETRIEVAL_LIMIT,
    IMPORTANCE_WEIGHT,
    RECENCY_LAMBDA_PER_DAY,
    RECENCY_WEIGHT,
    SEMANTIC_WEIGHT,
    WEIGHTED_CANDIDATE_LIMIT,
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
        self.assertIn("final=", context.text)
        self.assertIn("semantic=", context.text)
        self.assertIn("concise", context.text)
        self.assertNotIn("semantic_score", context.matches[0].memory.metadata)


class RecordingStore:
    def __init__(self, matches: list[MemoryMatch]) -> None:
        self.matches = matches
        self.search_limit = None

    def memory_search(self, query, *, limit=5, min_similarity=None, filters=None):
        self.search_limit = limit
        return self.matches[:limit]


def _read_memory(store: RecordingStore, reranker=None) -> AgentMemory:
    return AgentMemory(
        store,
        FakeMemorySelector(),
        source="test",
        session_id="session",
        reranker=reranker,
    )


def _memory(
    *,
    memory_id: str,
    content: str,
    importance: float,
    created_at: datetime,
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=MemoryType.FACT,
        role="user",
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
        self.assertIn("final=0.800", context.text)

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


if __name__ == "__main__":
    unittest.main()
