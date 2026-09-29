from __future__ import annotations

import tempfile
import unittest

from ai_agent_memory_infrastructure.memory_api import AgentMemory
from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_selector import (
    MemoryCandidate,
    MemorySelectionOutput,
)
from ai_agent_memory_infrastructure.memory_store import (
    ChromaMemoryStore,
    MemoryFilters,
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
        self.assertIn("similarity=", context.text)
        self.assertIn("concise", context.text)


if __name__ == "__main__":
    unittest.main()
