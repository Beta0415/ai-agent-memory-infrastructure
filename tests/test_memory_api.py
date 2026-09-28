from __future__ import annotations

import tempfile
import unittest

from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_store import (
    ChromaMemoryStore,
    MemoryFilters,
    MemoryType,
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
        memory = self.store.memory_add(
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

    def test_runtime_helpers_save_and_retrieve_exchange(self) -> None:
        runtime = MemoryRuntime(
            self.store,
            source="test_chat",
            session_id="session-2",
        )
        user_memory, assistant_memory = (
            runtime.runtime_save_exchange_to_memory(
                "Please keep answers concise.",
                "I will keep them concise.",
            )
        )

        self.assertEqual(self.store.count(), 2)
        self.assertEqual(user_memory.role, "user")
        self.assertEqual(assistant_memory.role, "assistant")
        self.assertEqual(
            user_memory.metadata["exchange_id"],
            assistant_memory.metadata["exchange_id"],
        )

        context = runtime.runtime_build_memory_context("concise")
        self.assertTrue(context.matches)
        self.assertIn("similarity=", context.text)
        self.assertIn("concise", context.text)


if __name__ == "__main__":
    unittest.main()
