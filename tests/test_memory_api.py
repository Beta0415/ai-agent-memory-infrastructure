from __future__ import annotations

import math
import unittest
from datetime import UTC, datetime, timedelta

from ai_agent_memory_infrastructure.memory_maintenance import (
    MaintenanceSettings,
)
from ai_agent_memory_infrastructure.memory_api import (
    AgentMemory,
    ReadPathSettings,
    WritePathSettings,
)
from ai_agent_memory_infrastructure.memory_atomic_score import (
    AtomicScoreWeights,
    component_score,
    needs_atomic_review,
    score_atomic_memory,
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
    score_retrieved_memory,
)
from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_selector import (
    MemoryCandidate,
    MemorySelectionOutput,
)
from ai_agent_memory_infrastructure.memory_store import (
    Memory,
    MemoryFilters,
    MemoryLinkRelation,
    MemoryMatch,
    MemoryStatus,
    MemoryType,
)
from ai_agent_memory_infrastructure.postgres_memory_store import (
    PostgresMemoryStore,
)


class FakeMemorySelector:
    model_name = "fake-memory-selector"

    def extract_memory(self, conversation):
        return MemorySelectionOutput(
            selected=[
                MemoryCandidate(
                    content="Preference: The user prefers concise answers.",
                    subject="The user",
                    predicate="prefers",
                    object="concise answers",
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
                    subject="The assistant",
                    predicate="acknowledged",
                    object="the user's request",
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
        self.store = PostgresMemoryStore(
            database_url="postgresql://localhost:5432/agent_memory_test",
            api_key="test-key",
            dimensions=3,
        )
        self.store._embed = self._fake_embed
        self.store.clear()

    def tearDown(self) -> None:
        self.store.clear()
        self.store._connection.close()

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
        self.assertEqual(loaded.status, MemoryStatus.ACTIVE)
        self.assertEqual(loaded.token_count, count_tokens(loaded.content))
        self.assertIsNone(loaded.last_accessed_at)

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
        self.assertEqual(updated.status, MemoryStatus.ACTIVE)

        self.assertTrue(self.store.memory_delete(memory.id))
        self.assertFalse(self.store.memory_delete(memory.id))

    def test_new_memory_reinforcement_matches_created_at(self) -> None:
        memory = self.store.write_memory(
            "The user prefers Python.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-1",
            importance=0.5,
        )
        self.assertEqual(memory.reinforced_at, memory.created_at)

    def test_equivalent_duplicate_refreshes_reinforced_at(self) -> None:
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        self.store.import_record(
            memory_id="kept",
            content="The user prefers concise answers.",
            memory_type="preference",
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
            created_at=created_at,
            updated_at=created_at,
            metadata={},
            embedding=[0.0, 1.0, 0.0],
            reinforced_at=created_at,
        )

        result = self._write_selected_candidate(
            ScriptedCandidateRelationClassifier("equivalent")
        )

        kept = self.store.memory_get("kept")
        assert kept is not None
        self.assertIsNone(result.decisions[0].stored)
        self.assertEqual(kept.created_at, created_at)
        self.assertGreater(kept.reinforced_at, created_at)
        self.assertEqual(kept.status, MemoryStatus.ACTIVE)

    def test_equivalent_duplicate_revives_an_expired_memory(self) -> None:
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        self.store.import_record(
            memory_id="expired-kept",
            content="The user prefers concise answers.",
            memory_type="preference",
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
            created_at=created_at,
            updated_at=created_at,
            metadata={},
            embedding=[0.0, 1.0, 0.0],
            reinforced_at=created_at,
            expired=True,
        )

        result = self._write_selected_candidate(
            ScriptedCandidateRelationClassifier("equivalent")
        )

        kept = self.store.memory_get("expired-kept")
        assert kept is not None
        self.assertEqual(
            [match.memory.id for match in result.decisions[0].related],
            ["expired-kept"],
        )
        self.assertIsNone(result.decisions[0].stored)
        self.assertFalse(kept.expired)
        self.assertGreater(kept.reinforced_at, created_at)
        self.assertEqual(kept.status, MemoryStatus.ACTIVE)

    def test_maintenance_dedup_moves_equivalent_copies_to_the_bin(self) -> None:
        moment = datetime(2026, 1, 1, tzinfo=UTC)
        later = datetime(2026, 6, 1, tzinfo=UTC)
        self._import_dated(
            "older-copy",
            moment,
            content="The user prefers concise answers.",
            embedding=[0.0, 1.0, 0.0],
        )
        self._import_dated(
            "newer-copy",
            later,
            content="The user likes concise replies.",
            embedding=[0.0, 1.0, 0.0],
            reinforced_at=later,
        )
        self._import_dated(
            "different",
            moment,
            content="The user prefers Python.",
            embedding=[1.0, 0.0, 0.0],
        )
        classifier = ScriptedDedupClassifier("equivalent")
        now = datetime(2026, 8, 1, tzinfo=UTC)

        result = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            dedup_classifier=classifier,
        ).memory_maintain(now=now)

        self.assertEqual(result.removed_count, 1)
        self.assertIsNone(self.store.memory_get("older-copy"))
        kept = self.store.memory_get("newer-copy")
        assert kept is not None
        self.assertEqual(kept.reinforced_at, now)
        self.assertFalse(kept.expired)
        self.assertEqual(self.store.bin_count(), 1)
        self.assertEqual(
            self.store.bin_content("older-copy"),
            "The user prefers concise answers.",
        )
        self.assertIsNotNone(self.store.memory_get("different"))
        self.assertEqual(classifier.calls[0][0], "newer-copy")

    def test_maintenance_dedup_keeps_the_higher_importance_recency_score(self) -> None:
        now = datetime(2026, 8, 1, tzinfo=UTC)
        self._import_dated(
            "important-old",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user prefers concise answers.",
            embedding=[0.0, 1.0, 0.0],
            reinforced_at=datetime(2026, 1, 1, tzinfo=UTC),
            importance=1.0,
        )
        self._import_dated(
            "minor-recent",
            datetime(2026, 7, 15, tzinfo=UTC),
            content="The user likes concise replies.",
            embedding=[0.0, 1.0, 0.0],
            reinforced_at=datetime(2026, 7, 15, tzinfo=UTC),
            importance=0.1,
        )

        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            dedup_classifier=ScriptedDedupClassifier("equivalent"),
        ).memory_maintain(now=now)

        self.assertIsNotNone(self.store.memory_get("important-old"))
        self.assertIsNone(self.store.memory_get("minor-recent"))
        self.assertEqual(
            self.store.bin_content("minor-recent"),
            "The user likes concise replies.",
        )

    def test_maintenance_dedup_keeps_a_non_equivalent_group(self) -> None:
        moment = datetime(2026, 6, 1, tzinfo=UTC)
        self._import_dated(
            "left",
            moment,
            content="The user prefers concise answers.",
            embedding=[0.0, 1.0, 0.0],
        )
        self._import_dated(
            "right",
            moment,
            content="The user likes concise replies.",
            embedding=[0.0, 1.0, 0.0],
        )

        result = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            dedup_classifier=ScriptedDedupClassifier("not_equivalent"),
        ).memory_maintain(now=datetime(2026, 8, 1, tzinfo=UTC))

        self.assertEqual(result.removed_count, 0)
        self.assertIsNotNone(self.store.memory_get("left"))
        self.assertIsNotNone(self.store.memory_get("right"))
        self.assertEqual(self.store.bin_count(), 0)

    def test_maintenance_dedup_skips_a_linked_pair(self) -> None:
        moment = datetime(2026, 6, 1, tzinfo=UTC)
        self._import_dated(
            "source",
            moment,
            content="The user likes Java now.",
            embedding=[0.0, 1.0, 0.0],
        )
        self._import_dated(
            "target",
            moment,
            content="The user likes Python.",
            embedding=[0.0, 1.0, 0.0],
        )
        self.store.record_memory_link(
            from_memory_id="source",
            to_memory_id="target",
            relation=MemoryLinkRelation.SUPERSEDES,
            to_status=MemoryStatus.ACTIVE,
        )
        classifier = ScriptedDedupClassifier("equivalent")

        result = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            dedup_classifier=classifier,
        ).memory_maintain(now=datetime(2026, 8, 1, tzinfo=UTC))

        self.assertEqual(result.removed_count, 0)
        self.assertEqual(classifier.calls, [])
        self.assertIsNotNone(self.store.memory_get("source"))

    def test_retrieval_records_last_access(self) -> None:
        memory = self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )
        accessed_at = datetime(2026, 10, 3, tzinfo=UTC)
        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
        ).memory_search("concise", now=accessed_at)

        loaded = self.store.memory_get(memory.id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.last_accessed_at, accessed_at)

    def test_forget_expires_only_inactive_active_memories(self) -> None:
        old = datetime(2026, 1, 1, tzinfo=UTC)
        recent = datetime(2026, 7, 15, tzinfo=UTC)
        accessed = datetime(2026, 7, 20, tzinfo=UTC)
        now = datetime(2026, 8, 1, tzinfo=UTC)
        self._import_dated("old-active", old, content="An old active memory.")
        self._import_dated("recent-active", recent, content="A recent memory.")
        self._import_dated(
            "old-accessed",
            old,
            content="An old memory that was retrieved.",
            last_accessed_at=accessed,
        )
        self._import_dated(
            "old-superseded",
            old,
            content="An old superseded memory.",
            status=MemoryStatus.SUPERSEDED,
        )
        self._import_dated(
            "old-reinforced",
            old,
            content="An old memory that was restated.",
            reinforced_at=datetime(2026, 7, 20, tzinfo=UTC),
        )

        expired = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
        ).memory_maintain(now=now)

        self.assertEqual(expired.expired_count, 2)
        old_active = self.store.memory_get("old-active")
        recent_active = self.store.memory_get("recent-active")
        old_accessed = self.store.memory_get("old-accessed")
        old_superseded = self.store.memory_get("old-superseded")
        assert old_active is not None
        assert recent_active is not None
        assert old_accessed is not None
        assert old_superseded is not None
        self.assertEqual(old_active.status, MemoryStatus.ACTIVE)
        self.assertTrue(old_active.expired)
        self.assertEqual(recent_active.status, MemoryStatus.ACTIVE)
        self.assertFalse(recent_active.expired)
        self.assertEqual(old_accessed.status, MemoryStatus.ACTIVE)
        self.assertFalse(old_accessed.expired)
        self.assertEqual(old_superseded.status, MemoryStatus.SUPERSEDED)
        self.assertTrue(old_superseded.expired)
        old_reinforced = self.store.memory_get("old-reinforced")
        assert old_reinforced is not None
        self.assertFalse(old_reinforced.expired)

    def test_over_capacity_runs_forget_after_a_write(self) -> None:
        old = datetime(2026, 1, 1, tzinfo=UTC)
        self._import_dated("stored-old", old, content="An old stored memory.")
        runtime = MemoryRuntime(
            AgentMemory(
                self.store,
                FakeMemorySelector(),
                source="test",
                session_id="session-2",
            )
        )

        runtime.runtime_write_memory(
            _sample_exchange(),
            maintenance_settings=MaintenanceSettings(token_capacity=1),
        )

        stored_old = self.store.memory_get("stored-old")
        assert stored_old is not None
        self.assertEqual(stored_old.status, MemoryStatus.ACTIVE)
        self.assertTrue(stored_old.expired)
        self.assertEqual(
            self.store.count(),
            2,
        )

    def test_legacy_conversation_memories_map_to_episodes(self) -> None:
        self.store.import_record(
            memory_id="legacy-memory",
            content="A past event occurred.",
            memory_type="conversation",
            role="user",
            source="legacy",
            session_id="legacy-session",
            importance=0.5,
            created_at=datetime(2026, 9, 30, tzinfo=UTC),
            updated_at=datetime(2026, 9, 30, tzinfo=UTC),
            metadata={},
            embedding=[0.0, 0.0, 1.0],
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
        self.assertEqual(stored_memory.subject, "The user")
        self.assertEqual(stored_memory.predicate, "prefers")
        self.assertEqual(stored_memory.object, "concise answers")
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
        self.assertEqual(len(result.decisions), 1)
        self.assertIsNone(result.decisions[0].relation)
        self.assertEqual(
            result.decisions[0].stored.id,
            stored_memory.id,
        )

    def test_write_stores_candidate_below_related_threshold(self) -> None:
        classifier = ScriptedCandidateRelationClassifier("equivalent")
        self.store.write_memory(
            "The user prefers Python.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )

        result = self._write_selected_candidate(classifier)

        self.assertEqual(classifier.calls, [])
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(len(result.stored_memories), 1)
        self.assertIsNone(result.decisions[0].relation)
        self.assertEqual(result.decisions[0].related, ())

    def test_candidate_relations_set_status_and_links(self) -> None:
        cases = (
            ("equivalent", None, MemoryStatus.ACTIVE, None, 1),
            (
                "refinement",
                MemoryStatus.ACTIVE,
                MemoryStatus.SUPERSEDED,
                MemoryLinkRelation.REFINES,
                2,
            ),
            (
                "supersession",
                MemoryStatus.ACTIVE,
                MemoryStatus.SUPERSEDED,
                MemoryLinkRelation.SUPERSEDES,
                2,
            ),
            (
                "correction",
                MemoryStatus.ACTIVE,
                MemoryStatus.CORRECTED,
                MemoryLinkRelation.CORRECTS,
                2,
            ),
            (
                "contradiction",
                MemoryStatus.CONFLICTED,
                MemoryStatus.CONFLICTED,
                MemoryLinkRelation.CONFLICTS,
                2,
            ),
            (
                "related_distinct",
                MemoryStatus.ACTIVE,
                MemoryStatus.ACTIVE,
                None,
                2,
            ),
        )
        for (
            relation,
            new_status,
            old_status,
            link_relation,
            count,
        ) in cases:
            with self.subTest(relation=relation):
                self.store.clear()
                existing = self.store.write_memory(
                    "The user prefers concise answers.",
                    memory_type=MemoryType.PREFERENCE,
                    role="user",
                    source="test",
                    session_id="session-2",
                    importance=0.8,
                )
                classifier = ScriptedCandidateRelationClassifier(relation)

                result = self._write_selected_candidate(classifier)

                self.assertEqual(self.store.count(), count)
                decision = result.decisions[0]
                self.assertEqual(decision.relation, relation)
                self.assertEqual(
                    [match.memory.id for match in decision.related],
                    [existing.id],
                )
                kept = self.store.memory_get(existing.id)
                self.assertIsNotNone(kept)
                assert kept is not None
                self.assertEqual(kept.status, old_status)
                if new_status is None:
                    self.assertIsNone(decision.stored)
                    self.assertEqual(decision.links, ())
                    self.assertEqual(result.stored_memories, ())
                    continue
                self.assertIsNotNone(decision.stored)
                assert decision.stored is not None
                self.assertEqual(decision.stored.status, new_status)
                self.assertEqual(
                    result.stored_memories,
                    (decision.stored,),
                )
                if link_relation is None:
                    self.assertEqual(decision.links, ())
                    self.assertEqual(
                        self.store.memory_links_from(decision.stored.id),
                        [],
                    )
                    continue
                self.assertEqual(len(decision.links), 1)
                link = decision.links[0]
                self.assertEqual(link.from_memory_id, decision.stored.id)
                self.assertEqual(link.relation, link_relation)
                self.assertEqual(link.to_memory_id, existing.id)
                self.assertEqual(
                    self.store.memory_links_from(decision.stored.id),
                    [link],
                )

    def test_refinement_links_every_related_memory(self) -> None:
        first = self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )
        second = self.store.write_memory(
            "The user likes concise replies.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.7,
        )
        classifier = ScriptedCandidateRelationClassifier("refinement")

        result = self._write_selected_candidate(classifier)

        self.assertEqual(
            {link.to_memory_id for link in result.decisions[0].links},
            {first.id, second.id},
        )
        self.assertEqual(
            self.store.memory_get(first.id).status,
            MemoryStatus.SUPERSEDED,
        )
        self.assertEqual(
            self.store.memory_get(second.id).status,
            MemoryStatus.SUPERSEDED,
        )

    def test_superseded_memory_stays_out_of_retrieval(self) -> None:
        existing = self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )
        result = self._write_selected_candidate(
            ScriptedCandidateRelationClassifier("refinement")
        )
        stored = result.stored_memories[0]
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
        )

        found = memory_api.memory_search("concise answers")

        found_ids = [item.memory.id for item in found.matches]
        self.assertIn(stored.id, found_ids)
        self.assertNotIn(existing.id, found_ids)

    def test_conflicted_memory_stays_out_of_retrieval(self) -> None:
        active = self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )
        conflicted = self.store.write_memory(
            "The user likes concise replies.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.7,
        )
        self.store.record_memory_link(
            from_memory_id=active.id,
            to_memory_id=conflicted.id,
            relation=MemoryLinkRelation.CONFLICTS,
            to_status=MemoryStatus.CONFLICTED,
        )
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
        )

        found = memory_api.memory_search("concise answers")

        found_ids = [item.memory.id for item in found.matches]
        self.assertIn(active.id, found_ids)
        self.assertNotIn(conflicted.id, found_ids)

    def test_write_keeps_similarity_above_point_nine_only(self) -> None:
        classifier = ScriptedCandidateRelationClassifier("correction")
        self.store._embed = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        self._import_vector("at-threshold", _vector_at_similarity(0.9))
        self._import_vector("above-threshold", _vector_at_similarity(0.95))

        result = self._write_selected_candidate(
            classifier,
            WritePathSettings(related_similarity=0.9),
        )

        self.assertEqual(
            [match.memory.id for match in result.decisions[0].related],
            ["above-threshold"],
        )
        self.assertEqual(result.decisions[0].relation, "correction")
        self.assertEqual(self.store.count(), 3)
        corrected = self.store.memory_get("above-threshold")
        untouched = self.store.memory_get("at-threshold")
        self.assertIsNotNone(corrected)
        self.assertIsNotNone(untouched)
        assert corrected is not None and untouched is not None
        self.assertEqual(corrected.status, MemoryStatus.CORRECTED)
        self.assertEqual(untouched.status, MemoryStatus.ACTIVE)
        self.assertEqual(
            result.decisions[0].links[0].relation,
            MemoryLinkRelation.CORRECTS,
        )
        self.assertEqual(
            result.decisions[0].links[0].to_memory_id,
            "above-threshold",
        )

    def test_write_classifies_at_most_five_related_memories(self) -> None:
        classifier = ScriptedCandidateRelationClassifier("related_distinct")
        for index in range(6):
            self._import_vector(
                f"related-{index}",
                [0.0, 1.0, 0.0],
                content=f"The user prefers concise answers {index}.",
            )

        result = self._write_selected_candidate(classifier)

        self.assertEqual(len(result.decisions[0].related), 5)
        self.assertEqual(len(classifier.calls[0][1]), 5)
        self.assertEqual(result.decisions[0].relation, "related_distinct")
        self.assertEqual(result.decisions[0].links, ())
        self.assertEqual(self.store.count(), 7)
        stored = result.decisions[0].stored
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.status, MemoryStatus.ACTIVE)
        for match in result.decisions[0].related:
            kept = self.store.memory_get(match.memory.id)
            self.assertIsNotNone(kept)
            assert kept is not None
            self.assertEqual(kept.status, MemoryStatus.ACTIVE)

    def test_related_write_requires_a_classifier(self) -> None:
        self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
        )

        with self.assertRaises(RuntimeError):
            memory_api.memory_add(_sample_exchange())

        self.assertEqual(self.store.count(), 1)

    def test_unknown_candidate_relation_is_rejected(self) -> None:
        classifier = ScriptedCandidateRelationClassifier("maybe")
        self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )

        with self.assertRaises(RuntimeError):
            self._write_selected_candidate(classifier)

        self.assertEqual(self.store.count(), 1)

    def _write_selected_candidate(self, classifier, settings=None):
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            candidate_relation_classifier=classifier,
        )
        return MemoryRuntime(memory_api).runtime_write_memory(
            _sample_exchange(),
            settings=settings,
        )

    def _write_non_atomic_candidate(self, checker):
        memory_api = AgentMemory(
            self.store,
            NonAtomicMemorySelector(),
            source="test",
            session_id="session-2",
            atomic_checker=checker,
        )
        return memory_api.memory_add(_sample_exchange())

    def test_score_below_threshold_skips_the_atomic_checker(self) -> None:
        checker = ScriptedAtomicChecker("invalid")
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            atomic_checker=checker,
        )

        result = memory_api.memory_add(_sample_exchange())

        self.assertEqual(checker.calls, [])
        self.assertIsNone(result.decisions[0].atomic_verdict)
        self.assertEqual(len(result.stored_memories), 1)
        self.assertLess(result.decisions[0].atomic_score.score, 0.2)

    def test_raised_atomic_threshold_skips_the_checker(self) -> None:
        checker = ScriptedAtomicChecker("invalid")
        memory_api = AgentMemory(
            self.store,
            NonAtomicMemorySelector(),
            source="test",
            session_id="session-2",
            atomic_checker=checker,
        )

        result = memory_api.memory_add(
            _sample_exchange(),
            settings=WritePathSettings(atomic_score_threshold=1.0),
        )

        self.assertEqual(checker.calls, [])
        self.assertIsNone(result.decisions[0].atomic_verdict)
        self.assertEqual(len(result.stored_memories), 1)

    def test_invalid_atomic_memory_is_discarded(self) -> None:
        checker = ScriptedAtomicChecker("invalid")

        result = self._write_non_atomic_candidate(checker)

        self.assertEqual(checker.calls, ["User likes Python and Java."])
        self.assertEqual(result.decisions[0].atomic_verdict, "invalid")
        self.assertIsNone(result.decisions[0].stored)
        self.assertEqual(result.stored_memories, ())
        self.assertGreaterEqual(result.decisions[0].atomic_score.score, 0.2)
        self.assertEqual(self.store.count(), 0)

    def test_valid_atomic_memory_continues_to_storage(self) -> None:
        checker = ScriptedAtomicChecker("valid")

        result = self._write_non_atomic_candidate(checker)

        self.assertEqual(result.decisions[0].atomic_verdict, "valid")
        self.assertEqual(len(result.stored_memories), 1)
        self.assertEqual(self.store.count(), 1)

    def test_atomic_review_requires_a_checker(self) -> None:
        memory_api = AgentMemory(
            self.store,
            NonAtomicMemorySelector(),
            source="test",
            session_id="session-2",
        )

        with self.assertRaises(RuntimeError):
            memory_api.memory_add(_sample_exchange())

        self.assertEqual(self.store.count(), 0)

    def test_related_limit_keeps_only_the_requested_count(self) -> None:
        self.store.write_memory(
            "The user prefers concise answers.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.8,
        )
        self.store.write_memory(
            "The user likes concise replies.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
            importance=0.7,
        )

        result = self._write_selected_candidate(
            ScriptedCandidateRelationClassifier("related_distinct"),
            WritePathSettings(related_limit=1),
        )

        self.assertEqual(len(result.decisions[0].related), 1)
        self.assertEqual(result.decisions[0].links, ())

    def test_higher_related_similarity_ignores_a_weaker_match(self) -> None:
        self.store._embed = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        self._import_vector("near-match", _vector_at_similarity(0.95))

        result = self._write_selected_candidate(
            ScriptedCandidateRelationClassifier("correction"),
            WritePathSettings(related_similarity=0.99),
        )

        self.assertEqual(result.decisions[0].related, ())
        self.assertIsNone(result.decisions[0].relation)
        self.assertEqual(len(result.stored_memories), 1)
        self.assertEqual(self.store.count(), 2)

    def _import_dated(
        self,
        memory_id: str,
        moment: datetime,
        *,
        content: str,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        last_accessed_at: datetime | None = None,
        reinforced_at: datetime | None = None,
        embedding: list[float] | None = None,
        importance: float = 0.5,
    ) -> None:
        self.store.import_record(
            memory_id=memory_id,
            content=content,
            memory_type="fact",
            role="user",
            source="test",
            session_id="session-2",
            importance=importance,
            created_at=moment,
            updated_at=moment,
            metadata={},
            embedding=embedding or [0.0, 0.0, 1.0],
            status=status,
            last_accessed_at=last_accessed_at,
            reinforced_at=reinforced_at,
        )

    def _import_vector(
        self,
        memory_id: str,
        embedding: list[float],
        content: str = "An existing memory.",
    ) -> None:
        moment = datetime(2026, 9, 30, tzinfo=UTC)
        self.store.import_record(
            memory_id=memory_id,
            content=content,
            memory_type="fact",
            role="user",
            source="test",
            session_id="session-2",
            importance=0.5,
            created_at=moment,
            updated_at=moment,
            metadata={},
            embedding=embedding,
        )

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
        self.accessed_ids: list[str] = []

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

    def memory_mark_accessed(self, memory_ids, *, accessed_at):
        self.accessed_ids = list(memory_ids)
        self.accessed_at = accessed_at


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
        token_count=count_tokens(content),
    )


class ReadPathRankingTest(unittest.TestCase):
    def test_recency_decays_exponentially_from_created_at(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        created_at = now - timedelta(days=30)
        score = calculate_recency_score(created_at, now=now)
        self.assertAlmostEqual(score, math.exp(-RECENCY_LAMBDA_PER_DAY * 30))
        self.assertAlmostEqual(score, 0.5)

    def test_recency_uses_reinforced_at(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        memory = _memory(
            memory_id="reinforced",
            content="The user prefers Python.",
            importance=0.5,
            created_at=now - timedelta(days=30),
        )
        reinforced = Memory(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type,
            role=memory.role,
            source=memory.source,
            session_id=memory.session_id,
            importance=memory.importance,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            metadata=memory.metadata,
            token_count=memory.token_count,
            reinforced_at=now,
        )
        scored = score_retrieved_memory(
            MemoryMatch(memory=reinforced, similarity_score=1.0),
            now=now,
        )
        self.assertAlmostEqual(scored.recency_score, 1.0)

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


def _sample_exchange() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "Please keep answers concise.",
        },
        {
            "role": "assistant",
            "content": "I will keep them concise.",
        },
    ]


def _vector_at_similarity(similarity: float) -> list[float]:
    lateral = math.sqrt(1.0 - similarity * similarity)
    return [similarity, lateral, 0.0]


class NonAtomicMemorySelector:
    model_name = "fake-memory-selector"

    def extract_memory(self, conversation):
        return MemorySelectionOutput(
            selected=[
                MemoryCandidate(
                    content="User likes Python and Java.",
                    subject="User",
                    predicate="likes",
                    object="Python and Java",
                    type=MemoryType.PREFERENCE,
                    importance=0.8,
                    confidence=0.9,
                    role="user",
                    debug_source_quote="I like Python and Java.",
                    debug_reason="The user named two languages.",
                )
            ],
            not_selected=[],
        )


class ScriptedDedupClassifier:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, list[str]]] = []

    def classify(self, candidate, related):
        self.calls.append(
            (candidate.id, [memory.id for memory in related])
        )
        return self.verdict


class ScriptedAtomicChecker:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls: list[str] = []

    def check(self, candidate):
        self.calls.append(candidate.content)
        return self.verdict


class ScriptedCandidateRelationClassifier:
    def __init__(self, relation: str) -> None:
        self.relation = relation
        self.calls: list[tuple[str, list[str]]] = []

    def classify(self, candidate_content, related):
        self.calls.append(
            (
                candidate_content,
                [match.memory.id for match in related],
            )
        )
        return self.relation


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


class WritePathSettingsTest(unittest.TestCase):
    def test_rejects_atomic_weights_that_do_not_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            WritePathSettings(predicate_weight=0.5)

    def test_rejects_a_related_count_below_one(self) -> None:
        with self.assertRaises(ValueError):
            WritePathSettings(related_limit=0)

    def test_rejects_dedup_weights_that_do_not_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            MaintenanceSettings(importance_weight=0.8, recency_weight=0.8)

    def test_rejects_an_atomic_threshold_outside_zero_to_one(self) -> None:
        with self.assertRaises(ValueError):
            WritePathSettings(atomic_score_threshold=1.1)


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


class AtomicScoreTest(unittest.TestCase):
    def test_component_weights_match_the_formula(self) -> None:
        scored = component_score(
            predicate_count=2,
            coordinated_item_count=3,
            clause_count=3,
            coordination_count=2,
            value_count=3,
            token_count=45,
        )
        self.assertEqual(scored.predicate, 1)
        self.assertEqual(scored.coordinated_items, 1)
        self.assertEqual(scored.clause, 1)
        self.assertEqual(scored.coordination, 1)
        self.assertEqual(scored.value, 1)
        self.assertEqual(scored.tokens, 1)
        self.assertAlmostEqual(scored.score, 1.0)

        partial = component_score(1, 2, 1, 1, 2, 15)
        self.assertEqual(partial.predicate, 0)
        self.assertAlmostEqual(partial.coordinated_items, 0.5)
        self.assertEqual(partial.clause, 0)
        self.assertEqual(partial.coordination, 1)
        self.assertAlmostEqual(partial.value, 0.5)
        self.assertEqual(partial.tokens, 0)
        self.assertAlmostEqual(partial.score, 0.25)

    def test_single_proposition_scores_zero(self) -> None:
        scored = score_atomic_memory("User likes Python.")
        self.assertEqual(
            (
                scored.predicate_count,
                scored.clause_count,
                scored.coordinated_item_count,
                scored.coordination_count,
                scored.value_count,
            ),
            (1, 1, 0, 0, 0),
        )
        self.assertEqual(scored.score, 0)

    def test_coordinated_list_raises_item_and_coordination_scores(self) -> None:
        pair = score_atomic_memory("User likes Python and Java.")
        self.assertEqual(pair.coordinated_item_count, 2)
        self.assertEqual(pair.coordination_count, 1)
        self.assertAlmostEqual(pair.coordinated_items, 0.5)
        self.assertEqual(pair.coordination, 1)
        self.assertAlmostEqual(pair.score, 0.2)

        trio = score_atomic_memory("User likes Python, Java, and Rust.")
        self.assertEqual(trio.coordinated_item_count, 3)
        self.assertEqual(trio.coordinated_items, 1)
        self.assertAlmostEqual(trio.score, 0.3)

    def test_two_numbers_raise_the_value_score(self) -> None:
        scored = score_atomic_memory(
            "User scored 116 on TOEFL and 1500 on SAT."
        )
        self.assertEqual(scored.predicate_count, 1)
        self.assertEqual(scored.value_count, 2)
        self.assertAlmostEqual(scored.value, 0.5)
        self.assertEqual(scored.coordination_count, 1)
        self.assertGreater(scored.score, 0)

    def test_one_version_number_does_not_raise_the_value_score(self) -> None:
        scored = score_atomic_memory("MemoryOS uses Python 3.12.")
        self.assertEqual(scored.value_count, 1)
        self.assertEqual(scored.value, 0)
        self.assertEqual(scored.predicate_count, 1)
        self.assertEqual(scored.score, 0)

    def test_two_predicates_and_a_contrast_raise_the_score(self) -> None:
        scored = score_atomic_memory("User likes Python but prefers Java.")
        self.assertEqual(scored.predicate_count, 2)
        self.assertEqual(scored.clause_count, 2)
        self.assertEqual(scored.coordination_count, 1)
        self.assertEqual(scored.coordinated_item_count, 0)
        self.assertEqual(scored.predicate, 1)
        self.assertAlmostEqual(scored.clause, 0.5)
        self.assertAlmostEqual(scored.score, 0.525)

    def test_review_starts_at_point_two(self) -> None:
        self.assertFalse(needs_atomic_review(0.199))
        self.assertTrue(needs_atomic_review(0.2))

    def test_custom_weights_change_the_score(self) -> None:
        weights = AtomicScoreWeights(
            predicate=1.0,
            coordinated_items=0.0,
            clause=0.0,
            coordination=0.0,
            value=0.0,
            tokens=0.0,
        )
        scored = component_score(2, 3, 3, 2, 3, 45, weights=weights)
        self.assertAlmostEqual(scored.score, 1.0)

    def test_coordinated_properties_count_as_two_predicates(self) -> None:
        scored = score_atomic_memory("User is tall and happy.")
        self.assertEqual(scored.predicate_count, 2)
        self.assertEqual(scored.clause_count, 1)
        self.assertEqual(scored.coordinated_item_count, 2)
        self.assertEqual(scored.coordination_count, 1)


if __name__ == "__main__":
    unittest.main()
