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
from ai_agent_memory_infrastructure.memory_atomic_rewriter import (
    AtomicRewritePiece,
)
from ai_agent_memory_infrastructure.memory_entity_similarity import (
    embedding_cosine_similarity,
    jaccard_token_similarity,
    normalized_levenshtein_similarity,
)
from ai_agent_memory_infrastructure.memory_graph import (
    GraphEdgeDraft,
    memory_connections,
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
        self.assertEqual(result.groups[0].verdict, "equivalent")
        actions = {
            member.memory_id: member.action
            for member in result.groups[0].members
        }
        self.assertEqual(actions["newer-copy"], "kept")
        self.assertEqual(actions["older-copy"], "removed")
        removed = next(
            member
            for member in result.groups[0].members
            if member.memory_id == "older-copy"
        )
        self.assertEqual(
            removed.content,
            "The user prefers concise answers.",
        )
        self.assertEqual(removed.status, "active")

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
        self.assertEqual(result.groups[0].verdict, "not_equivalent")
        self.assertTrue(
            all(
                member.action == "unchanged"
                for member in result.groups[0].members
            )
        )
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
        expired_by_id = {
            memory.memory_id: memory for memory in expired.expired_memories
        }
        self.assertEqual(
            set(expired_by_id),
            {"old-active", "old-superseded"},
        )
        self.assertEqual(expired_by_id["old-active"].status, "active")
        self.assertEqual(
            expired_by_id["old-superseded"].content,
            "An old superseded memory.",
        )
        self.assertEqual(expired_by_id["old-superseded"].status, "superseded")
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

    def test_each_related_memory_keeps_its_own_relation(self) -> None:
        calculus = self.store.write_memory(
            "The user prefers concise answers about calculus.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
        )
        algebra = self.store.write_memory(
            "The user prefers concise answers about algebra.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
        )
        classifier = ScriptedCandidateRelationClassifier(
            {
                calculus.id: "refinement",
                algebra.id: "related_distinct",
            }
        )

        result = self._write_selected_candidate(classifier)

        self.assertIsNone(result.decisions[0].relation)
        self.assertEqual(
            {link.to_memory_id for link in result.decisions[0].links},
            {calculus.id},
        )
        self.assertEqual(
            self.store.memory_get(calculus.id).status,
            MemoryStatus.SUPERSEDED,
        )
        self.assertEqual(
            self.store.memory_get(algebra.id).status,
            MemoryStatus.ACTIVE,
        )
        stored = result.decisions[0].stored
        assert stored is not None
        self.assertEqual(stored.status, MemoryStatus.ACTIVE)

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

    def _write_non_atomic_candidate(self, checker, rewriter=None):
        memory_api = AgentMemory(
            self.store,
            NonAtomicMemorySelector(),
            source="test",
            session_id="session-2",
            atomic_checker=checker,
            atomic_rewriter=rewriter,
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

    def test_invalid_atomic_memory_is_rewritten_and_stored(self) -> None:
        checker = ScriptedAtomicChecker("invalid")
        rewriter = ScriptedAtomicRewriter(
            [
                ("User likes Python.", "User", "likes", "Python"),
                ("User likes Java.", "User", "likes", "Java"),
            ]
        )

        result = self._write_non_atomic_candidate(checker, rewriter)

        self.assertEqual(checker.calls, ["User likes Python and Java."])
        self.assertEqual(rewriter.calls, ["User likes Python and Java."])
        self.assertEqual(result.decisions[0].atomic_verdict, "invalid")
        self.assertIsNone(result.decisions[0].stored)
        self.assertEqual(
            result.decisions[0].rewritten,
            ("User likes Python.", "User likes Java."),
        )
        self.assertEqual(len(result.stored_memories), 2)
        self.assertEqual(
            {memory.content for memory in result.stored_memories},
            {"User likes Python.", "User likes Java."},
        )
        self.assertTrue(
            all(decision.from_rewrite for decision in result.decisions[1:])
        )
        self.assertEqual(
            result.decisions[1].candidate.importance,
            0.8,
        )
        self.assertEqual(self.store.count(), 2)

    def test_invalid_atomic_memory_requires_a_rewriter(self) -> None:
        with self.assertRaises(RuntimeError):
            self._write_non_atomic_candidate(ScriptedAtomicChecker("invalid"))

        self.assertEqual(self.store.count(), 0)

    def test_rewrite_that_repeats_the_candidate_is_discarded(self) -> None:
        rewriter = ScriptedAtomicRewriter(
            [
                (
                    "User likes Python and Java.",
                    "User",
                    "likes",
                    "Python and Java",
                )
            ]
        )

        result = self._write_non_atomic_candidate(
            ScriptedAtomicChecker("invalid"),
            rewriter,
        )

        self.assertEqual(result.decisions[0].rewritten, ())
        self.assertEqual(result.stored_memories, ())
        self.assertEqual(self.store.count(), 0)

    def test_a_second_invalid_rewrite_is_discarded(self) -> None:
        checker = ScriptedAtomicChecker("invalid")
        rewriter = ScriptedAtomicRewriter(
            [
                (
                    "User likes Python and Java too.",
                    "User",
                    "likes",
                    "Python and Java",
                )
            ]
        )

        result = self._write_non_atomic_candidate(checker, rewriter)

        self.assertEqual(
            checker.calls,
            [
                "User likes Python and Java.",
                "User likes Python and Java too.",
            ],
        )
        self.assertEqual(rewriter.calls, ["User likes Python and Java."])
        self.assertEqual(result.decisions[1].atomic_verdict, "invalid")
        self.assertEqual(result.decisions[1].rewritten, ())
        self.assertEqual(result.stored_memories, ())
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

    def test_graph_connects_memories_that_share_an_entity(self) -> None:
        major = self.store.write_memory(
            "The user is majoring in Economics at New York University.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        graduate = self.store.write_memory(
            "The user expects to graduate from New York University in May 2027.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        self._import_dated(
            "old-job",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user worked at a previous company.",
            status=MemoryStatus.SUPERSEDED,
        )
        self._import_dated(
            "quiet",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user lived in Boston.",
            expired=True,
        )
        extractor = ScriptedGraphExtractor(
            {
                major.id: (
                    ("The user", "majoring in", "Economics"),
                    ("The user", "majoring at", "New York University"),
                ),
                graduate.id: (
                    (
                        "The user",
                        "expects to graduate from",
                        "New York University",
                    ),
                    ("The user", "expects to graduate in", "May 2027"),
                ),
                "old-job": (("The user", "worked at", "Previous Company"),),
                "quiet": (("The user", "lived in", "Boston"),),
            }
        )

        result = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
            entity_judge=ScriptedEntityJudge("not_same_entity"),
        ).memory_graph_build()

        self.assertEqual(extractor.calls, [major.id, graduate.id])
        self.assertEqual(result.built_count, 2)
        self.assertEqual(result.entity_count, 4)
        records = self.store.graph_edge_records()
        self.assertEqual(
            {
                (record[0], record[2])
                for record in records
                if record[2] == "New York University"
            },
            {
                ("The user", "New York University"),
            },
        )
        self.assertEqual(
            {record[3] for record in records},
            {major.id, graduate.id},
        )

        again = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
        ).memory_graph_build()

        self.assertEqual(again.built_count, 0)
        self.assertEqual(extractor.calls, [major.id, graduate.id])

    def test_new_entity_mention_is_scored_against_each_existing_entity(
        self,
    ) -> None:
        self._import_dated(
            "school",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user studies at New York University.",
        )
        self._import_dated(
            "abbreviation",
            datetime(2026, 1, 2, tzinfo=UTC),
            content="The user studies at NYU.",
        )
        self._import_dated(
            "repeated",
            datetime(2026, 1, 3, tzinfo=UTC),
            content="The user attends New York University.",
        )
        extractor = ScriptedGraphExtractor(
            {
                "school": (
                    ("The user", "studies at", "New York University"),
                    ("The user", "studies", "Python"),
                ),
                "abbreviation": (("The user", "studies at", "NYU"),),
                "repeated": (
                    ("The user", "attends", "New York University"),
                ),
            }
        )
        embedded: list[str] = []
        original_embed = self.store._embed

        def tracking_embed(texts):
            embedded.extend(text.casefold() for text in texts)
            return original_embed(texts)

        self.store._embed = tracking_embed
        judge = ScriptedEntityJudge("not_same_entity")
        graph = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
            entity_judge=judge,
        )

        graph.memory_graph_build()

        self.assertEqual(
            [existing for _candidate, existing in judge.calls],
            ["New York University", "The user"],
        )
        self.assertEqual(embedded.count("new york university"), 1)
        self.assertEqual(embedded.count("nyu"), 1)
        self.assertEqual(self.store.graph_entity_count(), 4)
        self.assertEqual(self.store.graph_entity_aliases(), ())

    def test_same_entity_judgment_stores_an_alias_instead_of_a_new_entity(
        self,
    ) -> None:
        self._import_dated(
            "school",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user studies at New York University.",
        )
        self._import_dated(
            "abbreviation",
            datetime(2026, 1, 2, tzinfo=UTC),
            content="The user studies at NYU.",
        )
        extractor = ScriptedGraphExtractor(
            {
                "school": (
                    ("The user", "studies at", "New York University"),
                    ("The user", "studies", "Python"),
                ),
                "abbreviation": (("The user", "studies at", "NYU"),),
            }
        )
        judge = ScriptedEntityJudge(
            {
                ("nyu", "new york university"): "same_entity",
                ("nyu", "the user"): "not_same_entity",
            }
        )
        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
            entity_judge=judge,
        ).memory_graph_build()

        self.assertEqual(
            [existing for _candidate, existing in judge.calls],
            ["New York University", "The user"],
        )
        self.assertEqual(self.store.graph_entity_count(), 3)
        self.assertEqual(
            self.store.graph_entity_aliases(),
            (("NYU", "New York University"),),
        )
        self.assertNotIn(
            "NYU",
            {name for _entity_id, name, _embedding in self.store.graph_entity_names()},
        )
        self.assertIn(
            ("The user", "studies at", "New York University", "abbreviation"),
            {
                (record[0], record[1], record[2], record[3])
                for record in self.store.graph_edge_records()
            },
        )

        self._import_dated(
            "later",
            datetime(2026, 1, 4, tzinfo=UTC),
            content="NYU is still the school.",
        )
        extractor.edges_by_id["later"] = (("NYU", "is", "NYU"),)
        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
            entity_judge=judge,
        ).memory_graph_build()

        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(self.store.graph_entity_count(), 3)

    def test_several_same_entities_are_merged_into_the_selected_entity(
        self,
    ) -> None:
        self._import_dated(
            "school",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user studies at New York University.",
        )
        self._import_dated(
            "abbreviation",
            datetime(2026, 1, 2, tzinfo=UTC),
            content="The user studies at NYU.",
        )
        judge = ScriptedEntityJudge(
            {
                ("nyu", "new york university"): "same_entity",
                ("nyu", "the user"): "same_entity",
            }
        )
        selector = ScriptedEntitySelector("New York University")
        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=ScriptedGraphExtractor(
                {
                    "school": (
                        ("The user", "studies at", "New York University"),
                        ("The user", "studies", "Python"),
                    ),
                    "abbreviation": (("The user", "studies at", "NYU"),),
                }
            ),
            entity_judge=judge,
            entity_selector=selector,
        ).memory_graph_build()

        self.assertEqual(
            selector.calls,
            [("New York University", "The user")],
        )
        self.assertEqual(
            {name for _entity_id, name, _embedding in self.store.graph_entity_names()},
            {"New York University", "Python"},
        )
        self.assertEqual(
            self.store.graph_entity_aliases(),
            (
                ("NYU", "New York University"),
                ("The user", "New York University"),
            ),
        )
        self.assertFalse(
            any(
                record[0] == "The user" or record[2] == "The user"
                for record in self.store.graph_edge_records()
            )
        )

    def test_a_name_below_every_threshold_is_stored_without_a_judge(self) -> None:
        self._import_dated(
            "tool",
            datetime(2026, 1, 1, tzinfo=UTC),
            content="The user uses Python.",
        )
        self._import_dated(
            "animal",
            datetime(2026, 1, 2, tzinfo=UTC),
            content="The user saw a zebra.",
        )

        def embed(texts):
            vectors = {"python": [1.0, 0.0, 0.0], "zebra": [0.0, 1.0, 0.0]}
            return [vectors[text.casefold()] for text in texts]

        self.store._embed = embed
        judge = ScriptedEntityJudge("same_entity")
        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=ScriptedGraphExtractor(
                {
                    "tool": (("Python", "is used", "Python"),),
                    "animal": (("Zebra", "was seen", "Zebra"),),
                }
            ),
            entity_judge=judge,
        ).memory_graph_build()

        self.assertEqual(judge.calls, [])
        self.assertEqual(
            {name for _entity_id, name, _embedding in self.store.graph_entity_names()},
            {"Python", "Zebra"},
        )
        self.assertEqual(self.store.graph_entity_aliases(), ())

    def test_graph_drops_edges_when_a_memory_is_no_longer_current(self) -> None:
        kept = self.store.write_memory(
            "The user is majoring in Economics at New York University.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        removed = self.store.write_memory(
            "The user expects to graduate from New York University in May 2027.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        extractor = ScriptedGraphExtractor(
            {
                kept.id: (
                    ("The user", "majoring at", "New York University"),
                ),
                removed.id: (
                    (
                        "The user",
                        "expects to graduate from",
                        "New York University",
                    ),
                ),
            }
        )
        memory_api = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
        )
        memory_api.memory_graph_build()
        self.store.record_memory_link(
            from_memory_id=kept.id,
            to_memory_id=removed.id,
            relation=MemoryLinkRelation.SUPERSEDES,
            to_status=MemoryStatus.SUPERSEDED,
        )

        result = memory_api.memory_graph_build()

        self.assertEqual(result.edge_count, 1)
        self.assertEqual(
            {record[3] for record in self.store.graph_edge_records()},
            {kept.id},
        )
        self.assertEqual(result.entity_count, 2)

    def test_memory_link_records_include_both_contents(self) -> None:
        older = self.store.write_memory(
            "The user likes Python.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
        )
        newer = self.store.write_memory(
            "The user likes Java now.",
            memory_type=MemoryType.PREFERENCE,
            role="user",
            source="test",
            session_id="session-2",
        )
        self.store.record_memory_link(
            from_memory_id=newer.id,
            to_memory_id=older.id,
            relation=MemoryLinkRelation.SUPERSEDES,
            to_status=MemoryStatus.SUPERSEDED,
        )

        self.assertEqual(
            self.store.memory_link_records(),
            [("SUPERSEDES", "The user likes Java now.", "The user likes Python.")],
        )

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
        expired: bool = False,
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
            expired=expired,
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

    def test_episodes_follow_entity_degree_and_keep_an_unchanged_title(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import EpisodeSettings

        first = self.store.write_memory(
            "Ada studies at North College.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        second = self.store.write_memory(
            "Ada continues at North College.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        third = self.store.write_memory(
            "Ada lives in City.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        fourth = self.store.write_memory(
            "Ada visits Town.",
            memory_type=MemoryType.FACT,
            role="user",
            source="test",
            session_id="session-2",
        )
        for index in range(4):
            self.store.write_memory(
                f"Ada keeps note {index}.",
                memory_type=MemoryType.FACT,
                role="user",
                source="test",
                session_id="session-2",
            )
        extractor = ScriptedGraphExtractor(
            {
                first.id: (("Ada", "studies at", "North College"),),
                second.id: (("Ada", "continues at", "North College"),),
                third.id: (("Ada", "lives in", "City"),),
                fourth.id: (("Ada", "visits", "Town"),),
            }
        )
        AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            graph_extractor=extractor,
            entity_judge=ScriptedEntityJudge("not_same_entity"),
        ).memory_graph_build()
        titler = ScriptedEpisodeTitler()
        memory = AgentMemory(
            self.store,
            FakeMemorySelector(),
            source="test",
            session_id="session-2",
            episode_titler=titler,
        )
        built = memory.memory_episode_build()
        self.assertEqual(built.episode_count, 1)
        self.assertEqual(built.titled_count, 1)
        college_members = tuple(sorted((first.id, second.id)))
        self.assertEqual(
            set(self.store.memory_episode_groups()),
            {
                (
                    "North College",
                    ("North College",),
                    college_members,
                ),
            },
        )
        again = memory.memory_episode_build()
        self.assertEqual(again.episode_count, 1)
        self.assertEqual(again.titled_count, 0)
        self.assertEqual(len(titler.calls), 1)
        self.assertEqual(self.store.memory_episode_member_counts(), (2,))
        self.assertIn("North College", self.store.memory_episode_summaries()[0])
        self.assertEqual(self.store.memory_episode_embeddings(), ([0.0, 0.0, 1.0],))
        embed_calls = {"count": 0}
        original_embed = self.store._embed

        def _counting_embed(texts):
            embed_calls["count"] += len(texts)
            return original_embed(texts)

        self.store._embed = _counting_embed
        again = memory.memory_episode_build()
        self.assertEqual(again.episode_count, 1)
        self.assertEqual(again.titled_count, 0)
        self.assertEqual(len(titler.calls), 1)
        self.assertEqual(embed_calls["count"], 0)
        self.store.clear_built_episodes()
        self.assertEqual(self.store.memory_episode_groups(), ())
        self.assertGreater(self.store.graph_edge_count(), 0)
        self.assertGreater(self.store.count(), 0)
        self.store.clear_built_graph()
        self.assertEqual(self.store.graph_edge_count(), 0)
        self.assertEqual(self.store.graph_entity_count(), 0)
        self.assertGreater(self.store.count(), 0)
        self.store.clear()
        self.assertEqual(self.store.memory_episode_groups(), ())

    def test_episode_search_returns_the_nearest_five(self) -> None:
        now = datetime.now(UTC)
        vectors = (
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        )
        with self.store._connection.cursor() as cursor:
            for index, vector in enumerate(vectors):
                cursor.execute(
                    """
                    INSERT INTO memory_episodes (
                        episode_id, entity_key, title, summary, source_key,
                        created_at, updated_at, member_count, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        f"ep{index}",
                        f"key{index}",
                        f"Title {index}",
                        "A stored summary.",
                        f"source{index}",
                        now,
                        now,
                        2,
                        vector,
                    ),
                )
        self.store._connection.commit()

        from ai_agent_memory_infrastructure.memory_hierarchical import (
            hierarchical_episode_retrieval,
        )

        found = hierarchical_episode_retrieval(
            self.store,
            "college notes",
        )
        self.assertEqual(
            [episode.episode_id for episode in found],
            ["ep0", "ep3", "ep4", "ep5", "ep1"],
        )
        self.assertAlmostEqual(found[0].similarity_score, 1.0)
        self.assertAlmostEqual(found[4].similarity_score, 0.0)
        self.assertEqual(found[0].member_count, 2)


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


class MemoryConnectionTest(unittest.TestCase):
    def test_links_two_memories_that_share_an_entity(self) -> None:
        view = memory_connections(
            [
                ("graduate", "Graduate from NYU.", "nyu", "New York University"),
                ("gpa", "GPA at NYU.", "nyu", "New York University"),
                ("graduate", "Graduate from NYU.", "user", "The user"),
                ("gpa", "GPA at NYU.", "user", "The user"),
                ("boston", "Lived in Boston.", "user", "The user"),
            ],
            broad_limit=2,
        )

        self.assertEqual(
            view.links,
            (("gpa", "graduate", ("New York University",)),),
        )
        self.assertEqual(
            view.groups,
            (
                ("user", "The user", ("boston", "gpa", "graduate")),
                ("nyu", "New York University", ("gpa", "graduate")),
            ),
        )
        self.assertEqual(view.broad_entities, (("The user", 3),))
        self.assertEqual(
            view.hubs,
            (("user", "The user", ("boston", "gpa", "graduate")),),
        )


class EpisodeRuleTest(unittest.TestCase):
    def test_prevalence_forms_specific_episodes_and_mixed_pairs(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import form_episodes

        mentions = []
        for memory_id in ("m1", "m2", "m3", "m4"):
            mentions.append((memory_id, memory_id, "person", "Person"))
            mentions.append((memory_id, memory_id, "wide", "Wide"))
        mentions.extend(
            (
                ("m1", "m1", "college", "College"),
                ("m2", "m2", "college", "College"),
                ("m1", "m1", "topic", "Topic"),
                ("m3", "m3", "city", "City"),
            )
        )
        episodes = form_episodes(mentions, 8)
        formed = {
            (episode.entity_names, episode.memory_ids) for episode in episodes
        }
        self.assertEqual(
            formed,
            {
                (("College",), ("m1", "m2")),
                (("College", "Person"), ("m1", "m2")),
                (("College", "Wide"), ("m1", "m2")),
            },
        )

    def test_a_smaller_store_raises_the_prevalence_threshold(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import (
            form_episodes,
            prevalence_threshold,
        )

        mentions = [
            (memory_id, memory_id, "person", "Person")
            for memory_id in ("m1", "m2", "m3", "m4")
        ]
        mentions.extend(
            (memory_id, memory_id, "wide", "Wide")
            for memory_id in ("m1", "m2", "m3", "m4")
        )
        self.assertAlmostEqual(prevalence_threshold(0), 0.50)
        self.assertAlmostEqual(prevalence_threshold(100), 0.25)
        self.assertLess(prevalence_threshold(200), 0.20)
        self.assertGreater(prevalence_threshold(10), prevalence_threshold(100))
        self.assertGreater(
            prevalence_threshold(100, decay_scale=500),
            prevalence_threshold(100, decay_scale=100),
        )
        self.assertAlmostEqual(
            prevalence_threshold(0, baseline=0.10),
            0.40,
        )
        episodes = form_episodes(mentions, 10)
        self.assertEqual(
            {episode.entity_names for episode in episodes},
            {("Person",), ("Wide",), ("Person", "Wide")},
        )
        self.assertEqual(form_episodes(mentions, 8), ())

    def test_pair_overlap_uses_the_less_common_entity(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import form_episodes

        mentions = []
        for index in range(1, 7):
            mentions.append((f"rare-{index}", f"rare-{index}", "rare", "Rare"))
        for index in range(1, 21):
            mentions.append((f"wide-{index}", f"wide-{index}", "wide", "Wide"))
        mentions.extend(
            (
                ("rare-1", "rare-1", "wide", "Wide"),
                ("rare-2", "rare-2", "wide", "Wide"),
            )
        )
        self.assertEqual(
            [
                episode.entity_names
                for episode in form_episodes(mentions, 100)
                if len(episode.entity_ids) == 2
            ],
            [],
        )
        mentions.append(("rare-3", "rare-3", "wide", "Wide"))
        pairs = [
            episode
            for episode in form_episodes(mentions, 100)
            if len(episode.entity_ids) == 2
        ]
        self.assertEqual([episode.entity_names for episode in pairs], [("Rare", "Wide")])
        self.assertEqual(pairs[0].memory_ids, ("rare-1", "rare-2", "rare-3"))

    def test_contained_episode_is_removed_before_completion(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import (
            EpisodeDraft,
            drop_redundant_episodes,
            episode_set_scores,
        )

        large = EpisodeDraft(
            ("wide",),
            ("Wide",),
            tuple(f"m{index}" for index in range(12)),
            tuple(f"m{index}" for index in range(12)),
        )
        inside = EpisodeDraft(
            ("rare",),
            ("Rare",),
            tuple(f"m{index}" for index in range(8)) + ("x1", "x2"),
            tuple(f"m{index}" for index in range(8)) + ("x1", "x2"),
        )
        separate = EpisodeDraft(
            ("other",),
            ("Other",),
            ("m0", "m1", "m8", "y1", "y2"),
            ("m0", "m1", "m8", "y1", "y2"),
        )
        jaccard, containment, novelty = episode_set_scores(
            set(inside.memory_ids),
            set(large.memory_ids),
        )
        self.assertAlmostEqual(jaccard, 8 / 14)
        self.assertAlmostEqual(containment, 0.80)
        self.assertAlmostEqual(novelty, 0.20)
        kept = drop_redundant_episodes(
            (inside, large, separate),
            containment_threshold=0.80,
            novelty_threshold=0.20,
        )
        self.assertEqual(
            [episode.entity_names for episode in kept],
            [("Wide",), ("Other",)],
        )

    def test_one_memory_stays_below_the_episode_minimum(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import form_episodes

        mentions = [("m1", "m1", "once", "Once")]
        self.assertEqual(form_episodes(mentions, 100), ())

    def test_two_memories_above_the_prevalence_bound_stay_apart(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import form_episodes

        mentions = [
            ("m1", "m1", "person", "Person"),
            ("m2", "m2", "person", "Person"),
        ]
        self.assertEqual(form_episodes(mentions, 4), ())

    def test_prevalence_must_be_between_zero_and_one(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import EpisodeSettings

        with self.assertRaises(ValueError):
            EpisodeSettings(mean_cohesion=1.5)
        with self.assertRaises(ValueError):
            EpisodeSettings(prevalence_decay_scale=0.5)

    def test_cohesion_summary_matches_the_centroid_similarities(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import (
            cohesion_summary,
            embedding_cohesion,
        )

        cohesion, minimum, deviation = cohesion_summary((0.88, 0.84, 0.86, 0.82))
        self.assertAlmostEqual(cohesion, 0.85)
        self.assertAlmostEqual(minimum, 0.82)
        self.assertAlmostEqual(deviation, math.sqrt(0.0005))
        cohesion, minimum, deviation = cohesion_summary((0.91, 0.88, 0.86, 0.43))
        self.assertAlmostEqual(cohesion, 0.77)
        self.assertAlmostEqual(minimum, 0.43)
        self.assertGreater(deviation, 0.15)

        same, same_minimum, same_deviation, scores = embedding_cohesion(
            ([1.0, 0.0], [1.0, 0.0])
        )
        self.assertAlmostEqual(same, 1.0)
        self.assertAlmostEqual(same_minimum, 1.0)
        self.assertAlmostEqual(same_deviation, 0.0)
        self.assertEqual(len(scores), 2)

    def test_one_entity_episode_drops_a_memory_below_minimum_cohesion(self) -> None:
        from ai_agent_memory_infrastructure.memory_episode import (
            EpisodeDraft,
            apply_single_entity_cohesion,
        )

        single = EpisodeDraft(
            entity_ids=("college",),
            entity_names=("College",),
            memory_ids=("m1", "m2", "m3"),
            memory_contents=("one", "two", "three"),
        )
        pair = EpisodeDraft(
            entity_ids=("college", "person"),
            entity_names=("College", "Person"),
            memory_ids=("m1", "m3"),
            memory_contents=("one", "three"),
        )
        embeddings = {
            "m1": [1.0, 0.0, 0.0],
            "m2": [1.0, 0.0, 0.0],
            "m3": [0.0, 1.0, 0.0],
        }
        kept = apply_single_entity_cohesion(
            (single, pair),
            embeddings,
            minimum_cohesion=0.60,
            mean_cohesion=0.70,
        )
        self.assertEqual(kept[0].memory_ids, ("m1", "m2"))
        self.assertAlmostEqual(kept[0].cohesion, 1.0)
        self.assertAlmostEqual(kept[0].minimum_cohesion, 1.0)
        self.assertAlmostEqual(kept[0].cohesion_stddev, 0.0)
        self.assertEqual(kept[1].memory_ids, ("m1", "m3"))
        self.assertIsNone(kept[1].cohesion)

        removed = apply_single_entity_cohesion(
            (single,),
            embeddings,
            minimum_cohesion=0.95,
            mean_cohesion=0.70,
        )
        self.assertEqual(removed, ())

        angle = 2 * math.acos(0.65)
        loose = EpisodeDraft(
            entity_ids=("college",),
            entity_names=("College",),
            memory_ids=("m1", "m2"),
            memory_contents=("one", "two"),
        )
        loose_kept = apply_single_entity_cohesion(
            (loose,),
            {
                "m1": [1.0, 0.0],
                "m2": [math.cos(angle), math.sin(angle)],
            },
            minimum_cohesion=0.60,
            mean_cohesion=0.70,
        )
        self.assertEqual(loose_kept, ())


class EntitySimilarityTest(unittest.TestCase):
    def test_score_vector_uses_levenshtein_jaccard_and_cosine(self) -> None:
        self.assertEqual(
            normalized_levenshtein_similarity("NYU", "nyu"),
            1.0,
        )
        self.assertAlmostEqual(
            normalized_levenshtein_similarity("cat", "bat"),
            2 / 3,
        )
        self.assertEqual(
            jaccard_token_similarity(
                "New York University",
                "York University",
            ),
            2 / 3,
        )
        self.assertEqual(jaccard_token_similarity("NYU", "New York University"), 0.0)
        self.assertAlmostEqual(
            embedding_cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            0.0,
        )
        self.assertAlmostEqual(
            embedding_cosine_similarity([1.0, 0.0], [1.0, 0.0]),
            1.0,
        )

    def test_any_score_at_a_threshold_is_ready_for_the_judge(self) -> None:
        from ai_agent_memory_infrastructure.memory_entity_similarity import (
            EntityComparison,
            EntityResolutionSettings,
            passes_entity_threshold,
        )

        below = EntityComparison("A", "id", "B", 0.79, 0.49, 0.79)
        levenshtein = EntityComparison("A", "id", "B", 0.80, 0.0, 0.0)
        jaccard = EntityComparison("A", "id", "B", 0.0, 0.50, 0.0)
        embedding = EntityComparison("A", "id", "B", 0.0, 0.0, 0.80)
        self.assertFalse(passes_entity_threshold(below))
        self.assertTrue(passes_entity_threshold(levenshtein))
        self.assertTrue(passes_entity_threshold(jaccard))
        self.assertTrue(passes_entity_threshold(embedding))
        stricter = EntityResolutionSettings(levenshtein_threshold=0.90)
        self.assertFalse(passes_entity_threshold(levenshtein, stricter))


class ScriptedEntitySelector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, ...]] = []

    def select(self, names) -> str:
        self.calls.append(tuple(names))
        return self.name


class ScriptedEntityJudge:
    def __init__(self, verdict: str | dict[tuple[str, str], str]) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def judge(self, candidate: str, existing: str) -> str:
        self.calls.append((candidate, existing))
        if isinstance(self.verdict, dict):
            from ai_agent_memory_infrastructure.memory_entity_similarity import (
                entity_name_key,
            )

            return self.verdict[(entity_name_key(candidate), entity_name_key(existing))]
        return self.verdict


class ScriptedEpisodeTitler:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def title(self, entity_names, memory_contents) -> tuple[str, str]:
        names = tuple(entity_names)
        contents = tuple(memory_contents)
        self.calls.append((names, contents))
        return " / ".join(names), " ".join(contents)


class ScriptedGraphExtractor:
    def __init__(
        self,
        edges_by_id: dict[str, tuple[tuple[str, str, str], ...]],
    ) -> None:
        self.edges_by_id = edges_by_id
        self.calls: list[str] = []

    def extract(self, memory):
        self.calls.append(memory.id)
        return [
            GraphEdgeDraft(
                from_entity=from_entity,
                relation=relation,
                to_entity=to_entity,
            )
            for from_entity, relation, to_entity in self.edges_by_id.get(
                memory.id,
                (),
            )
        ]


class ScriptedAtomicRewriter:
    def __init__(
        self,
        memories: list[tuple[str, str, str, str]],
    ) -> None:
        self.memories = memories
        self.calls: list[str] = []

    def rewrite(self, candidate):
        self.calls.append(candidate.content)
        return [
            AtomicRewritePiece(
                content=content,
                subject=subject,
                predicate=predicate,
                object=relation_object,
            )
            for content, subject, predicate, relation_object in self.memories
        ]


class ScriptedAtomicChecker:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls: list[str] = []

    def check(self, candidate):
        self.calls.append(candidate.content)
        return self.verdict


class ScriptedCandidateRelationClassifier:
    def __init__(self, relation: str | dict[str, str]) -> None:
        self.relation = relation
        self.calls: list[tuple[str, list[str]]] = []

    def classify(self, candidate_content, related):
        self.calls.append(
            (
                candidate_content,
                [match.memory.id for match in related],
            )
        )
        if isinstance(self.relation, dict):
            return [
                (match.memory.id, self.relation[match.memory.id])
                for match in related
            ]
        return [
            (match.memory.id, self.relation) for match in related
        ]


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


class HierarchicalAtomicRetrievalTest(unittest.TestCase):
    def test_semantic_pool_is_cut_to_the_weighted_top(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            hierarchical_atomic_retrieval,
        )

        moment = datetime(2026, 6, 1, tzinfo=UTC)
        matches = []
        for index in range(12):
            matches.append(
                MemoryMatch(
                    memory=Memory(
                        id=f"m{index}",
                        content=f"memory {index}",
                        memory_type=MemoryType.FACT,
                        role="user",
                        source="test",
                        session_id="session",
                        importance=0.5,
                        created_at=moment,
                        updated_at=moment,
                        metadata={},
                        reinforced_at=moment,
                    ),
                    similarity_score=1 - (index * 0.05),
                )
            )

        class _Store:
            def __init__(self) -> None:
                self.limit = None
                self.accessed: list[str] = []

            def memory_search(self, query, *, limit, min_similarity=None, filters=None):
                self.limit = limit
                return matches[:limit]

            def memory_mark_accessed(self, memory_ids, accessed_at) -> None:
                self.accessed = list(memory_ids)

        store = _Store()
        settings = ReadPathSettings()
        weighted, candidates = hierarchical_atomic_retrieval(
            store,
            "query",
            settings=settings,
            now=moment,
        )
        self.assertEqual(store.limit, 30)
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len(weighted), 10)
        self.assertEqual(
            [item.memory.id for item in weighted],
            [f"m{index}" for index in range(10)],
        )
        self.assertEqual(store.accessed, [f"m{index}" for index in range(12)])

    def test_one_reranker_orders_atomics_and_episodes_together(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            RetrievedEpisode,
            atomic_candidate_text,
            coselect_hierarchical_candidates,
            episode_candidate_text,
        )
        from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory

        moment = datetime(2026, 6, 1, tzinfo=UTC)
        atomic = RetrievedMemory(
            memory=Memory(
                id="memory-1",
                content="The user built an XGBoost scorecard with KS 0.271.",
                memory_type=MemoryType.FACT,
                role="user",
                source="test",
                session_id="session",
                importance=0.5,
                created_at=moment,
                updated_at=moment,
                metadata={},
            ),
            semantic_score=0.4,
            recency_score=0.4,
            retrieval_score=0.4,
            final_score=0.4,
        )
        episode = RetrievedEpisode(
            episode_id="episode-1",
            title="WeBank Internship",
            summary=(
                "Summer internship involving consumer lending analytics, "
                "risk dashboards, an AI assistant, XGBoost modeling, and "
                "recovery analysis."
            ),
            member_count=3,
            similarity_score=0.8,
            entity_names=("WeBank", "XGBoost", "AlloBank"),
        )
        self.assertEqual(
            atomic_candidate_text(atomic.memory.content),
            "TYPE: ATOMIC_MEMORY\nCONTENT: The user built an XGBoost "
            "scorecard with KS 0.271.",
        )
        self.assertEqual(
            episode_candidate_text(
                episode.title,
                episode.summary,
                episode.entity_names,
            ),
            "TYPE: EPISODE\n"
            "TITLE: WeBank Internship\n"
            "SUMMARY: Summer internship involving consumer lending "
            "analytics, risk dashboards, an AI assistant, XGBoost modeling, "
            "and recovery analysis.\n"
            "ENTITIES: WeBank, XGBoost, AlloBank.",
        )

        class _Reranker:
            def rerank_mixed(self, query, candidates):
                return {
                    candidate_id: 0.2 if candidate_id.startswith("atomic:") else 0.9
                    for candidate_id, _text in candidates
                }

        ranked = coselect_hierarchical_candidates(
            "How did the internship go?",
            (atomic,),
            (episode,),
            _Reranker(),
        )
        self.assertEqual([item.kind for item in ranked], ["EPISODE", "ATOMIC_MEMORY"])
        self.assertEqual(ranked[0].score, 0.9)
        self.assertEqual(ranked[1].score, 0.2)

    def test_rerank_tie_prefers_an_episode_then_embedding_similarity(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            RetrievedEpisode,
            rank_hierarchical_candidates,
        )
        from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory

        moment = datetime(2026, 6, 1, tzinfo=UTC)

        def atomic(memory_id: str, semantic_score: float) -> RetrievedMemory:
            return RetrievedMemory(
                memory=Memory(
                    id=memory_id,
                    content=memory_id,
                    memory_type=MemoryType.FACT,
                    role="user",
                    source="test",
                    session_id="session",
                    importance=0.5,
                    created_at=moment,
                    updated_at=moment,
                    metadata={},
                ),
                semantic_score=semantic_score,
                recency_score=0.0,
                retrieval_score=0.0,
                final_score=0.0,
            )

        def episode(episode_id: str, similarity: float) -> RetrievedEpisode:
            return RetrievedEpisode(
                episode_id=episode_id,
                title=episode_id,
                summary=episode_id,
                member_count=2,
                similarity_score=similarity,
            )

        ranked = rank_hierarchical_candidates(
            (
                atomic("atomic-high", 0.9),
                atomic("atomic-low", 0.2),
                atomic("atomic-best", 0.1),
            ),
            (
                episode("episode-low", 0.1),
                episode("episode-high", 0.8),
            ),
            {
                "atomic:atomic-high": 0.5,
                "atomic:atomic-low": 0.5,
                "atomic:atomic-best": 0.9,
                "episode:episode-low": 0.5,
                "episode:episode-high": 0.5,
            },
        )
        self.assertEqual(
            [
                item.episode.episode_id if item.episode else item.memory.memory.id
                for item in ranked
            ],
            [
                "atomic-best",
                "episode-high",
                "episode-low",
                "atomic-high",
                "atomic-low",
            ],
        )

    def test_group_selector_keeps_episode_or_atomics_then_top_five(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            RankedCandidate,
            RetrievedEpisode,
            select_hierarchical_return,
        )
        from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory

        moment = datetime(2026, 6, 1, tzinfo=UTC)

        def atomic(memory_id: str, score: float) -> RankedCandidate:
            memory = RetrievedMemory(
                memory=Memory(
                    id=memory_id,
                    content=f"content {memory_id}",
                    memory_type=MemoryType.FACT,
                    role="user",
                    source="test",
                    session_id="session",
                    importance=0.5,
                    created_at=moment,
                    updated_at=moment,
                    metadata={},
                ),
                semantic_score=score,
                recency_score=score,
                retrieval_score=score,
                final_score=score,
                rerank_score=score,
            )
            return RankedCandidate(
                kind="ATOMIC_MEMORY",
                score=score,
                text=memory.memory.content,
                memory=memory,
            )

        def episode(episode_id: str, score: float) -> RankedCandidate:
            stored = RetrievedEpisode(
                episode_id=episode_id,
                title=episode_id,
                summary=f"summary {episode_id}",
                member_count=2,
                similarity_score=score,
                entity_names=("WeBank",),
                rerank_score=score,
            )
            return RankedCandidate(
                kind="EPISODE",
                score=score,
                text=stored.summary,
                episode=stored,
            )

        class _Selector:
            def __init__(self, choice: str) -> None:
                self.choice = choice
                self.calls = 0

            def choose(self, query, title, summary, entity_names, atomic_contents):
                self.calls += 1
                self.seen_contents = list(atomic_contents)
                self.seen_summary = summary
                return self.choice

        ranked = (
            episode("ep-high", 0.95),
            atomic("inside", 0.90),
            atomic("outside", 0.80),
            episode("ep-low", 0.10),
        )
        members = {
            "ep-high": {"inside", "held-back", "held-back-two"},
            "ep-low": set(),
        }
        episode_only = _Selector("episode_only")
        kept = select_hierarchical_return(
            "query",
            ranked,
            members,
            episode_only,
            limit=5,
        )
        self.assertEqual(episode_only.calls, 1)
        self.assertEqual(episode_only.seen_contents, ["content inside"])
        self.assertEqual(episode_only.seen_summary, "summary ep-high")
        self.assertEqual(
            [item.kind + ":" + (
                item.episode.episode_id if item.episode else item.memory.memory.id
            ) for item in kept],
            ["EPISODE:ep-high", "ATOMIC_MEMORY:outside", "EPISODE:ep-low"],
        )

        atom_only = _Selector("atom_only")
        kept_atoms = select_hierarchical_return(
            "query",
            ranked,
            members,
            atom_only,
            limit=5,
        )
        self.assertEqual(
            [item.kind for item in kept_atoms],
            ["ATOMIC_MEMORY", "ATOMIC_MEMORY", "EPISODE"],
        )
        self.assertEqual(kept_atoms[0].memory.memory.id, "inside")
        self.assertEqual(kept_atoms[2].episode.episode_id, "ep-low")

        many = tuple(atomic(f"m{index}", 1 - index / 100) for index in range(6))
        cut = select_hierarchical_return("query", many, {}, None, limit=5)
        self.assertEqual(len(cut), 5)
        self.assertEqual(cut[0].memory.memory.id, "m0")
        self.assertEqual(cut[4].memory.memory.id, "m4")
        self.assertEqual(cut.statuses, ("selected",) * 5 + ("",))

        perfect = (
            episode("perfect-episode", 1.0),
            *[atomic(f"perfect-{index}", 1.0) for index in range(5)],
            atomic("lower", 0.4),
        )
        kept_perfect = select_hierarchical_return(
            "query",
            perfect,
            {},
            None,
            limit=5,
        )
        self.assertEqual(len(kept_perfect), 6)
        self.assertTrue(all(item.score == 1.0 for item in kept_perfect))
        self.assertEqual(kept_perfect[0].episode.episode_id, "perfect-episode")
        self.assertEqual(kept_perfect.statuses, ("selected",) * 6 + ("",))

    def test_eighty_percent_coverage_keeps_episode_before_selector(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            RankedCandidate,
            RetrievedEpisode,
            select_hierarchical_return,
        )
        from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory

        moment = datetime(2026, 6, 1, tzinfo=UTC)

        def atomic(memory_id: str, score: float) -> RankedCandidate:
            memory = RetrievedMemory(
                memory=Memory(
                    id=memory_id,
                    content=f"content {memory_id}",
                    memory_type=MemoryType.FACT,
                    role="user",
                    source="test",
                    session_id="session",
                    importance=0.5,
                    created_at=moment,
                    updated_at=moment,
                    metadata={},
                ),
                semantic_score=score,
                recency_score=score,
                retrieval_score=score,
                final_score=score,
                rerank_score=score,
            )
            return RankedCandidate(
                kind="ATOMIC_MEMORY",
                score=score,
                text=memory.memory.content,
                memory=memory,
            )

        def episode(episode_id: str, score: float) -> RankedCandidate:
            stored = RetrievedEpisode(
                episode_id=episode_id,
                title=episode_id,
                summary=f"summary {episode_id}",
                member_count=5,
                similarity_score=score,
                entity_names=("WeBank",),
                rerank_score=score,
            )
            return RankedCandidate(
                kind="EPISODE",
                score=score,
                text=stored.summary,
                episode=stored,
            )

        class _Selector:
            def __init__(self) -> None:
                self.calls = 0
                self.titles: list[str] = []

            def choose(self, query, title, summary, entity_names, atomic_contents):
                self.calls += 1
                self.titles.append(title)
                return "atom_only"

        ranked = (
            episode("covered", 0.95),
            atomic("m1", 0.90),
            atomic("m2", 0.85),
            atomic("m3", 0.80),
            atomic("m4", 0.75),
            atomic("outside", 0.70),
            episode("partial", 0.60),
            atomic("only-partial", 0.50),
        )
        members = {
            "covered": {"m1", "m2", "m3", "m4", "not-retrieved"},
            "partial": {"only-partial", "away-one", "away-two"},
        }
        selector = _Selector()
        kept = select_hierarchical_return(
            "query",
            ranked,
            members,
            selector,
            limit=5,
        )
        self.assertEqual(selector.calls, 1)
        self.assertEqual(selector.titles, ["partial"])
        self.assertEqual(
            [
                item.episode.episode_id if item.episode else item.memory.memory.id
                for item in kept
            ],
            ["covered", "outside", "only-partial"],
        )
        self.assertEqual(
            kept.statuses,
            (
                "selected",
                "removed",
                "removed",
                "removed",
                "removed",
                "selected",
                "removed",
                "selected",
            ),
        )

    def test_prompt_orders_atomics_then_episodes_by_rerank(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            RankedCandidate,
            RetrievedEpisode,
            format_hierarchical_context,
        )
        from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory

        moment = datetime(2026, 6, 1, tzinfo=UTC)

        def atomic(memory_id: str, content: str, memory_type: MemoryType, score: float) -> RankedCandidate:
            memory = RetrievedMemory(
                memory=Memory(
                    id=memory_id,
                    content=content,
                    memory_type=memory_type,
                    role="user",
                    source="test",
                    session_id="session",
                    importance=0.5,
                    created_at=moment,
                    updated_at=moment,
                    metadata={},
                ),
                semantic_score=score,
                recency_score=score,
                retrieval_score=score,
                final_score=score,
                rerank_score=score,
            )
            return RankedCandidate(
                kind="ATOMIC_MEMORY",
                score=score,
                text=content,
                memory=memory,
            )

        def episode(
            episode_id: str,
            score: float,
            contents: tuple[str, ...],
        ) -> RankedCandidate:
            stored = RetrievedEpisode(
                episode_id=episode_id,
                title=f"Title {episode_id}",
                summary=f"Summary {episode_id}",
                member_count=len(contents),
                similarity_score=score,
                entity_names=("WeBank", "XGBoost"),
                rerank_score=score,
                member_contents=contents,
            )
            return RankedCandidate(
                kind="EPISODE",
                score=score,
                text=stored.summary,
                episode=stored,
            )

        text = format_hierarchical_context(
            (
                episode("low", 0.20, ("Older memory.", "Newer memory.")),
                atomic("pref", "User wants short answers.", MemoryType.PREFERENCE, 0.910),
                episode("high", 0.95, ("The user interned at WeBank.",)),
                atomic("fact", "The user's name is Beta.", MemoryType.FACT, 0.400),
            )
        )
        self.assertEqual(
            text,
            "\n".join(
                [
                    "fact",
                    "- [user; relevance=0.400; ] The user's name is Beta.",
                    "preference",
                    "- [user; relevance=0.910; ] User wants short answers.",
                    "",
                    "Title high (0.950)",
                    "summary: Summary high",
                    "entities: WeBank, XGBoost",
                    "- The user interned at WeBank.",
                    "",
                    "Title low (0.200)",
                    "summary: Summary low",
                    "entities: WeBank, XGBoost",
                    "- Older memory.",
                    "- Newer memory.",
                ]
            ),
        )


    def test_choice_marks_keep_returned_candidates(self) -> None:
        from ai_agent_memory_infrastructure.memory_hierarchical import (
            RankedCandidate,
            RetrievedEpisode,
            select_hierarchical_return,
        )
        from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory

        moment = datetime(2026, 6, 1, tzinfo=UTC)

        def atomic(memory_id: str, score: float) -> RankedCandidate:
            memory = RetrievedMemory(
                memory=Memory(
                    id=memory_id,
                    content=memory_id,
                    memory_type=MemoryType.FACT,
                    role="user",
                    source="test",
                    session_id="session",
                    importance=0.5,
                    created_at=moment,
                    updated_at=moment,
                    metadata={},
                ),
                semantic_score=score,
                recency_score=0.0,
                retrieval_score=score,
                final_score=score,
            )
            return RankedCandidate(
                kind="ATOMIC_MEMORY",
                score=score,
                text=memory_id,
                memory=memory,
            )

        episode = RankedCandidate(
            kind="EPISODE",
            score=0.4,
            text="episode",
            episode=RetrievedEpisode(
                episode_id="ep",
                title="ep",
                summary="ep",
                member_count=2,
                similarity_score=0.4,
            ),
        )
        kept = select_hierarchical_return(
            "query",
            (atomic("first", 0.9), atomic("second", 0.8), episode),
            {},
            None,
            limit=2,
        )
        self.assertEqual(kept.statuses, ("selected", "selected", ""))


class RuntimeTimingTest(unittest.TestCase):
    def test_trace_records_nested_sections_and_collapses_repeats(self) -> None:
        from ai_agent_memory_infrastructure.memory_timing import (
            collapse_runtime_spans,
            runtime_span,
            runtime_trace,
        )

        with runtime_trace("Chat", "Message") as trace:
            with runtime_span("Read"):
                with runtime_span("Embed"):
                    pass
                with runtime_span("Embed"):
                    pass
        self.assertEqual(trace.path, "Chat")
        self.assertEqual(trace.trigger, "Message")
        self.assertEqual(trace.spans[0].name, "Read")
        self.assertEqual(
            [span.name for span in trace.spans[0].children],
            ["Embed", "Embed"],
        )
        collapsed = collapse_runtime_spans(trace.spans)
        self.assertEqual(collapsed[0].children[0].name, "Embed × 2")
        self.assertEqual(
            collapsed[0].children[0].seconds,
            trace.spans[0].children[0].seconds + trace.spans[0].children[1].seconds,
        )

    def test_span_outside_a_trace_does_nothing(self) -> None:
        from ai_agent_memory_infrastructure.memory_timing import runtime_span

        with runtime_span("Read"):
            pass


class GraphRagTest(unittest.TestCase):
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
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings

    def _memory(
        self,
        memory_id: str,
        content: str,
        *,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        expired: bool = False,
        embedding: list[float] | None = None,
    ) -> None:
        moment = datetime(2026, 6, 1, tzinfo=UTC)
        self.store.import_record(
            memory_id=memory_id,
            content=content,
            memory_type="fact",
            role="user",
            source="test",
            session_id="graph-rag",
            importance=0.5,
            created_at=moment,
            updated_at=moment,
            metadata={},
            embedding=embedding or [0.0, 0.0, 1.0],
            status=status,
            expired=expired,
        )

    def _edge(
        self,
        memory_id: str,
        source: str,
        relation: str,
        target: str,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self.store.replace_memory_graph(
            memory_id,
            f"key-{memory_id}-{source}-{target}",
            [(source, relation, target)],
            embeddings,
        )

    def test_explicit_seed_walks_one_hop_and_reranks(self) -> None:
        from ai_agent_memory_infrastructure.memory_graph_rag import (
            GraphRagSettings,
            run_graph_rag,
        )

        self._memory("m1", "The assistant uses Dify.")
        self._memory("m2", "The assistant uses DuckDB.")
        self._memory("m3", "Lived in Boston.")
        self._memory("m4", "Wrote notes.")
        self._memory("m5", "Kept a journal.")
        self._memory(
            "old",
            "The assistant uses a retired tool.",
            expired=True,
        )
        self.store.replace_memory_graph(
            "m1",
            "key-m1",
            [
                ("AI Assistant", "uses", "Dify"),
                ("The user", "built", "AI Assistant"),
            ],
        )
        self.store.replace_memory_graph(
            "m2",
            "key-m2",
            [
                ("AI Assistant", "uses", "DuckDB"),
                ("The user", "built", "AI Assistant"),
            ],
        )
        self._edge("m3", "The user", "lived in", "Boston")
        self._edge("m4", "The user", "wrote", "Notes")
        self._edge("m5", "The user", "kept", "Journal")
        self._edge("old", "AI Assistant", "uses", "Retired")
        before_entities = self.store.graph_entity_count()
        before_aliases = len(self.store.graph_rag_aliases())

        class _Reranker:
            def rerank(self, query, candidates):
                return {
                    candidate.memory.id: (
                        0.9 if "DuckDB" in candidate.memory.content else 0.2
                    )
                    for candidate in candidates
                }

        class _Answerer:
            def __init__(self) -> None:
                self.calls = 0

            def answer(self, query, paths, memories):
                self.calls += 1
                self.memory_ids = [memory.memory_id for memory in memories]
                return "You used DuckDB."

        answerer = _Answerer()
        result = run_graph_rag(
            self.store,
            "What tools did I use for my AI assistant?",
            reranker=_Reranker(),
            answerer=answerer,
            settings=GraphRagSettings(
                final_top_k=1,
                max_edges_per_seed=30,
            ),
        )
        self.assertEqual([seed.name for seed in result.seed_entities], ["AI Assistant"])
        self.assertNotIn("Retired", [path.target_name for path in result.paths])
        self.assertTrue(all(path.hop == 1 for path in result.paths))
        self.assertEqual(result.retrieved_memories[0].memory_id, "m2")
        self.assertEqual(result.retrieved_memories[0].rerank_score, 0.9)
        self.assertEqual(answerer.calls, 1)
        self.assertEqual(answerer.memory_ids, ["m2"])
        self.assertIn("m1", [path.memory_id for path in result.paths])
        self.assertEqual(self.store.graph_entity_count(), before_entities)
        self.assertEqual(len(self.store.graph_rag_aliases()), before_aliases)
        record = result.as_record()
        self.assertEqual(record["answer"], "You used DuckDB.")
        self.assertEqual(record["paths"][0]["hop"], 1)

    def test_broad_semantic_match_is_not_expanded(self) -> None:
        from ai_agent_memory_infrastructure.memory_graph_rag import run_graph_rag

        for index in range(5):
            self._memory(f"m{index}", f"python note {index}", embedding=[1.0, 0.0, 0.0])
            self._edge(
                f"m{index}",
                "AI Assistant",
                "uses",
                f"Tool {index}",
                {"ai assistant": [1.0, 0.0, 0.0]},
            )

        class _Answerer:
            def __init__(self) -> None:
                self.calls = 0

            def answer(self, query, paths, memories):
                self.calls += 1
                return "should not run"

        answerer = _Answerer()
        result = run_graph_rag(
            self.store,
            "python notes",
            reranker=object(),
            answerer=answerer,
        )
        self.assertEqual(result.seed_entities, ())
        self.assertEqual(answerer.calls, 0)
        self.assertIn("No stored entity", result.answer)

    def test_explicit_broad_seed_caps_edges(self) -> None:
        from ai_agent_memory_infrastructure.memory_graph_rag import (
            GraphRagSettings,
            run_graph_rag,
        )

        for index in range(4):
            self._memory(f"m{index}", f"Fact {index}.")
            self._edge(f"m{index}", "The user", "did", f"Task {index}")

        class _Reranker:
            def rerank(self, query, candidates):
                return {candidate.memory.id: 0.5 for candidate in candidates}

        class _Answerer:
            def answer(self, query, paths, memories):
                return "Some tasks."

        result = run_graph_rag(
            self.store,
            "What did the user do?",
            reranker=_Reranker(),
            answerer=_Answerer(),
            settings=GraphRagSettings(max_edges_per_seed=2),
        )
        self.assertEqual([seed.name for seed in result.seed_entities], ["The user"])
        self.assertTrue(result.seed_entities[0].broad)
        self.assertEqual(result.paths, ())
        self.assertIn("insufficient", result.answer)

    def test_walk_stops_at_three_hops_and_skips_a_weak_edge(self) -> None:
        from ai_agent_memory_infrastructure.memory_graph_rag import (
            GraphRagSettings,
            run_graph_rag,
        )

        self._memory("m1", "Widget uses Part.", embedding=[0.0, 0.0, 1.0])
        self._memory("m2", "Part contains Piece.", embedding=[0.0, 0.0, 1.0])
        self._memory("m3", "Piece links End.", embedding=[0.0, 0.0, 1.0])
        self._memory("m4", "End goes Beyond.", embedding=[0.0, 0.0, 1.0])
        self._memory("m5", "python stray from Widget.", embedding=[1.0, 0.0, 0.0])
        self._edge("m1", "Widget", "uses", "Part")
        self._edge("m2", "Part", "contains", "Piece")
        self._edge("m3", "Piece", "links", "End")
        self._edge("m4", "End", "goes to", "Beyond")
        self._edge("m5", "Widget", "ignores", "Other")

        class _Reranker:
            def rerank(self, query, candidates):
                return {candidate.memory.id: 0.5 for candidate in candidates}

        class _Answerer:
            def answer(self, query, paths, memories):
                return "A chain."

        result = run_graph_rag(
            self.store,
            "Tell me about Widget",
            reranker=_Reranker(),
            answerer=_Answerer(),
            settings=GraphRagSettings(
                final_top_k=5,
                frontier_cap=5,
                min_hop_similarity=0.50,
            ),
        )
        self.assertEqual(
            {path.memory_id for path in result.paths},
            {"m1", "m2", "m3"},
        )
        self.assertEqual({path.hop for path in result.paths}, {1, 2, 3})
        self.assertNotIn("m4", [path.memory_id for path in result.paths])
        self.assertNotIn("m5", [path.memory_id for path in result.paths])


if __name__ == "__main__":
    unittest.main()
