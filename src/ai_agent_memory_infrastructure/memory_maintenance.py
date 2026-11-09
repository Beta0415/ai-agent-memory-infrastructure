from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ai_agent_memory_infrastructure.memory_retrieval import (
    RECENCY_HALF_LIFE_DAYS,
    calculate_recency_score,
)
from ai_agent_memory_infrastructure.memory_candidate_relation import (
    WRITE_RELATED_LIMIT,
    similarity_above,
)
from ai_agent_memory_infrastructure.memory_timing import runtime_span
from ai_agent_memory_infrastructure.memory_dedup import (
    DEDUP_VERDICTS,
    MemoryDedupClassifier,
)
from ai_agent_memory_infrastructure.memory_store import (
    RELATED_MEMORY_STATUSES,
    Memory,
    MemoryFilters,
)
from ai_agent_memory_infrastructure.postgres_memory_store import (
    PostgresMemoryStore,
)


FORGET_INACTIVITY_DAYS = 60
DEFAULT_TOKEN_CAPACITY = 10_000
MAINTENANCE_DEDUP_SIMILARITY = 0.9


@dataclass(frozen=True)
class MaintenanceSettings:
    token_capacity: int = DEFAULT_TOKEN_CAPACITY
    similarity_threshold: float = MAINTENANCE_DEDUP_SIMILARITY
    importance_weight: float = 0.5
    recency_weight: float = 0.5
    half_life_days: float = RECENCY_HALF_LIFE_DAYS

    def __post_init__(self) -> None:
        if self.token_capacity < 1:
            raise ValueError("Memory token capacity must be at least 1.")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "Maintenance similarity must be between 0 and 1."
            )
        weights = (self.importance_weight, self.recency_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Dedup weights must be zero or greater."
            )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Dedup weights must sum to 1.")
        if self.half_life_days <= 0:
            raise ValueError(
                "Dedup half-life must be greater than zero."
            )


@dataclass(frozen=True)
class MaintenanceMember:
    memory_id: str
    content: str
    status: str
    expired: bool
    importance: float
    keep_score: float
    action: str


@dataclass(frozen=True)
class MaintenanceGroup:
    verdict: str
    members: tuple[MaintenanceMember, ...]


@dataclass(frozen=True)
class MaintenanceExpired:
    memory_id: str
    content: str
    status: str


@dataclass(frozen=True)
class MaintenanceResult:
    expired_count: int
    removed_count: int = 0
    groups: tuple[MaintenanceGroup, ...] = ()
    expired_memories: tuple[MaintenanceExpired, ...] = ()


class MemoryMaintenance:
    def __init__(
        self,
        store: PostgresMemoryStore,
        dedup_classifier: MemoryDedupClassifier | None = None,
    ) -> None:
        self._store = store
        self._dedup_classifier = dedup_classifier

    def forget(
        self,
        *,
        now: datetime,
        inactivity_days: int = FORGET_INACTIVITY_DAYS,
    ) -> list[Memory]:
        return self._store.expire_inactive_memories(
            now=now,
            inactivity_days=inactivity_days,
        )

    def deduplicate(
        self,
        *,
        now: datetime,
        settings: MaintenanceSettings | None = None,
    ) -> tuple[int, tuple[MaintenanceGroup, ...]]:
        if self._dedup_classifier is None:
            return 0, ()
        active = settings or MaintenanceSettings()
        groups = _duplicate_groups(
            self._store,
            active.similarity_threshold,
        )
        removed = 0
        reports: list[MaintenanceGroup] = []
        for group in groups:
            keeper = max(
                group,
                key=lambda memory: _keep_key(memory, now, active),
            )
            others = [memory for memory in group if memory.id != keeper.id]
            verdict = self._dedup_classifier.classify(keeper, others)
            label = verdict.strip().lower()
            if label not in DEDUP_VERDICTS:
                raise RuntimeError(
                    "The dedup classifier returned an unknown verdict: "
                    f"{verdict}"
                )
            if label == "equivalent":
                self._store.reinforce_memories(
                    [keeper.id],
                    reinforced_at=now,
                )
                for memory in others:
                    self._store.move_memory_to_bin(memory.id)
                    removed += 1
            reports.append(
                MaintenanceGroup(
                    verdict=label,
                    members=tuple(
                        _member_report(
                            memory,
                            now,
                            active,
                            _member_action(label, memory.id, keeper.id),
                        )
                        for memory in (keeper, *others)
                    ),
                )
            )
        return removed, tuple(reports)

    def run(
        self,
        *,
        now: datetime,
        settings: MaintenanceSettings | None = None,
    ) -> MaintenanceResult:
        with runtime_span("Deduplicate"):
            removed_count, groups = self.deduplicate(now=now, settings=settings)
        with runtime_span("Forget"):
            expired = self.forget(now=now)
        return MaintenanceResult(
            expired_count=len(expired),
            removed_count=removed_count,
            groups=groups,
            expired_memories=tuple(
                MaintenanceExpired(
                    memory_id=memory.id,
                    content=memory.content,
                    status=memory.status.value,
                )
                for memory in expired
            ),
        )

    def over_capacity(self, token_capacity: int) -> bool:
        return self._store.total_token_count() > token_capacity


def _member_action(verdict: str, memory_id: str, keeper_id: str) -> str:
    if verdict != "equivalent":
        return "unchanged"
    if memory_id == keeper_id:
        return "kept"
    return "removed"


def _member_report(
    memory: Memory,
    now: datetime,
    settings: MaintenanceSettings,
    action: str,
) -> MaintenanceMember:
    return MaintenanceMember(
        memory_id=memory.id,
        content=memory.content,
        status=memory.status.value,
        expired=memory.expired,
        importance=memory.importance,
        keep_score=_keep_key(memory, now, settings)[0],
        action=action,
    )


def _keep_key(
    memory: Memory,
    now: datetime,
    settings: MaintenanceSettings,
) -> tuple[float, datetime, datetime]:
    reinforced_at = memory.reinforced_at or memory.created_at
    recency = calculate_recency_score(
        reinforced_at,
        now=now,
        lambda_per_day=math.log(2) / settings.half_life_days,
    )
    score = (
        settings.importance_weight * memory.importance
        + settings.recency_weight * recency
    )
    return (score, reinforced_at, memory.created_at)


def _duplicate_groups(
    store: PostgresMemoryStore,
    similarity_threshold: float,
) -> list[list[Memory]]:
    filters = MemoryFilters(
        statuses=RELATED_MEMORY_STATUSES,
        expired=None,
    )
    pool = store.memory_ids(filters)
    if len(pool) < 2:
        return []
    linked = store.linked_memory_pairs()
    parent = {memory_id: memory_id for memory_id in pool}

    def find(memory_id: str) -> str:
        while parent[memory_id] != memory_id:
            parent[memory_id] = parent[parent[memory_id]]
            memory_id = parent[memory_id]
        return memory_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for memory_id in pool:
        neighbors = store.similar_memories(
            memory_id,
            limit=WRITE_RELATED_LIMIT,
            min_similarity=similarity_threshold,
            filters=filters,
        )
        for neighbor in neighbors:
            neighbor_id = neighbor.memory.id
            if neighbor_id not in parent:
                continue
            pair = (
                (memory_id, neighbor_id)
                if memory_id < neighbor_id
                else (neighbor_id, memory_id)
            )
            if pair in linked:
                continue
            if not similarity_above(
                neighbor.similarity_score,
                similarity_threshold,
            ):
                continue
            union(memory_id, neighbor_id)

    grouped: dict[str, list[str]] = {}
    for memory_id in pool:
        grouped.setdefault(find(memory_id), []).append(memory_id)
    groups: list[list[Memory]] = []
    for memory_ids in grouped.values():
        if len(memory_ids) < 2:
            continue
        members = []
        for memory_id in memory_ids:
            memory = store.memory_get(memory_id)
            if memory is not None:
                members.append(memory)
        if len(members) >= 2:
            groups.append(members)
    return groups
