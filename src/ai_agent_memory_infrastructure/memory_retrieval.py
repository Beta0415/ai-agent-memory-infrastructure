from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from ai_agent_memory_infrastructure.memory_store import Memory, MemoryMatch

CANDIDATE_RETRIEVAL_LIMIT = 30
WEIGHTED_CANDIDATE_LIMIT = 10
DEFAULT_RESULT_LIMIT = 4
SEMANTIC_WEIGHT = 0.6
RECENCY_WEIGHT = 0.2
IMPORTANCE_WEIGHT = 0.2
# Recency falls by half every 30 days.
RECENCY_HALF_LIFE_DAYS = 30
RECENCY_LAMBDA_PER_DAY = math.log(2) / RECENCY_HALF_LIFE_DAYS


@dataclass(frozen=True)
class RetrievedMemory:
    memory: Memory
    semantic_score: float
    recency_score: float
    retrieval_score: float
    final_score: float
    rerank_score: float | None = None


class MemoryReranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedMemory],
    ) -> Mapping[str, float]: ...


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _unit_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def calculate_recency_score(
    created_at: datetime,
    *,
    now: datetime,
    lambda_per_day: float = RECENCY_LAMBDA_PER_DAY,
) -> float:
    age_days = max(
        (_as_utc(now) - _as_utc(created_at)).total_seconds() / 86_400,
        0.0,
    )
    return math.exp(-lambda_per_day * age_days)


def score_retrieved_memory(
    match: MemoryMatch,
    *,
    now: datetime,
) -> RetrievedMemory:
    semantic_score = _unit_score(match.similarity_score)
    recency_score = calculate_recency_score(
        match.memory.created_at,
        now=now,
    )
    importance = _unit_score(match.memory.importance)
    retrieval_score = (
        SEMANTIC_WEIGHT * semantic_score
        + RECENCY_WEIGHT * recency_score
        + IMPORTANCE_WEIGHT * importance
    )
    return RetrievedMemory(
        memory=match.memory,
        semantic_score=semantic_score,
        recency_score=recency_score,
        retrieval_score=retrieval_score,
        final_score=retrieval_score,
    )


def rank_by_retrieval_score(
    matches: Sequence[MemoryMatch],
    *,
    limit: int,
    now: datetime,
) -> tuple[RetrievedMemory, ...]:
    scored = [
        score_retrieved_memory(match, now=now) for match in matches
    ]
    scored.sort(
        key=lambda item: (
            item.retrieval_score,
            item.semantic_score,
            _as_utc(item.memory.created_at),
        ),
        reverse=True,
    )
    return tuple(scored[:limit])


def apply_rerank_scores(
    candidates: Sequence[RetrievedMemory],
    scores: Mapping[str, float],
) -> tuple[RetrievedMemory, ...]:
    rescored = []
    for candidate in candidates:
        if candidate.memory.id not in scores:
            raise RuntimeError(
                "The memory reranker omitted a weighted candidate."
            )
        rerank_score = _unit_score(float(scores[candidate.memory.id]))
        rescored.append(
            replace(
                candidate,
                final_score=rerank_score,
                rerank_score=rerank_score,
            )
        )
    rescored.sort(
        key=lambda item: (
            item.final_score,
            item.retrieval_score,
            item.semantic_score,
            _as_utc(item.memory.created_at),
        ),
        reverse=True,
    )
    return tuple(rescored)
