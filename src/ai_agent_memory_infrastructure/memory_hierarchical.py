from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ai_agent_memory_infrastructure.memory_api import ReadPathSettings
from ai_agent_memory_infrastructure.memory_context import format_context_memories
from ai_agent_memory_infrastructure.memory_retrieval import (
    RetrievedMemory,
    rank_by_retrieval_score,
)
from ai_agent_memory_infrastructure.memory_store import MemoryFilters
from ai_agent_memory_infrastructure.memory_timing import runtime_span


def hierarchical_atomic_retrieval(
    store: object,
    query: str,
    *,
    settings: ReadPathSettings,
    min_similarity: float | None = None,
    filters: MemoryFilters | None = None,
    now: datetime | None = None,
) -> tuple[tuple[RetrievedMemory, ...], tuple[RetrievedMemory, ...]]:
    """Return the weighted top memories and the scored semantic pool.

    This is the atomic half of hierarchical retrieval: the top semantic
    candidates, then the weighted score, then the weighted pool. Episode
    retrieval and co-selection are not part of this step.
    """
    with runtime_span("Semantic Search"):
        semantic_matches = store.memory_search(
            query,
            limit=settings.candidate_limit,
            min_similarity=min_similarity,
            filters=filters,
        )
    scored_at = now or datetime.now(UTC)
    with runtime_span("Weighted Score"):
        store.memory_mark_accessed(
            [match.memory.id for match in semantic_matches],
            accessed_at=scored_at,
        )
        scored_candidates = rank_by_retrieval_score(
            semantic_matches,
            limit=max(len(semantic_matches), 1),
            now=scored_at,
            semantic_weight=settings.semantic_weight,
            recency_weight=settings.recency_weight,
            importance_weight=settings.importance_weight,
            lambda_per_day=settings.recency_lambda_per_day,
        )
        weighted = scored_candidates[: settings.weighted_limit]
    return weighted, scored_candidates


EPISODE_RETRIEVAL_LIMIT = 5


@dataclass(frozen=True)
class RetrievedEpisode:
    episode_id: str
    title: str
    summary: str
    member_count: int
    similarity_score: float
    entity_names: tuple[str, ...] = ()
    rerank_score: float | None = None
    member_contents: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedCandidate:
    kind: str
    score: float
    text: str
    memory: RetrievedMemory | None = None
    episode: RetrievedEpisode | None = None


def atomic_candidate_text(content: str) -> str:
    return f"TYPE: ATOMIC_MEMORY\nCONTENT: {content}"


def episode_candidate_text(
    title: str,
    summary: str,
    entity_names: Sequence[str],
) -> str:
    names = ", ".join(name.strip() for name in entity_names if name.strip())
    entities = f"{names}." if names else ""
    return (
        "TYPE: EPISODE\n"
        f"TITLE: {title}\n"
        f"SUMMARY: {summary}\n"
        f"ENTITIES: {entities}"
    )


def atomic_candidate_id(memory_id: str) -> str:
    return f"atomic:{memory_id}"


def episode_candidate_id(episode_id: str) -> str:
    return f"episode:{episode_id}"


def rank_hierarchical_candidates(
    atomics: Sequence[RetrievedMemory],
    episodes: Sequence[RetrievedEpisode],
    scores: Mapping[str, float],
) -> tuple[RankedCandidate, ...]:
    """Order atomic memories and episodes by one reranker score.

    A tied rerank score keeps an episode ahead of an atomic memory. Two
    episodes, or two atomic memories, then break the tie by embedding
    similarity.
    """
    ranked: list[RankedCandidate] = []
    for memory in atomics:
        candidate_id = atomic_candidate_id(memory.memory.id)
        score = scores[candidate_id]
        ranked.append(
            RankedCandidate(
                kind="ATOMIC_MEMORY",
                score=score,
                text=atomic_candidate_text(memory.memory.content),
                memory=replace(memory, rerank_score=score, final_score=score),
            )
        )
    for episode in episodes:
        candidate_id = episode_candidate_id(episode.episode_id)
        score = scores[candidate_id]
        ranked.append(
            RankedCandidate(
                kind="EPISODE",
                score=score,
                text=episode_candidate_text(
                    episode.title,
                    episode.summary,
                    episode.entity_names,
                ),
                episode=replace(episode, rerank_score=score),
            )
        )
    ranked.sort(
        key=lambda item: (
            item.score,
            item.episode is not None,
            (
                item.episode.similarity_score
                if item.episode is not None
                else item.memory.semantic_score
                if item.memory is not None
                else 0.0
            ),
        ),
        reverse=True,
    )
    return tuple(ranked)


def hierarchical_episode_retrieval(
    store: object,
    query: str,
    *,
    limit: int = EPISODE_RETRIEVAL_LIMIT,
) -> tuple[RetrievedEpisode, ...]:
    """Return the nearest episodes by cosine similarity of their embeddings.

    The embedding is the one stored for the title, summary, and entity names.
    This search does not use the atomic candidate pool.
    """
    rows = store.episode_search(query, limit=limit)
    names = store.episode_entity_names([row[0] for row in rows])
    return tuple(
        RetrievedEpisode(
            episode_id=episode_id,
            title=title,
            summary=summary,
            member_count=member_count,
            similarity_score=similarity,
            entity_names=names.get(episode_id, ()),
        )
        for episode_id, title, summary, member_count, similarity in rows
    )


def coselect_hierarchical_candidates(
    query: str,
    atomics: Sequence[RetrievedMemory],
    episodes: Sequence[RetrievedEpisode],
    reranker: object,
) -> tuple[RankedCandidate, ...]:
    """Score atomic memories and episodes on one scale and rank them together."""
    if not atomics and not episodes:
        return ()
    if reranker is None or not hasattr(reranker, "rerank_mixed"):
        raise RuntimeError("Hierarchical retrieval needs a reranker.")
    payload = [
        (
            atomic_candidate_id(memory.memory.id),
            atomic_candidate_text(memory.memory.content),
        )
        for memory in atomics
    ]
    payload.extend(
        (
            episode_candidate_id(episode.episode_id),
            episode_candidate_text(
                episode.title,
                episode.summary,
                episode.entity_names,
            ),
        )
        for episode in episodes
    )
    scores = reranker.rerank_mixed(query, payload)
    return rank_hierarchical_candidates(atomics, episodes, scores)


HIERARCHICAL_RETURN_LIMIT = 5
HIERARCHICAL_PERFECT_SCORE = 1.0
HIERARCHICAL_EPISODE_COVERAGE = 0.80


def hierarchical_return_count(
    ranked: Sequence[RankedCandidate],
    *,
    limit: int = HIERARCHICAL_RETURN_LIMIT,
) -> int:
    """Keep the top results, or every perfect score when that set is larger."""
    perfect = sum(1 for item in ranked if item.score >= HIERARCHICAL_PERFECT_SCORE)
    if perfect > limit:
        return perfect
    return limit


def episode_coverage_memory_ids(
    ranked: Sequence[RankedCandidate],
    members_by_episode: Mapping[str, set[str]],
    *,
    coverage: float = HIERARCHICAL_EPISODE_COVERAGE,
) -> set[str]:
    """Return atomic ids that already cover a retrieved episode.

    Coverage is the share of that episode's member memories present in the
    ranked atomic pool. At or above the coverage bound, those atomic memories
    stand in for the episode and are removed before the group selector.
    """
    atomic_ids = {
        item.memory.memory.id
        for item in ranked
        if item.memory is not None
    }
    covered: set[str] = set()
    for item in ranked:
        if item.episode is None:
            continue
        member_ids = members_by_episode.get(item.episode.episode_id, set())
        if not member_ids:
            continue
        overlap = atomic_ids & member_ids
        if len(overlap) / len(member_ids) >= coverage:
            covered.update(overlap)
    return covered


def select_hierarchical_return(
    query: str,
    ranked: Sequence[RankedCandidate],
    members_by_episode: Mapping[str, set[str]],
    selector: object | None,
    *,
    limit: int = HIERARCHICAL_RETURN_LIMIT,
) -> HierarchicalSelection:
    """Drop overlap inside each episode group, then keep the top results.

    When the ranked atomic memories include at least 80% of an episode's
    members, those atomic memories are removed and the episode stays. The
    selector is not asked about that group. A remaining group is one episode
    plus the ranked atomic memories that still belong to it. The selector
    then chooses episode_only or atom_only. An atomic memory is removed only
    when every remaining episode that contains it chooses episode_only. An
    episode is removed when its group chooses atom_only. The survivors stay
    in rerank order. The first 5 are returned, unless more than 5 of them
    scored 1.0, in which case every 1.0 survivor is returned.
    """
    covered_memory_ids = episode_coverage_memory_ids(
        ranked,
        members_by_episode,
    )
    pending = [
        item
        for item in ranked
        if not (
            item.memory is not None
            and item.memory.memory.id in covered_memory_ids
        )
    ]
    atomics = [item for item in pending if item.memory is not None]
    episodes = [item for item in pending if item.episode is not None]
    episode_only_memory_ids: set[str] = set()
    atom_only_memory_ids: set[str] = set()
    atom_only_episode_ids: set[str] = set()
    for episode_item in episodes:
        episode = episode_item.episode
        assert episode is not None
        member_ids = members_by_episode.get(episode.episode_id, set())
        overlapping = [
            item
            for item in atomics
            if item.memory is not None and item.memory.memory.id in member_ids
        ]
        if not overlapping:
            continue
        if selector is None or not hasattr(selector, "choose"):
            raise RuntimeError(
                "An episode that overlaps retrieved atomic memories needs "
                "an episode group selector."
            )
        choice = selector.choose(
            query,
            episode.title,
            episode.summary,
            episode.entity_names,
            [item.memory.memory.content for item in overlapping if item.memory],
        )
        if choice not in ("episode_only", "atom_only"):
            raise RuntimeError(
                "The episode group selector returned an unknown choice: "
                f"{choice}"
            )
        memory_ids = {
            item.memory.memory.id
            for item in overlapping
            if item.memory is not None
        }
        if choice == "episode_only":
            episode_only_memory_ids.update(memory_ids)
        else:
            atom_only_episode_ids.add(episode.episode_id)
            atom_only_memory_ids.update(memory_ids)
    remove_memory_ids = covered_memory_ids | (
        episode_only_memory_ids - atom_only_memory_ids
    )
    kept = [
        item
        for item in ranked
        if not (
            item.memory is not None
            and item.memory.memory.id in remove_memory_ids
        )
        and not (
            item.episode is not None
            and item.episode.episode_id in atom_only_episode_ids
        )
    ]
    returned = tuple(kept[: hierarchical_return_count(kept, limit=limit)])
    returned_keys = {_rank_key(item) for item in returned}
    removed_keys = {
        _rank_key(item)
        for item in ranked
        if (
            item.memory is not None
            and item.memory.memory.id in remove_memory_ids
        )
        or (
            item.episode is not None
            and item.episode.episode_id in atom_only_episode_ids
        )
    }
    statuses = tuple(
        "selected"
        if _rank_key(item) in returned_keys
        else "removed"
        if _rank_key(item) in removed_keys
        else ""
        for item in ranked
    )
    return HierarchicalSelection(returned, statuses)


class HierarchicalSelection(tuple[RankedCandidate, ...]):
    """The returned candidates, plus one status for every reranked candidate."""

    statuses: tuple[str, ...]

    def __new__(
        cls,
        kept: Sequence[RankedCandidate],
        statuses: Sequence[str],
    ) -> HierarchicalSelection:
        selection = super().__new__(cls, kept)
        selection.statuses = tuple(statuses)
        return selection


def _rank_key(item: RankedCandidate) -> tuple[str, str]:
    if item.memory is not None:
        return ("atomic", item.memory.memory.id)
    if item.episode is not None:
        return ("episode", item.episode.episode_id)
    return ("", "")


def format_hierarchical_context(
    ranked: Sequence[RankedCandidate],
) -> str:
    """Render atomic memories first, then episodes by rerank score.

    Atomic memories use the same type grouping and line format as the atomic
    read path. Each episode follows as its title and rerank score, summary,
    entity names, and member memories.
    """
    atomics = [item.memory for item in ranked if item.memory is not None]
    episodes = [item.episode for item in ranked if item.episode is not None]
    episodes.sort(
        key=lambda episode: (
            -(episode.rerank_score if episode.rerank_score is not None else 0.0),
            episode.episode_id,
        )
    )
    sections = [format_context_memories(atomics)]
    for episode in episodes:
        score = (
            episode.rerank_score
            if episode.rerank_score is not None
            else episode.similarity_score
        )
        names = ", ".join(
            name.strip() for name in episode.entity_names if name.strip()
        )
        lines = [
            f"{' '.join(episode.title.split())} ({score:.3f})",
            f"summary: {' '.join(episode.summary.split())}",
            f"entities: {names}",
        ]
        lines.extend(
            f"- {' '.join(content.split())}"
            for content in episode.member_contents
            if content.strip()
        )
        sections.append("\n".join(lines))
    return "\n\n".join(section for section in sections if section.strip())
