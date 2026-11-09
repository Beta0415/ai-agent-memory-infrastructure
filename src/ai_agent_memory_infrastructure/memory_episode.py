from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import combinations
from typing import Protocol
from uuid import uuid4

from openai import OpenAI

from ai_agent_memory_infrastructure.memory_timing import runtime_span, timed
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_entity_similarity import (
    embedding_cosine_similarity,
)

DEFAULT_PREVALENCE_BASELINE = 0.20
DEFAULT_PREVALENCE_DECAY_SCALE = 100.0
PREVALENCE_HEADROOM = 0.30
DEFAULT_EPISODE_MEAN_COHESION = 0.0
DEFAULT_EPISODE_MINIMUM_COHESION = 0.0
DEFAULT_EPISODE_OVERLAP = 0.50
DEFAULT_EPISODE_CONTAINMENT = 0.80
DEFAULT_EPISODE_NOVELTY = 0.20
EPISODE_MINIMUM_MEMORIES = 2


class EpisodeTitleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=1000)


MEMORY_EPISODE_TITLE_INSTRUCTIONS = """
You name one episode and write a brief summary for later retrieval.

The episode is a set of atomic memories joined by the listed entities.
Do not use a conversation.

Return:
- title: at most 12 words naming the situation those memories share.
- summary: one or two sentences.
  Summarize only facts explicitly supported by the member atomic memories.
  Do not add inferred or external facts.
  Keep the summary concise and retrieval-oriented.
""".strip()


@dataclass(frozen=True)
class EpisodeSettings:
    prevalence_baseline: float = DEFAULT_PREVALENCE_BASELINE
    prevalence_decay_scale: float = DEFAULT_PREVALENCE_DECAY_SCALE
    mean_cohesion: float = DEFAULT_EPISODE_MEAN_COHESION
    minimum_cohesion: float = DEFAULT_EPISODE_MINIMUM_COHESION
    overlap_threshold: float = DEFAULT_EPISODE_OVERLAP
    containment_threshold: float = DEFAULT_EPISODE_CONTAINMENT
    novelty_threshold: float = DEFAULT_EPISODE_NOVELTY

    def __post_init__(self) -> None:
        if not 0.0 <= self.prevalence_baseline <= 1.0:
            raise ValueError("Prevalence baseline must be between 0 and 1.")
        if self.prevalence_decay_scale < 1:
            raise ValueError("Prevalence decay scale must be at least 1.")
        if not 0.0 <= self.mean_cohesion <= 1.0:
            raise ValueError("Mean cohesion must be between 0 and 1.")
        if not 0.0 <= self.minimum_cohesion <= 1.0:
            raise ValueError("Minimum cohesion must be between 0 and 1.")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("Entity overlap must be between 0 and 1.")
        if not 0.0 <= self.containment_threshold <= 1.0:
            raise ValueError("Episode containment must be between 0 and 1.")
        if not 0.0 <= self.novelty_threshold <= 1.0:
            raise ValueError("Episode novelty must be between 0 and 1.")


@dataclass(frozen=True)
class EpisodeDraft:
    entity_ids: tuple[str, ...]
    entity_names: tuple[str, ...]
    memory_ids: tuple[str, ...]
    memory_contents: tuple[str, ...]
    cohesion: float | None = None
    minimum_cohesion: float | None = None
    cohesion_stddev: float | None = None


@dataclass(frozen=True)
class EpisodeBuildResult:
    episode_count: int
    titled_count: int


@dataclass(frozen=True)
class EpisodeWrite:
    episode_id: str
    entity_key: str
    title: str
    summary: str
    source_key: str
    created_at: datetime
    updated_at: datetime
    entity_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    cohesion: float | None = None
    minimum_cohesion: float | None = None
    cohesion_stddev: float | None = None
    embedding: list[float] | None = None


class EpisodeTitler(Protocol):
    def title(
        self,
        entity_names: Sequence[str],
        memory_contents: Sequence[str],
    ) -> tuple[str, str]: ...


class OpenAIEpisodeTitler:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.6-luna",
    ) -> None:
        self._model = model
        self._openai = OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @timed("Episode Titler")
    def title(
        self,
        entity_names: Sequence[str],
        memory_contents: Sequence[str],
    ) -> tuple[str, str]:
        names = [name.strip() for name in entity_names if name.strip()]
        memories = [content.strip() for content in memory_contents if content.strip()]
        if not names or not memories:
            raise ValueError("An episode title needs entities and memories.")
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_EPISODE_TITLE_INSTRUCTIONS,
            input=json.dumps(
                {"entities": names, "memories": memories},
                ensure_ascii=False,
            ),
            text_format=EpisodeTitleOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError("The episode titler returned no structured output.")
        title = response.output_parsed.title.strip()
        summary = response.output_parsed.summary.strip()
        if not title or not summary:
            raise RuntimeError("The episode titler returned an empty title or summary.")
        return title, summary


def form_episodes(
    mentions: Sequence[tuple[str, str, str, str]],
    total_memories: int,
    settings: EpisodeSettings | None = None,
) -> tuple[EpisodeDraft, ...]:
    """Group memories by entity prevalence. No entity name is special.

    Prevalence is the number of active memories containing an entity divided
    by the total number of active memories, N. The specific bound is
    (baseline + 0.30) / (1 + N/scale). Both the baseline and the 0.30 decay.
    A specific entity is in at least 2 memories and its prevalence is at or
    below that bound. Those memories are one episode. Two entities are one
    episode when at least 2 memories contain both, the less common entity is
    specific, and at least half of that entity's memories also contain the
    other one.
    """
    active = settings or EpisodeSettings()
    if total_memories < 1:
        return ()
    threshold = prevalence_threshold(
        total_memories,
        baseline=active.prevalence_baseline,
        decay_scale=active.prevalence_decay_scale,
    )
    contents: dict[str, str] = {}
    names: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    for memory_id, content, entity_id, name in mentions:
        if not memory_id or not entity_id or not name.strip():
            continue
        contents[memory_id] = content
        names.setdefault(entity_id, name.strip())
        members.setdefault(entity_id, set()).add(memory_id)
    degrees = {
        entity_id: len(memory_ids) for entity_id, memory_ids in members.items()
    }
    drafts: list[EpisodeDraft] = []
    for entity_id, degree in degrees.items():
        if _is_specific(degree, total_memories, threshold):
            drafts.append(_draft((entity_id,), members, names, contents))
    candidates = [
        entity_id
        for entity_id, degree in degrees.items()
        if degree >= EPISODE_MINIMUM_MEMORIES
    ]
    for left_id, right_id in combinations(sorted(candidates), 2):
        lesser = min(degrees[left_id], degrees[right_id])
        if not _is_specific(lesser, total_memories, threshold):
            continue
        shared = members[left_id] & members[right_id]
        if len(shared) < EPISODE_MINIMUM_MEMORIES:
            continue
        overlap = len(shared) / lesser
        if overlap < active.overlap_threshold:
            continue
        drafts.append(
            _draft((left_id, right_id), members, names, contents, shared)
        )
    return tuple(
        sorted(
            drafts,
            key=lambda draft: (
                len(draft.entity_ids),
                draft.entity_names,
                draft.entity_ids,
            ),
        )
    )


def prevalence_threshold(
    total_memories: int,
    *,
    baseline: float = DEFAULT_PREVALENCE_BASELINE,
    decay_scale: float = DEFAULT_PREVALENCE_DECAY_SCALE,
) -> float:
    """Return (baseline + 0.30) / (1 + N/scale). Both parts decay toward 0."""
    if total_memories < 0:
        raise ValueError("Memory count must be at least 0.")
    if decay_scale < 1:
        raise ValueError("Prevalence decay scale must be at least 1.")
    bound = (baseline + PREVALENCE_HEADROOM) / (
        1.0 + total_memories / decay_scale
    )
    return min(bound, 1.0)


def _is_specific(degree: int, total_memories: int, prevalence_limit: float) -> bool:
    if degree < EPISODE_MINIMUM_MEMORIES:
        return False
    return degree / total_memories <= prevalence_limit


def embedding_cohesion(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, float, float, tuple[float, ...]]:
    """Return cohesion, its minimum, the standard deviation, and each cosine.

    Cohesion is the average cosine similarity between the memory embeddings
    and their centroid. The standard deviation is of those similarities.
    """
    if not vectors:
        raise ValueError("Embedding cohesion needs at least one vector.")
    width = len(vectors[0])
    if width < 1:
        raise ValueError("Embedding cohesion needs a non-empty vector.")
    centroid = [0.0] * width
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("Embedding vectors must have the same length.")
        for index, value in enumerate(vector):
            centroid[index] += value
    count = len(vectors)
    centroid = [value / count for value in centroid]
    similarities = tuple(
        embedding_cosine_similarity(vector, centroid) for vector in vectors
    )
    cohesion, minimum, deviation = cohesion_summary(similarities)
    return cohesion, minimum, deviation, similarities


def cohesion_summary(
    similarities: Sequence[float],
) -> tuple[float, float, float]:
    """Return the average, minimum, and population standard deviation."""
    if not similarities:
        raise ValueError("Cohesion summary needs at least one similarity.")
    count = len(similarities)
    cohesion = sum(similarities) / count
    minimum = min(similarities)
    variance = sum((score - cohesion) ** 2 for score in similarities) / count
    return cohesion, minimum, math.sqrt(variance)


def apply_single_entity_cohesion(
    drafts: Sequence[EpisodeDraft],
    embeddings: Mapping[str, Sequence[float]],
    *,
    minimum_cohesion: float,
    mean_cohesion: float,
) -> tuple[EpisodeDraft, ...]:
    """Drop a loose memory, then keep a one-entity episode that stays cohesive.

    A two-entity episode is unchanged. A memory stays when its cosine to the
    centroid is at or above minimum_cohesion. The episode remains when the
    memories that stay number at least two, their mean cohesion is at or
    above mean_cohesion, and their minimum cohesion is still at or above
    minimum_cohesion.
    """
    kept: list[EpisodeDraft] = []
    for draft in drafts:
        if len(draft.entity_ids) != 1:
            kept.append(draft)
            continue
        _cohesion, _minimum, _deviation, similarities = embedding_cohesion(
            [embeddings[memory_id] for memory_id in draft.memory_ids]
        )
        selected = tuple(
            memory_id
            for memory_id, score in zip(
                draft.memory_ids,
                similarities,
                strict=True,
            )
            if score >= minimum_cohesion
        )
        if len(selected) < EPISODE_MINIMUM_MEMORIES:
            continue
        contents = dict(
            zip(draft.memory_ids, draft.memory_contents, strict=True)
        )
        cohesion, minimum, deviation, _scores = embedding_cohesion(
            [embeddings[memory_id] for memory_id in selected]
        )
        if cohesion < mean_cohesion or minimum < minimum_cohesion:
            continue
        kept.append(
            replace(
                draft,
                memory_ids=selected,
                memory_contents=tuple(contents[memory_id] for memory_id in selected),
                cohesion=cohesion,
                minimum_cohesion=minimum,
                cohesion_stddev=deviation,
            )
        )
    return tuple(kept)


def episode_set_scores(
    candidate: set[str],
    other: set[str],
) -> tuple[float, float, float]:
    """Return Jaccard, containment of the candidate in the other, and novelty.

    Jaccard is |A ∩ B| / |A ∪ B|. Containment is |A ∩ B| / |B| for candidate
    B inside other A. Novelty is |B − A| / |B|.
    """
    shared = candidate & other
    union = candidate | other
    jaccard = len(shared) / len(union) if union else 1.0
    if not candidate:
        return jaccard, 1.0, 0.0
    containment = len(shared) / len(candidate)
    novelty = len(candidate - other) / len(candidate)
    return jaccard, containment, novelty


def drop_redundant_episodes(
    drafts: Sequence[EpisodeDraft],
    *,
    containment_threshold: float,
    novelty_threshold: float,
) -> tuple[EpisodeDraft, ...]:
    """Remove an episode that is mostly already inside another episode.

    B is removed when its containment in A is at or above the containment
    threshold and its novelty given A is at or below the novelty threshold.
    If each episode meets that test for the other, the larger memory set
    stays. A tie keeps the episode with fewer entities.
    """
    kept: list[EpisodeDraft] = []
    for candidate in drafts:
        candidate_ids = set(candidate.memory_ids)
        redundant = False
        for other in drafts:
            if other is candidate:
                continue
            other_ids = set(other.memory_ids)
            if not _is_redundant_episode(
                candidate_ids,
                other_ids,
                containment_threshold,
                novelty_threshold,
            ):
                continue
            if _is_redundant_episode(
                other_ids,
                candidate_ids,
                containment_threshold,
                novelty_threshold,
            ) and _redundancy_key(other) > _redundancy_key(candidate):
                continue
            redundant = True
            break
        if not redundant:
            kept.append(candidate)
    return tuple(kept)


def _is_redundant_episode(
    candidate: set[str],
    other: set[str],
    containment_threshold: float,
    novelty_threshold: float,
) -> bool:
    _jaccard, containment, novelty = episode_set_scores(candidate, other)
    return (
        containment >= containment_threshold
        and novelty <= novelty_threshold
    )


def _redundancy_key(draft: EpisodeDraft) -> tuple[int, int, tuple[str, ...]]:
    return (-len(draft.memory_ids), len(draft.entity_ids), draft.entity_ids)


def _episode_embeddings(
    store: object,
    drafts: Sequence[EpisodeDraft],
) -> dict[str, Sequence[float]]:
    memory_ids = list(
        {
            memory_id
            for draft in drafts
            if len(draft.entity_ids) == 1
            for memory_id in draft.memory_ids
        }
    )
    if not memory_ids:
        return {}
    return store.memory_embeddings(memory_ids)


def episode_retrieval_text(
    title: str,
    summary: str,
    entity_names: Sequence[str],
) -> str:
    """Return the text embedded for one episode: title, summary, entity names."""
    return "\n".join(
        (
            title.strip(),
            summary.strip(),
            ", ".join(name.strip() for name in entity_names if name.strip()),
        )
    )


def episode_entity_key(entity_ids: Sequence[str]) -> str:
    return "\n".join(sorted(entity_ids))


def episode_source_key(
    entity_ids: Sequence[str],
    memory_ids: Sequence[str],
) -> str:
    payload = episode_entity_key(entity_ids) + "\n\n" + "\n".join(sorted(memory_ids))
    return hashlib.sha256(payload.encode()).hexdigest()


class MemoryEpisode:
    def __init__(self, store: object, titler: EpisodeTitler | None = None) -> None:
        self._store = store
        self._titler = titler

    def build(
        self,
        settings: EpisodeSettings | None = None,
    ) -> EpisodeBuildResult:
        active = settings or EpisodeSettings()
        with runtime_span("Form Episodes"):
            drafts = form_episodes(
                self._store.graph_memory_mentions(),
                self._store.active_memory_count(),
                active,
            )
        with runtime_span("Cohesion"):
            drafts = apply_single_entity_cohesion(
                drafts,
                _episode_embeddings(self._store, drafts),
                minimum_cohesion=active.minimum_cohesion,
                mean_cohesion=active.mean_cohesion,
            )
        with runtime_span("Drop Redundant"):
            drafts = drop_redundant_episodes(
                drafts,
                containment_threshold=active.containment_threshold,
                novelty_threshold=active.novelty_threshold,
            )
        existing = {
            entity_key: (
                episode_id,
                title,
                summary,
                source_key,
                created_at,
                updated_at,
                embedding,
            )
            for (
                episode_id,
                entity_key,
                title,
                summary,
                source_key,
                created_at,
                updated_at,
                embedding,
            ) in self._store.memory_episode_states()
        }
        now = datetime.now(UTC)
        titled_count = 0
        writes: list[EpisodeWrite] = []
        with runtime_span("Title And Embed"):
            for draft in drafts:
                entity_key = episode_entity_key(draft.entity_ids)
                source_key = episode_source_key(draft.entity_ids, draft.memory_ids)
                current = existing.get(entity_key)
                if current is not None and current[3] == source_key:
                    (
                        episode_id,
                        title,
                        summary,
                        _source_key,
                        created_at,
                        updated_at,
                        embedding,
                    ) = current
                    if embedding is None:
                        embedding = self._embed_retrieval(
                            title,
                            summary,
                            draft.entity_names,
                        )
                    writes.append(
                        EpisodeWrite(
                            episode_id=episode_id,
                            entity_key=entity_key,
                            title=title,
                            summary=summary,
                            source_key=source_key,
                            created_at=created_at,
                            updated_at=updated_at,
                            entity_ids=draft.entity_ids,
                            memory_ids=draft.memory_ids,
                            cohesion=draft.cohesion,
                            minimum_cohesion=draft.minimum_cohesion,
                            cohesion_stddev=draft.cohesion_stddev,
                            embedding=embedding,
                        )
                    )
                    continue
                title, summary = self._describe_episode(draft)
                titled_count += 1
                writes.append(
                    EpisodeWrite(
                        episode_id=current[0] if current is not None else uuid4().hex,
                        entity_key=entity_key,
                        title=title,
                        summary=summary,
                        source_key=source_key,
                        created_at=current[4] if current is not None else now,
                        updated_at=now,
                        entity_ids=draft.entity_ids,
                        memory_ids=draft.memory_ids,
                        cohesion=draft.cohesion,
                        minimum_cohesion=draft.minimum_cohesion,
                        cohesion_stddev=draft.cohesion_stddev,
                        embedding=self._embed_retrieval(
                            title,
                            summary,
                            draft.entity_names,
                        ),
                    )
                )
        with runtime_span("Store Episodes"):
            self._store.sync_memory_episodes(writes)
        return EpisodeBuildResult(
            episode_count=len(writes),
            titled_count=titled_count,
        )

    def _describe_episode(self, draft: EpisodeDraft) -> tuple[str, str]:
        if self._titler is None:
            raise RuntimeError("A new episode needs an episode titler.")
        title, summary = self._titler.title(
            draft.entity_names,
            draft.memory_contents,
        )
        title = title.strip()
        summary = summary.strip()
        if not title or not summary:
            raise RuntimeError(
                "The episode titler returned an empty title or summary."
            )
        return title, summary

    def _embed_retrieval(
        self,
        title: str,
        summary: str,
        entity_names: Sequence[str],
    ) -> list[float]:
        vectors = self._store.embed_texts(
            [episode_retrieval_text(title, summary, entity_names)]
        )
        return vectors[0]

def _draft(
    entity_ids: tuple[str, ...],
    members: dict[str, set[str]],
    names: dict[str, str],
    contents: dict[str, str],
    memory_ids: set[str] | None = None,
) -> EpisodeDraft:
    ordered_ids = tuple(sorted(entity_ids))
    if memory_ids is None:
        selected = set(members[ordered_ids[0]])
    else:
        selected = set(memory_ids)
    ordered_memories = tuple(sorted(selected))
    ordered_names = tuple(
        sorted((names[entity_id] for entity_id in ordered_ids), key=str.casefold)
    )
    return EpisodeDraft(
        entity_ids=ordered_ids,
        entity_names=ordered_names,
        memory_ids=ordered_memories,
        memory_contents=tuple(contents[memory_id] for memory_id in ordered_memories),
    )
