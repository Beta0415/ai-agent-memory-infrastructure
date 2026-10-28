from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def entity_name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    """Return 1 when the names match, and 0 when they share no characters."""
    folded_left = entity_name_key(left)
    folded_right = entity_name_key(right)
    if folded_left == folded_right:
        return 1.0
    longest = max(len(folded_left), len(folded_right))
    if longest == 0:
        return 1.0
    return 1.0 - _levenshtein_distance(folded_left, folded_right) / longest


def jaccard_token_similarity(left: str, right: str) -> float:
    """Return the share of whitespace-separated words the two names have in common."""
    left_tokens = set(entity_name_key(left).split())
    right_tokens = set(entity_name_key(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def embedding_cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return the cosine similarity of two embedding vectors."""
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero length.")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


@dataclass(frozen=True)
class EntityComparison:
    mention_name: str
    existing_entity_id: str
    existing_name: str
    levenshtein: float
    jaccard: float
    embedding: float


LEVENSHTEIN_SIMILARITY_THRESHOLD = 0.80
JACCARD_SIMILARITY_THRESHOLD = 0.50
EMBEDDING_SIMILARITY_THRESHOLD = 0.80


@dataclass(frozen=True)
class EntityResolutionSettings:
    levenshtein_threshold: float = LEVENSHTEIN_SIMILARITY_THRESHOLD
    jaccard_threshold: float = JACCARD_SIMILARITY_THRESHOLD
    embedding_threshold: float = EMBEDDING_SIMILARITY_THRESHOLD

    def __post_init__(self) -> None:
        for name, value in (
            ("Levenshtein similarity", self.levenshtein_threshold),
            ("Jaccard similarity", self.jaccard_threshold),
            ("Embedding similarity", self.embedding_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


def passes_entity_threshold(
    comparison: EntityComparison,
    settings: EntityResolutionSettings | None = None,
) -> bool:
    """Return whether any score is high enough for the semantic judge."""
    active = settings or EntityResolutionSettings()
    return (
        comparison.levenshtein >= active.levenshtein_threshold
        or comparison.jaccard >= active.jaccard_threshold
        or comparison.embedding >= active.embedding_threshold
    )


def score_new_mentions(
    mentions: Sequence[str],
    existing: Sequence[tuple[str, str]],
    vectors_by_key: Mapping[str, Sequence[float]],
) -> tuple[EntityComparison, ...]:
    """Score each new mention against every entity already stored.

    An exact name, ignoring case and extra spaces, is omitted. Mentions are
    not compared with one another, because they are not stored yet.
    """
    existing_keys = {entity_name_key(name) for _entity_id, name in existing}
    comparisons: list[EntityComparison] = []
    seen_mentions: set[str] = set()
    for mention in mentions:
        mention_key = entity_name_key(mention)
        if (
            not mention_key
            or mention_key in existing_keys
            or mention_key in seen_mentions
        ):
            continue
        seen_mentions.add(mention_key)
        mention_vector = vectors_by_key[mention_key]
        for entity_id, name in existing:
            existing_key = entity_name_key(name)
            comparisons.append(
                EntityComparison(
                    mention_name=mention.strip(),
                    existing_entity_id=entity_id,
                    existing_name=name,
                    levenshtein=normalized_levenshtein_similarity(mention, name),
                    jaccard=jaccard_token_similarity(mention, name),
                    embedding=embedding_cosine_similarity(
                        mention_vector,
                        vectors_by_key[existing_key],
                    ),
                )
            )
    return tuple(comparisons)


def _levenshtein_distance(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]
