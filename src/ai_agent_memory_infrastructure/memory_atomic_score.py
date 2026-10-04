from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.tokens import Doc, Token


CLAUSE_DEPS = {"ROOT", "advcl", "ccomp", "xcomp", "relcl", "acl", "parataxis"}
PREDICATIVE_DEPS = {"acomp", "attr", "oprd"}
CONTRAST_MARKS = {"while", "whereas", "although", "though", "whilst"}
ATOMIC_SCORE_REVIEW_THRESHOLD = 0.2


def needs_atomic_review(
    score: float,
    threshold: float = ATOMIC_SCORE_REVIEW_THRESHOLD,
) -> bool:
    return score >= threshold


@dataclass(frozen=True)
class AtomicScoreWeights:
    predicate: float = 0.35
    coordinated_items: float = 0.20
    clause: float = 0.15
    coordination: float = 0.10
    value: float = 0.10
    tokens: float = 0.10

    def __post_init__(self) -> None:
        weights = (
            self.predicate,
            self.coordinated_items,
            self.clause,
            self.coordination,
            self.value,
            self.tokens,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Atomic score weights must be zero or greater."
            )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Atomic score weights must sum to 1.")


@dataclass(frozen=True)
class AtomicScore:
    score: float
    predicate: float
    coordinated_items: float
    clause: float
    coordination: float
    value: float
    tokens: float
    predicate_count: int
    coordinated_item_count: int
    clause_count: int
    coordination_count: int
    value_count: int
    token_count: int


def score_atomic_memory(
    content: str,
    weights: AtomicScoreWeights | None = None,
) -> AtomicScore:
    text = content.strip()
    if not text:
        return component_score(0, 0, 0, 0, 0, 0, weights=weights)
    document = _nlp()(text)
    return component_score(
        predicate_count(document),
        coordinated_item_count(document),
        clause_count(document),
        coordination_count(document),
        value_count(document),
        token_count(document),
        weights=weights,
    )


def component_score(
    predicate_count: int,
    coordinated_item_count: int,
    clause_count: int,
    coordination_count: int,
    value_count: int,
    token_count: int,
    weights: AtomicScoreWeights | None = None,
) -> AtomicScore:
    active = weights or AtomicScoreWeights()
    predicate = min(max(predicate_count - 1, 0), 1)
    coordinated_items = min(max(coordinated_item_count - 1, 0) / 2, 1)
    clause = min(max(clause_count - 1, 0) / 2, 1)
    coordination = min(max(coordination_count, 0), 1)
    value = min(max(value_count - 1, 0) / 2, 1)
    tokens = min(max((token_count - 15) / 30, 0), 1)
    score = (
        active.predicate * predicate
        + active.coordinated_items * coordinated_items
        + active.clause * clause
        + active.coordination * coordination
        + active.value * value
        + active.tokens * tokens
    )
    return AtomicScore(
        score=score,
        predicate=predicate,
        coordinated_items=coordinated_items,
        clause=clause,
        coordination=coordination,
        value=value,
        tokens=tokens,
        predicate_count=predicate_count,
        coordinated_item_count=coordinated_item_count,
        clause_count=clause_count,
        coordination_count=coordination_count,
        value_count=value_count,
        token_count=token_count,
    )


def predicate_count(document: Doc) -> int:
    count = 0
    for token in document:
        if not _is_clause_head(token):
            continue
        if token.pos_ != "AUX":
            count += 1
            continue
        properties = [
            child
            for child in token.children
            if child.dep_ in PREDICATIVE_DEPS
        ]
        if not properties:
            count += 1
            continue
        for prop in properties:
            count += 1 + len(_conj_chain(prop))
    return count


def clause_count(document: Doc) -> int:
    return sum(1 for token in document if _is_clause_head(token))


def coordinated_item_count(document: Doc) -> int:
    largest = 0
    seen: set[int] = set()
    for token in document:
        if token.dep_ != "conj":
            continue
        head = token
        while head.dep_ == "conj":
            head = head.head
        if head.i in seen:
            continue
        seen.add(head.i)
        members = [head, *_conj_chain(head)]
        if all(member.pos_ in {"VERB", "AUX"} for member in members):
            continue
        largest = max(largest, len(members))
    return largest


def coordination_count(document: Doc) -> int:
    count = 0
    for token in document:
        lemma = token.lemma_.lower()
        if token.pos_ == "CCONJ":
            count += 1
        elif lemma in CONTRAST_MARKS and token.dep_ == "mark":
            count += 1
    return count


def value_count(document: Doc) -> int:
    return sum(
        1 for token in document if token.like_num or token.pos_ == "NUM"
    )


def token_count(document: Doc) -> int:
    return sum(
        1
        for token in document
        if not token.is_punct and not token.is_space
    )


def _is_clause_head(token: Token) -> bool:
    if token.pos_ not in {"VERB", "AUX"}:
        return False
    if token.dep_ in CLAUSE_DEPS:
        return True
    return token.dep_ == "conj" and token.head.pos_ in {"VERB", "AUX"}


def _conj_chain(token: Token) -> list[Token]:
    found: list[Token] = []
    seen: set[int] = set()
    stack = [token]
    while stack:
        current = stack.pop()
        for child in current.children:
            if child.dep_ == "conj" and child.i not in seen:
                seen.add(child.i)
                found.append(child)
                stack.append(child)
    return found


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm")
