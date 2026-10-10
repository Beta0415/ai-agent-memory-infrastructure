from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from ai_agent_memory_infrastructure.memory_store import (
    MemoryLinkRelation,
    MemoryMatch,
    MemoryStatus,
)


WRITE_RELATED_SIMILARITY = 0.6
WRITE_RELATED_LIMIT = 5


def similarity_above(
    score: float,
    threshold: float = WRITE_RELATED_SIMILARITY,
) -> bool:
    return round(score, 6) > threshold


CandidateRelation = Literal[
    "equivalent",
    "refinement",
    "supersession",
    "correction",
    "contradiction",
    "related_distinct",
]

CANDIDATE_RELATIONS = (
    "equivalent",
    "refinement",
    "supersession",
    "correction",
    "contradiction",
    "related_distinct",
)


@dataclass(frozen=True)
class CandidateRelationEffect:
    new_status: MemoryStatus
    previous_status: MemoryStatus | None = None
    link: MemoryLinkRelation | None = None


CANDIDATE_RELATION_EFFECTS = {
    "refinement": CandidateRelationEffect(
        MemoryStatus.ACTIVE,
        MemoryStatus.SUPERSEDED,
        MemoryLinkRelation.REFINES,
    ),
    "supersession": CandidateRelationEffect(
        MemoryStatus.ACTIVE,
        MemoryStatus.SUPERSEDED,
        MemoryLinkRelation.SUPERSEDES,
    ),
    "correction": CandidateRelationEffect(
        MemoryStatus.ACTIVE,
        MemoryStatus.CORRECTED,
        MemoryLinkRelation.CORRECTS,
    ),
    "contradiction": CandidateRelationEffect(
        MemoryStatus.CONFLICTED,
        MemoryStatus.CONFLICTED,
        MemoryLinkRelation.CONFLICTS,
    ),
    "related_distinct": CandidateRelationEffect(MemoryStatus.ACTIVE),
}


class CandidateMemoryRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    relation: CandidateRelation


class CandidateRelationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relations: list[CandidateMemoryRelation]


MEMORY_CANDIDATE_RELATION_INSTRUCTIONS = """
You compare one candidate memory with each existing memory that already
passed a vector similarity check. Judge each existing memory on its own.

Return one label for every existing memory id:
- equivalent: the candidate states the same fact as this memory. Wording
  may differ. It does not add, remove, update, or deny a fact.
- refinement: the candidate keeps this memory's claim and adds detail,
  precision, or a qualifier. It does not replace the claim or say this
  memory was wrong. A shorter sentence that drops a school, employer,
  place, time, or project is not a refinement.
- supersession: the candidate replaces this memory because the situation
  changed. This memory was reasonable when stored, and the candidate is
  the new state.
- correction: the candidate fixes this memory because this memory was
  wrong. This memory is a mistake, not a past state that later changed.
- contradiction: the candidate conflicts with this memory, and the texts
  do not show whether the candidate is a correction, a newer state, or an
  unresolved disagreement.
- related_distinct: the candidate is a separate fact and should stay
  beside this memory. Another item in the same list is related_distinct.
  A different course, tool, or task is related_distinct. It does not
  restate, refine, replace, correct, or contradict this memory.

Do not reuse one label for the whole set. Do not rewrite, merge, or
choose a stored action.
""".strip()


class MemoryCandidateRelationClassifier(Protocol):
    def classify(
        self,
        candidate_content: str,
        related: Sequence[MemoryMatch],
    ) -> Sequence[tuple[str, str]]: ...


class OpenAIMemoryCandidateRelationClassifier:
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

    def classify(
        self,
        candidate_content: str,
        related: Sequence[MemoryMatch],
    ) -> Sequence[tuple[str, str]]:
        if not candidate_content.strip():
            raise ValueError("Candidate memory content cannot be empty.")
        if not related:
            raise ValueError(
                "Candidate relation classification needs a related memory."
            )
        payload = {
            "candidate": candidate_content,
            "related_memories": [
                {
                    "memory_id": match.memory.id,
                    "content": match.memory.content,
                    "similarity": match.similarity_score,
                }
                for match in related
            ],
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_CANDIDATE_RELATION_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=CandidateRelationOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The candidate relation classifier returned no "
                "structured output."
            )
        return [
            (item.memory_id, item.relation)
            for item in response.output_parsed.relations
        ]
