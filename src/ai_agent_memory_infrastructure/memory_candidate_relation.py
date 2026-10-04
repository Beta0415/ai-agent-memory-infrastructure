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


class CandidateRelationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: CandidateRelation


MEMORY_CANDIDATE_RELATION_INSTRUCTIONS = """
You compare one candidate memory with existing memories that already passed
a vector similarity check. The existing memories are ordered from closest
to farthest.

Return one label for the candidate:
- equivalent: it states the same facts as the existing memories. Wording
  may differ. It does not add, remove, update, or deny a fact.
- refinement: it keeps an existing claim and adds detail, precision, or a
  qualifier. It does not replace the claim or say the old claim was wrong.
- supersession: it replaces an existing memory because the situation
  changed. The older memory was reasonable when stored, and the candidate
  is the new state.
- correction: it fixes an existing memory that was wrong. The older
  statement is a mistake, not a past state that later changed.
- contradiction: it conflicts with an existing memory, and the texts do
  not show whether the candidate is a correction, a newer state, or an
  unresolved disagreement.
- related_distinct: it is about the same subject but states a separate
  fact that should stay beside the existing memories. It does not restate,
  refine, replace, correct, or contradict them.

If one label does not fit every existing memory, use the label that fits
the closest memory. Do not rewrite, merge, or choose a stored action.
Return only the label.
""".strip()


class MemoryCandidateRelationClassifier(Protocol):
    def classify(
        self,
        candidate_content: str,
        related: Sequence[MemoryMatch],
    ) -> str: ...


class OpenAIMemoryCandidateRelationClassifier:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5-mini",
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
    ) -> str:
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
        return response.output_parsed.relation
