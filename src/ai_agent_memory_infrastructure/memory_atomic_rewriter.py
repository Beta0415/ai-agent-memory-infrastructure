from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_timing import timed
from ai_agent_memory_infrastructure.memory_selector import (
    ATOMIC_MEMORY_STANDARD,
    MemoryCandidate,
)


class AtomicRewritePiece(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1000)
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=300)
    object: str = Field(min_length=1, max_length=300)


class AtomicRewriteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[AtomicRewritePiece]


MEMORY_ATOMIC_REWRITE_INSTRUCTIONS = f"""
You repair one candidate memory that failed the atomic-memory standard.
Return one or more replacement memories. Use only facts already stated in
the candidate. Do not add a fact from outside that text, and do not drop
a fact that the candidate states.

{ATOMIC_MEMORY_STANDARD}

Each replacement is one proposition. Set subject, predicate, and object
to that same proposition. If the candidate is already one proposition,
return that one memory. If it contains more than one, return one memory
per proposition.
""".strip()


class MemoryAtomicRewriter(Protocol):
    def rewrite(
        self,
        candidate: MemoryCandidate,
    ) -> Sequence[AtomicRewritePiece]: ...


class OpenAIMemoryAtomicRewriter:
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

    @timed("Atomic Rewriter")
    def rewrite(
        self,
        candidate: MemoryCandidate,
    ) -> Sequence[AtomicRewritePiece]:
        if not candidate.content.strip():
            raise ValueError("Candidate memory content cannot be empty.")
        payload = {
            "content": candidate.content,
            "subject": candidate.subject,
            "predicate": candidate.predicate,
            "object": candidate.object,
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_ATOMIC_REWRITE_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=AtomicRewriteOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The atomic rewriter returned no structured output."
            )
        return response.output_parsed.memories
