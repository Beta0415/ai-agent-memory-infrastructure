from __future__ import annotations

import json
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from ai_agent_memory_infrastructure.memory_selector import (
    ATOMIC_MEMORY_STANDARD,
    MemoryCandidate,
)


class AtomicCheckOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["valid", "invalid"]


ATOMIC_CHECK_VERDICTS = ("valid", "invalid")

MEMORY_ATOMIC_CHECK_INSTRUCTIONS = f"""
You check one candidate memory. Use the same atomic-memory standard as
the memory extractor.

{ATOMIC_MEMORY_STANDARD}

Return valid when the candidate meets that standard.
Return invalid when it does not.
Do not rewrite or split the memory. Return only the label.
""".strip()


class MemoryAtomicChecker(Protocol):
    def check(self, candidate: MemoryCandidate) -> str: ...


class OpenAIMemoryAtomicChecker:
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

    def check(self, candidate: MemoryCandidate) -> str:
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
            instructions=MEMORY_ATOMIC_CHECK_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=AtomicCheckOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The atomic checker returned no structured output."
            )
        return response.output_parsed.verdict
