from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from ai_agent_memory_infrastructure.memory_store import Memory
from ai_agent_memory_infrastructure.memory_timing import timed


class DedupVerdictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["equivalent", "not_equivalent"]


DEDUP_VERDICTS = ("equivalent", "not_equivalent")

MEMORY_DEDUP_INSTRUCTIONS = """
You decide whether one stored memory states the same fact as the others.
The others already passed a vector similarity check.

Return one label:
- equivalent: they state the same facts. Wording may differ. No memory
  adds, removes, or denies a fact that another memory states.
- not_equivalent: anything else. This includes a refinement, a newer
  state, a correction, a contradiction, or a distinct fact.

Do not rewrite or choose which row to keep. Return only the label.
""".strip()


class MemoryDedupClassifier(Protocol):
    def classify(
        self,
        candidate: Memory,
        related: Sequence[Memory],
    ) -> str: ...


class OpenAIMemoryDedupClassifier:
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

    @timed("Dedup Classifier")
    def classify(
        self,
        candidate: Memory,
        related: Sequence[Memory],
    ) -> str:
        if not related:
            raise ValueError("Dedup classification needs a related memory.")
        payload = {
            "candidate": candidate.content,
            "related_memories": [
                {"memory_id": memory.id, "content": memory.content}
                for memory in related
            ],
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_DEDUP_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=DedupVerdictOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError("The dedup classifier returned no structured output.")
        return response.output_parsed.verdict
