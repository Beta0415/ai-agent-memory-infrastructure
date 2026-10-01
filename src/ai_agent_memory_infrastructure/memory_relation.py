from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory


class RelationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: Literal["equivalent", "contradictory"]


MEMORY_RELATION_INSTRUCTIONS = """
You classify the relation between memories grouped only because their
embeddings are similar.

Return one label for the whole group:
- equivalent: they state the same facts. Wording may differ. No memory
  adds, removes, or denies a fact that another memory states.
- contradictory: they disagree, one negates another, they give different
  values, or one memory states a fact the others do not.

Do not rewrite, merge, or choose a winner. Return only the label.
""".strip()


class MemoryRelationClassifier(Protocol):
    def classify(
        self,
        memories: Sequence[RetrievedMemory],
    ) -> str: ...


class OpenAIMemoryRelationClassifier:
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
        memories: Sequence[RetrievedMemory],
    ) -> str:
        if len(memories) < 2:
            raise ValueError(
                "Relation classification needs at least two memories."
            )
        payload = {
            "memories": [
                {
                    "memory_id": memory.memory.id,
                    "content": memory.memory.content,
                }
                for memory in memories
            ]
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_RELATION_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=RelationOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The relation classifier returned no structured output."
            )
        return response.output_parsed.relation
