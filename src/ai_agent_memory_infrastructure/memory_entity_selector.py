from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_entity_similarity import entity_name_key


class EntityChoiceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(min_length=1)


MEMORY_ENTITY_SELECTOR_INSTRUCTIONS = """
These names were judged to be the same entity. Choose the one best name to
keep as the stored entity. Prefer the fullest proper name.

Return that name exactly as it appears in the list.
""".strip()


class EntitySelector(Protocol):
    def select(self, names: Sequence[str]) -> str: ...


class OpenAIEntitySelector:
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

    def select(self, names: Sequence[str]) -> str:
        choices = [name.strip() for name in names if name.strip()]
        if len(choices) < 2:
            raise ValueError("Entity selection needs at least two names.")
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_ENTITY_SELECTOR_INSTRUCTIONS,
            input=json.dumps({"entities": choices}, ensure_ascii=False),
            text_format=EntityChoiceOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The entity selector returned no structured output."
            )
        chosen = response.output_parsed.entity_name.strip()
        chosen_key = entity_name_key(chosen)
        for name in choices:
            if entity_name_key(name) == chosen_key:
                return name
        raise RuntimeError(
            "The entity selector returned a name that was not one of the "
            f"entities: {chosen}"
        )
