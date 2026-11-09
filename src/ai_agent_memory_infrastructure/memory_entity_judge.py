from __future__ import annotations

import json
from typing import Literal, Protocol

from openai import OpenAI

from ai_agent_memory_infrastructure.memory_timing import timed
from pydantic import BaseModel, ConfigDict


class EntityJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["same_entity", "not_same_entity"]


ENTITY_JUDGE_VERDICTS = ("same_entity", "not_same_entity")

MEMORY_ENTITY_JUDGE_INSTRUCTIONS = """
You judge one pair of names. Decide whether they are the same entity.

Return one label:
- same_entity: the candidate is another name for the existing entity.
  This includes an abbreviation or an alternate spelling.
- not_same_entity: they are different entities. A part and its whole are
  different, such as a division and the organization it belongs to.

Return only the label.
""".strip()


class EntityJudge(Protocol):
    def judge(self, candidate: str, existing: str) -> str: ...


class OpenAIEntityJudge:
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

    @timed("Entity Judge")
    def judge(self, candidate: str, existing: str) -> str:
        if not candidate.strip() or not existing.strip():
            raise ValueError("Entity judgment needs two names.")
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_ENTITY_JUDGE_INSTRUCTIONS,
            input=json.dumps(
                {"candidate": candidate, "existing": existing},
                ensure_ascii=False,
            ),
            text_format=EntityJudgeOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError("The entity judge returned no structured output.")
        return response.output_parsed.verdict
