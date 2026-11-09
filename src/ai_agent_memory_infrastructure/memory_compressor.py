from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory
from ai_agent_memory_infrastructure.memory_timing import timed


class CompressedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class CompressionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[CompressedMemory]


MEMORY_COMPRESSOR_INSTRUCTIONS = """
You shorten memory text that is over a token budget.

You receive several memories at once. Look across them for wording that can
be shortened without losing information. Return one shortened content for
each memory_id.

Rules:
- Preserve every fact, name, number, constraint, negation, and other
  information already stated in that memory.
- Only shorten. Use fewer words than the original content.
- Do not reinterpret, generalize, paraphrase into a different claim, or expand.
- Do not add facts, reasons, labels, or explanations that were not stated.
- Do not omit a memory. Do not add a memory. Do not change a memory_id.
- Return the memories in the same order as the input.
""".strip()


class MemoryCompressor(Protocol):
    def compress(
        self,
        memories: Sequence[RetrievedMemory],
        *,
        token_budget: int,
    ) -> Mapping[str, str]: ...


class OpenAIMemoryCompressor:
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

    @timed("Compressor")
    def compress(
        self,
        memories: Sequence[RetrievedMemory],
        *,
        token_budget: int,
    ) -> Mapping[str, str]:
        if token_budget < 1:
            raise ValueError("Memory token budget must be at least 1.")
        if not memories:
            return {}

        payload = {
            "token_budget": token_budget,
            "memories": [
                {
                    "memory_id": memory.memory.id,
                    "tokens": memory.memory.token_count,
                    "content": memory.memory.content,
                }
                for memory in memories
            ],
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_COMPRESSOR_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=CompressionOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The memory compressor returned no structured output."
            )
        return {
            item.memory_id: item.content
            for item in response.output_parsed.memories
        }
