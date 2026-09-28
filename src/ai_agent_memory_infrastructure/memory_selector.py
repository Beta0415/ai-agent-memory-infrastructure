from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_store import MemoryType


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1000)
    type: MemoryType
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    should_store: bool
    role: Literal["user", "assistant", "conversation"]
    debug_source_quote: str = Field(min_length=1, max_length=2000)
    debug_reason: str = Field(min_length=1, max_length=500)


class MemorySelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[MemoryCandidate]


class MemorySelector(Protocol):
    @property
    def model_name(self) -> str: ...

    def extract_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemorySelectionOutput: ...


MEMORY_SELECTOR_INSTRUCTIONS = """
You are a memory selector for an AI agent.

Read the conversation and break the latest user/assistant exchange into
atomic candidate memory statements. Use earlier turns only to resolve
references and context. Do not repeatedly extract old information.

Each candidate must:
- Be a standalone, concise statement that makes sense outside the chat.
- Use one of these memory types:
  - fact: stable information that is presented as true
  - preference: a user preference or recurring choice
  - goal: an objective or desired future outcome
  - instruction: persistent guidance for future behavior
  - episode: something that happened at a specific point in time
- Preserve important qualifiers and avoid unsupported assumptions.
- Include an exact supporting quote in debug_source_quote.
- Include a brief storage rationale in debug_reason.

Set should_store=true only for durable, future-useful information such as:
- User facts
- Preferences
- Goals
- Persistent instructions
- Confirmed decisions or corrections

Set should_store=false for transient requests, greetings, filler, generic
assistant responses, speculation, unconfirmed claims, or low-value details.
Never store passwords, API keys, access tokens, credentials, or other secrets.

Return an empty candidates list when the latest exchange contains no useful
candidate memories.
""".strip()


class OpenAIMemorySelector:
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

    def extract_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemorySelectionOutput:
        clean_conversation = [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in conversation
        ]
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_SELECTOR_INSTRUCTIONS,
            input=json.dumps(
                {"conversation": clean_conversation},
                ensure_ascii=False,
            ),
            text_format=MemorySelectionOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The memory selector returned no structured output."
            )
        return response.output_parsed
