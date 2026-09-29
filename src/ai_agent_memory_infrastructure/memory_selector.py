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
    role: Literal["user", "assistant", "conversation"]
    debug_source_quote: str = Field(min_length=1, max_length=2000)
    debug_reason: str = Field(min_length=1, max_length=500)


class MemorySelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected: list[MemoryCandidate] = Field(
        description=(
            "Durable statements from the latest exchange that should be "
            "stored."
        )
    )
    not_selected: list[MemoryCandidate] = Field(
        description=(
            "Every other semantically meaningful statement in the latest "
            "user and assistant messages. This includes greetings, small "
            "talk, one-off questions, temporary details, and generic "
            "replies. Do not omit them. Leave this empty only when the "
            "latest exchange contains no non-durable content."
        )
    )


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

Audit the complete latest exchange. Put every semantically meaningful atomic
statement into exactly one list:
- selected: durable information that should be stored
- not_selected: everything else, including greetings, small talk, temporary
  details, questions, generic acknowledgements, and other rejected content

Never silently omit a meaningful statement. A statement that is absent from
selected must appear in not_selected, with the rejection explained in
debug_reason.

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
- For a secret or credential, redact the value in both content and
  debug_source_quote rather than reproducing it.

Importance measures future utility and consequence, not certainty. Use:
- 0.0-0.2: trivial or transient
- 0.3-0.4: limited future usefulness
- 0.5-0.6: useful recurring context
- 0.7-0.8: stable preference, goal, project, decision, or constraint
- 0.9-1.0: identity-critical fact, safety requirement, or persistent instruction

Confidence separately measures how strongly the conversation supports the
candidate.

Put these in selected when they are durable and future-useful:
- Long-term preferences
- Projects
- Goals
- Decisions
- Constraints
- Important facts
- Repeated interests
- Long-running tasks
- Relationships between important entities
- Persistent instructions
- Confirmed corrections

Put these in not_selected:
- Greetings
- Small talk
- Temporary statements
- One-off questions
- Information whose only relevance is already obvious from the current
  conversation and has no future value
- Random transient details
- Filler or generic assistant responses
- Speculation or unconfirmed claims
- Passwords, API keys, access tokens, credentials, or other secrets

Return empty selected and not_selected lists only when the latest exchange is
empty or has no semantically meaningful content. If the exchange mixes durable
and non-durable content, selected and not_selected must both be non-empty.

Example:
User: "My name is Beta. The weather is nice today."
Assistant: "Nice to meet you, Beta."
selected:
- "The user's name is Beta."
not_selected:
- "The user said the weather was nice."
- "The assistant greeted Beta."
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
