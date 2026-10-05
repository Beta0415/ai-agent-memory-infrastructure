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
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=300)
    object: str = Field(min_length=1, max_length=300)
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
            "Durable atomic memories from the latest exchange. Each item "
            "is exactly one independently updateable proposition."
        )
    )
    not_selected: list[MemoryCandidate] = Field(
        description=(
            "Every other semantically meaningful atomic statement in the "
            "latest user and assistant messages. Each item is one "
            "proposition. This includes greetings, small talk, one-off "
            "questions, temporary details, and generic replies. Do not "
            "omit them. Leave this empty only when the latest exchange "
            "contains no non-durable content."
        )
    )


class MemorySelector(Protocol):
    @property
    def model_name(self) -> str: ...

    def extract_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> MemorySelectionOutput: ...


ATOMIC_MEMORY_STANDARD = """
An atomic memory contains exactly one independently updateable proposition.
If one part could be corrected, superseded, or contradicted without
changing the rest, split it into separate memories.

Subject, predicate, and object must restate that same proposition and
nothing else. If the content needs a second subject, predicate, or object,
it is not atomic. A list of peers cannot be one object. If "and" joins two
things that could be corrected separately, return one memory for each.

Atomic does not mean the smallest possible sentence. Keep one proposition
together, including the qualifiers that belong to it. A qualifier is the
school, employer, place, time, or purpose that says which fact this is.
When peers are split, copy that shared qualifier onto every memory. Do not
shorten a fact by dropping it. Do not split a tool from the way it is
used, a product from its version, or a database from its extension when
those words state one fact. An ampersand inside a proper name is part of
that name. Do not repeat a proposition in different words, and do not
register a fragment that has no independent meaning.

Split:
BAD: "User scored 116 on TOEFL and 1500 on SAT."
GOOD:
- "User scored 116 on TOEFL."
- "User scored 1500 on SAT."
BAD: "The assistant uses DuckDB, Dify, and Feishu."
GOOD:
- "The assistant uses DuckDB."
- "The assistant uses a Dify knowledge base."
- "The assistant uses Feishu."
BAD: "User likes Python and Java."
GOOD:
- "User likes Python."
- "User likes Java."
BAD: "The user supported XGBoost credit risk model development using Python, NumPy, and scikit-learn."
GOOD:
- "The user supported XGBoost credit risk model development."
- "The user used Python for XGBoost credit risk model development."
- "The user used NumPy for XGBoost credit risk model development."
- "The user used scikit-learn for XGBoost credit risk model development."
BAD: "The user designed a SQL and Superset data pipeline for dashboards and metrics at WeBank."
GOOD:
- "The user designed a SQL data pipeline at WeBank."
- "The user designed a Superset data pipeline at WeBank."
- "The user designed a data pipeline for dashboards at WeBank."
- "The user designed a data pipeline for metrics at WeBank."
BAD: "The user analyzed Southern China forest productivity using environmental and remote-sensing data."
GOOD:
- "The user analyzed Southern China forest productivity using environmental data."
- "The user analyzed Southern China forest productivity using remote-sensing data."
BAD: "The user completed a summer course project on data mining and machine learning for business decision-making."
GOOD:
- "The user completed a summer course project on data mining for business decision-making."
- "The user completed a summer course project on machine learning for business decision-making."
BAD: "The user volunteered as a concert translator providing English-Chinese translation and interpretation."
GOOD:
- "The user volunteered as a concert translator providing English-Chinese translation."
- "The user volunteered as a concert translator providing English-Chinese interpretation."
BAD: "The user is majoring in Economics and Data Science at New York University."
GOOD:
- "The user is majoring in Economics at New York University."
- "The user is majoring in Data Science at New York University."

Do not split, and do not shorten:
BAD: "The user is majoring in Economics."
GOOD: "The user is majoring in Economics at New York University."
GOOD: "MemoryOS uses Python 3.12."
GOOD: "MemoryOS uses PostgreSQL with pgvector."
GOOD: "The user expects to graduate from New York University in May 2027."
GOOD: "The user worked as a Data Analyst Intern in WeBank's International Business & Development Division."
""".strip()


MEMORY_SELECTOR_INSTRUCTIONS = f"""
You are a memory selector for an AI agent.

Read the conversation and break the latest user/assistant exchange into
atomic candidate memories. One exchange may produce several. Use earlier
turns only to resolve references and context. Do not repeatedly extract
old information.

{ATOMIC_MEMORY_STANDARD}

"User scored 116 on TOEFL."
- subject: "User"
- predicate: "scored"
- object: "116 on TOEFL"
"MemoryOS uses PostgreSQL with pgvector."
- subject: "MemoryOS"
- predicate: "uses"
- object: "PostgreSQL with pgvector"

Audit the complete latest exchange. Put every semantically meaningful atomic
statement into exactly one list:
- selected: durable information that should be stored
- not_selected: everything else, including greetings, small talk, temporary
  details, questions, generic acknowledgements, and other rejected content

Never silently omit a meaningful statement. A statement that is absent from
selected must appear in not_selected, with the rejection explained in
debug_reason.

Each candidate must:
- Be one atomic proposition: a standalone, concise statement that makes
  sense outside the chat.
- Set subject, predicate, and object to that same proposition.
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
