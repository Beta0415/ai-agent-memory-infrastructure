from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from openai import OpenAI

from ai_agent_memory_infrastructure.memory_timing import timed
from pydantic import BaseModel, ConfigDict


class EpisodeGroupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: Literal["episode_only", "atom_only"]


EPISODE_GROUP_CHOICES = ("episode_only", "atom_only")

MEMORY_EPISODE_GROUP_INSTRUCTIONS = """
You decide how to represent one episode group for the current user query.

You see the episode title, summary, and entity names. You do not see the episode's other stored memories. You see the atomic memories from this retrieval that belong to the episode.

Does the episode summary preserve enough of the information needed to answer the current query?

Return one label:
- episode_only: the summary preserves enough. The atomic memories in this group are not needed.
- atom_only: the summary does not preserve enough. Keep the atomic memories and do not use the episode.
""".strip()


class EpisodeGroupSelector(Protocol):
    def choose(
        self,
        query: str,
        title: str,
        summary: str,
        entity_names: Sequence[str],
        atomic_contents: Sequence[str],
    ) -> str: ...


class OpenAIEpisodeGroupSelector:
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

    @timed("Group Selector")
    def choose(
        self,
        query: str,
        title: str,
        summary: str,
        entity_names: Sequence[str],
        atomic_contents: Sequence[str],
    ) -> str:
        if not query.strip():
            raise ValueError("Episode group selection needs a query.")
        if not atomic_contents:
            raise ValueError("Episode group selection needs atomic memories.")
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_EPISODE_GROUP_INSTRUCTIONS,
            input=json.dumps(
                {
                    "query": query,
                    "episode": {
                        "title": title,
                        "summary": summary,
                        "entities": list(entity_names),
                    },
                    "atomic_memories": list(atomic_contents),
                },
                ensure_ascii=False,
            ),
            text_format=EpisodeGroupOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The episode group selector returned no structured output."
            )
        choice = response.output_parsed.choice
        if choice not in EPISODE_GROUP_CHOICES:
            raise RuntimeError(
                "The episode group selector returned an unknown choice: "
                f"{choice}"
            )
        return choice
