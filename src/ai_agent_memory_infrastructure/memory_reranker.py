from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory


class RerankedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)


class RerankOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rankings: list[RerankedMemory]


MEMORY_RERANKER_INSTRUCTIONS = """
You are a memory reranker for an AI agent.

You receive a user query and a small set of candidate memories. Score how
useful each memory is for answering that query. Judge only the relationship
between the query and the memory content.

Return every memory_id exactly once.
relevance_score is a number from 0 to 1:
- 1.0: directly answers or is necessary for the query
- 0.5: related, but incomplete or indirect
- 0.0: unrelated to the query

Do not omit a candidate. Do not add new memories. Do not change memory content.
""".strip()


class OpenAIMemoryReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5-nano",
    ) -> None:
        self._model = model
        self._openai = OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedMemory],
    ) -> Mapping[str, float]:
        if not query.strip():
            raise ValueError("Memory rerank query cannot be empty.")
        if not candidates:
            return {}

        payload = {
            "query": query,
            "candidates": [
                {
                    "memory_id": candidate.memory.id,
                    "content": candidate.memory.content,
                }
                for candidate in candidates
            ],
        }
        response = self._openai.responses.parse(
            model=self._model,
            instructions=MEMORY_RERANKER_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=RerankOutput,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                "The memory reranker returned no structured output."
            )
        return _scores_for_candidates(
            candidates,
            response.output_parsed.rankings,
        )


def _scores_for_candidates(
    candidates: Sequence[RetrievedMemory],
    rankings: Sequence[RerankedMemory],
) -> dict[str, float]:
    expected_ids = [candidate.memory.id for candidate in candidates]
    returned_ids = [ranking.memory_id for ranking in rankings]
    if len(returned_ids) != len(set(returned_ids)):
        raise RuntimeError(
            "The memory reranker returned duplicate memory IDs."
        )
    if set(returned_ids) != set(expected_ids):
        raise RuntimeError(
            "The memory reranker did not score every candidate exactly once."
        )
    return {
        ranking.memory_id: ranking.relevance_score for ranking in rankings
    }
