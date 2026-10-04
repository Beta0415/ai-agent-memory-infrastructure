from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import tiktoken

from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory
from ai_agent_memory_infrastructure.memory_store import MemoryType

# Content embeddings above this cosine similarity are the same context item.
CONTEXT_DUPLICATE_SIMILARITY = 0.90
CONTEXT_TYPE_ORDER = (
    MemoryType.FACT,
    MemoryType.PREFERENCE,
    MemoryType.GOAL,
    MemoryType.INSTRUCTION,
    MemoryType.EPISODE,
)
# Token budget for the deduplicated memories passed to the main model.
CONTEXT_TOKEN_BUDGET = 200


@dataclass(frozen=True)
class DuplicateCluster:
    memories: tuple[RetrievedMemory, ...]
    relation: str
    selected: RetrievedMemory | None = None


def embedding_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError(
            "Embeddings must be non-empty and the same length."
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def cluster_duplicate_memories(
    memories: Sequence[RetrievedMemory],
    embeddings: Mapping[str, Sequence[float]],
    *,
    duplicate_similarity: float = CONTEXT_DUPLICATE_SIMILARITY,
) -> tuple[tuple[RetrievedMemory, ...], ...]:
    """Group memories whose content embeddings are near-duplicates.

    Pairwise cosine similarity greater than 0.90 links two memories.
    Linked memories form one group, including through a shared third
    memory. This step only groups. It does not drop or rewrite text.
    """
    if not 0.0 <= duplicate_similarity <= 1.0:
        raise ValueError("Duplicate similarity must be between 0 and 1.")

    vectors = [
        _embedding_for(memory, embeddings) for memory in memories
    ]
    parents = list(range(len(memories)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    for left in range(len(memories)):
        for right in range(left + 1, len(memories)):
            similarity = embedding_similarity(vectors[left], vectors[right])
            if similarity > duplicate_similarity:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(memories)):
        groups.setdefault(find(index), []).append(index)

    clustered = [
        tuple(memories[index] for index in sorted(indexes))
        for _, indexes in sorted(
            groups.items(),
            key=lambda item: min(item[1]),
        )
    ]
    return tuple(clustered)


def resolve_duplicate_group(
    group: Sequence[RetrievedMemory],
    relation: str,
) -> tuple[RetrievedMemory, ...]:
    """Keep the best reranked memory, or keep the whole group.

    equivalent keeps the highest rerank score. contradictory keeps every
    memory. Text is not rewritten, and stored records are not changed.
    """
    if len(group) < 2:
        return tuple(group)
    label = relation.strip().lower()
    if label == "equivalent":
        return (_keep_best(group),)
    if label == "contradictory":
        return tuple(group)
    raise RuntimeError(
        "The relation classifier returned an unknown relation."
    )


def _keep_best(
    group: Sequence[RetrievedMemory],
) -> RetrievedMemory:
    best = group[0]
    for member in group[1:]:
        if (member.final_score, member.semantic_score) > (
            best.final_score,
            best.semantic_score,
        ):
            best = member
    return best


def _embedding_for(
    memory: RetrievedMemory,
    embeddings: Mapping[str, Sequence[float]],
) -> Sequence[float]:
    memory_id = memory.memory.id
    if memory_id not in embeddings:
        raise RuntimeError(
            f"Stored memory embedding is missing: {memory_id}"
        )
    return embeddings[memory_id]


@lru_cache(maxsize=1)
def _token_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_token_encoding().encode(text))


def context_token_count(memories: Sequence[RetrievedMemory]) -> int:
    return sum(memory.memory.token_count for memory in memories)


def apply_context_compression(
    memories: Sequence[RetrievedMemory],
    compressed_contents: Mapping[str, str],
) -> tuple[RetrievedMemory, ...]:
    """Replace memory text only when the new wording is shorter.

    The compressor must return every memory once. A result that is empty
    or not shorter than the original is discarded, so compression cannot
    expand a memory or drop its content.
    """
    expected_ids = [memory.memory.id for memory in memories]
    returned_ids = list(compressed_contents)
    if len(returned_ids) != len(set(returned_ids)):
        raise RuntimeError(
            "The memory compressor returned duplicate memory IDs."
        )
    if set(returned_ids) != set(expected_ids):
        raise RuntimeError(
            "The memory compressor did not return every memory exactly once."
        )

    compressed: list[RetrievedMemory] = []
    for memory in memories:
        proposed = compressed_contents[memory.memory.id].strip()
        proposed_tokens = count_tokens(proposed)
        if not proposed or proposed_tokens >= memory.memory.token_count:
            compressed.append(memory)
            continue
        compressed.append(
            replace(
                memory,
                memory=replace(
                    memory.memory,
                    content=proposed,
                    token_count=proposed_tokens,
                ),
                compressed=True,
            )
        )
    return tuple(compressed)


def order_context_memories(
    memories: Sequence[RetrievedMemory],
) -> tuple[RetrievedMemory, ...]:
    """Group context memories by type, then by relevance within each type."""
    type_rank = {
        memory_type: index
        for index, memory_type in enumerate(CONTEXT_TYPE_ORDER)
    }
    return tuple(
        sorted(
            memories,
            key=lambda item: (
                type_rank.get(
                    item.memory.memory_type,
                    len(CONTEXT_TYPE_ORDER),
                ),
                -item.final_score,
                -item.semantic_score,
                item.memory.id,
            ),
        )
    )


def format_context_memories(
    memories: Sequence[RetrievedMemory],
) -> str:
    """Render each memory on one line under its type heading."""
    lines: list[str] = []
    current_type: str | None = None
    for item in order_context_memories(memories):
        memory_type = item.memory.memory_type.value
        if memory_type != current_type:
            lines.append(memory_type)
            current_type = memory_type
        content = " ".join(item.memory.content.split())
        line = (
            f"- [{item.memory.role}; "
            f"relevance={item.final_score:.3f}; ]"
        )
        if content:
            line = f"{line} {content}"
        lines.append(line)
    return "\n".join(lines)


