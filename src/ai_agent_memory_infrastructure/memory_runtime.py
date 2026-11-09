from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_agent_memory_infrastructure.memory_api import (
    MemoryAddResult,
    MemoryAPI,
    ReadPathSettings,
    WritePathSettings,
)
from ai_agent_memory_infrastructure.memory_entity_similarity import (
    EntityResolutionSettings,
)
from ai_agent_memory_infrastructure.memory_episode import (
    EpisodeBuildResult,
    EpisodeSettings,
)
from ai_agent_memory_infrastructure.memory_graph import GraphBuildResult
from ai_agent_memory_infrastructure.memory_maintenance import (
    MaintenanceResult,
    MaintenanceSettings,
)
from ai_agent_memory_infrastructure.memory_context import (
    DuplicateCluster,
    format_context_memories,
)
from ai_agent_memory_infrastructure.memory_hierarchical import (
    format_hierarchical_context,
)
from ai_agent_memory_infrastructure.memory_timing import runtime_span
from ai_agent_memory_infrastructure.memory_retrieval import (
    DEFAULT_RESULT_LIMIT,
    RetrievedMemory,
)
from ai_agent_memory_infrastructure.memory_store import MemoryFilters


@dataclass(frozen=True)
class MemoryContext:
    text: str
    matches: tuple[RetrievedMemory, ...]
    candidates: tuple[RetrievedMemory, ...]
    deduplicated: bool = False
    dropped_memories: tuple[RetrievedMemory, ...] = ()
    duplicate_clusters: tuple[DuplicateCluster, ...] = ()
    episodes: tuple = ()
    ranked: tuple = ()
    episode_candidates: tuple = ()
    pool: tuple = ()
    pool_statuses: tuple[str, ...] = ()


class MemoryRuntime:
    def __init__(self, memory_api: MemoryAPI) -> None:
        self._memory_api = memory_api
        self.last_maintenance: MaintenanceResult | None = None
        self.last_graph: GraphBuildResult | None = None
        self.last_graph_error: str | None = None
        self.last_episode: EpisodeBuildResult | None = None
        self.last_episode_error: str | None = None

    def runtime_build_memory_context(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        settings: ReadPathSettings | None = None,
    ) -> MemoryContext:
        result = self._memory_api.memory_context(
            query,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters,
            now=now,
            settings=settings,
        )
        matches = result.memories
        with runtime_span("Format Context"):
            text = format_context_memories(matches)
        return MemoryContext(
            text=text,
            matches=matches,
            candidates=result.candidates,
            deduplicated=result.deduplicated,
            dropped_memories=result.dropped_memories,
            duplicate_clusters=result.duplicate_clusters,
        )

    def runtime_build_hierarchical_context(
        self,
        query: str,
        *,
        min_similarity: float | None = None,
        filters: MemoryFilters | None = None,
        now: datetime | None = None,
        settings: ReadPathSettings | None = None,
    ) -> MemoryContext:
        result = self._memory_api.memory_hierarchical_context(
            query,
            min_similarity=min_similarity,
            filters=filters,
            now=now,
            settings=settings,
        )
        matches = result.memories
        with runtime_span("Format Context"):
            text = format_hierarchical_context(result.ranked)
        return MemoryContext(
            text=text,
            matches=matches,
            candidates=result.candidates,
            episodes=result.episodes,
            ranked=result.ranked,
            episode_candidates=result.episode_candidates,
            pool=result.pool,
            pool_statuses=result.pool_statuses,
        )

    def runtime_write_memory(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        settings: WritePathSettings | None = None,
        maintenance_settings: MaintenanceSettings | None = None,
    ) -> MemoryAddResult:
        self.last_maintenance = None
        with runtime_span("Write"):
            result = self._memory_api.memory_add(conversation, settings=settings)
        active_maintenance = maintenance_settings or MaintenanceSettings()
        if self._memory_api.memory_total_tokens() > active_maintenance.token_capacity:
            with runtime_span("Maintenance"):
                self.last_maintenance = self._memory_api.memory_maintain(
                    settings=active_maintenance,
                )
        return result

    def runtime_run_maintenance(
        self,
        *,
        now: datetime | None = None,
        maintenance_settings: MaintenanceSettings | None = None,
    ) -> MaintenanceResult:
        return self._memory_api.memory_maintain(
            now=now,
            settings=maintenance_settings,
        )

    def runtime_build_graph(
        self,
        settings: EntityResolutionSettings | None = None,
    ) -> GraphBuildResult:
        result = self._build_graph(settings)
        if self.last_graph_error is not None:
            raise RuntimeError(self.last_graph_error)
        return result

    def _build_graph(
        self,
        settings: EntityResolutionSettings | None = None,
    ) -> GraphBuildResult:
        try:
            result = self._memory_api.memory_graph_build(settings)
        except Exception as exc:
            self.last_graph = None
            self.last_graph_error = str(exc)
            return GraphBuildResult(0, 0, 0)
        self.last_graph = result
        self.last_graph_error = None
        return result

    def runtime_build_episodes(
        self,
        settings: EpisodeSettings | None = None,
    ) -> EpisodeBuildResult:
        result = self._build_episodes(settings)
        if self.last_episode_error is not None:
            raise RuntimeError(self.last_episode_error)
        return result

    def _build_episodes(
        self,
        settings: EpisodeSettings | None = None,
    ) -> EpisodeBuildResult:
        try:
            result = self._memory_api.memory_episode_build(settings)
        except Exception as exc:
            self.last_episode = None
            self.last_episode_error = str(exc)
            return EpisodeBuildResult(0, 0)
        self.last_episode = result
        self.last_episode_error = None
        return result
