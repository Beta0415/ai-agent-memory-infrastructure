from __future__ import annotations

import streamlit as st

from ai_agent_memory_infrastructure.memory_api import (
    ReadPathSettings,
    WritePathSettings,
)
from ai_agent_memory_infrastructure.memory_entity_similarity import (
    EntityResolutionSettings,
)
from ai_agent_memory_infrastructure.memory_maintenance import (
    MaintenanceSettings,
)

st.set_page_config(
    page_title="Parameter Configuration",
    page_icon="🧠",
    layout="wide",
)


def _seed(key: str, value: object) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


applied = st.session_state.get("read_path_settings")
if not isinstance(applied, ReadPathSettings):
    applied = ReadPathSettings()
    st.session_state.read_path_settings = applied

_seed("param_semantic_weight", applied.semantic_weight)
_seed("param_recency_weight", applied.recency_weight)
_seed("param_importance_weight", applied.importance_weight)
_seed(
    "param_recency_half_life_days",
    float(applied.recency_half_life_days),
)
_seed("param_candidate_limit", applied.candidate_limit)
_seed("param_weighted_limit", applied.weighted_limit)
_seed("param_result_limit", applied.result_limit)
_seed("param_duplicate_similarity", applied.duplicate_similarity)
_seed("param_token_budget", applied.token_budget)

write_applied = st.session_state.get("write_path_settings")
if not isinstance(write_applied, WritePathSettings):
    write_applied = WritePathSettings()
    st.session_state.write_path_settings = write_applied

_seed("param_predicate_weight", write_applied.predicate_weight)
_seed("param_list_weight", write_applied.list_weight)
_seed("param_clause_weight", write_applied.clause_weight)
_seed("param_coordination_weight", write_applied.coordination_weight)
_seed("param_value_weight", write_applied.value_weight)
_seed("param_length_weight", write_applied.length_weight)
_seed("param_related_similarity", write_applied.related_similarity)
_seed("param_related_limit", write_applied.related_limit)
_seed(
    "param_atomic_score_threshold",
    write_applied.atomic_score_threshold,
)

maintenance_applied = st.session_state.get("maintenance_settings")
if not isinstance(maintenance_applied, MaintenanceSettings):
    maintenance_applied = MaintenanceSettings()
    st.session_state.maintenance_settings = maintenance_applied
_seed("param_token_capacity", maintenance_applied.token_capacity)
_seed(
    "param_dedup_similarity",
    maintenance_applied.similarity_threshold,
)
_seed(
    "param_dedup_importance_weight",
    maintenance_applied.importance_weight,
)
_seed(
    "param_dedup_recency_weight",
    maintenance_applied.recency_weight,
)
_seed(
    "param_dedup_half_life_days",
    float(maintenance_applied.half_life_days),
)

entity_applied = st.session_state.get("entity_resolution_settings")
if not isinstance(entity_applied, EntityResolutionSettings):
    entity_applied = EntityResolutionSettings()
    st.session_state.entity_resolution_settings = entity_applied
_seed(
    "param_levenshtein_similarity",
    entity_applied.levenshtein_threshold,
)
_seed("param_jaccard_similarity", entity_applied.jaccard_threshold)
_seed(
    "param_embedding_similarity",
    entity_applied.embedding_threshold,
)

st.title("Parameter Configuration")
if st.button("Back To Chat"):
    st.switch_page("app.py")

st.subheader("Retrieval Weights")
semantic_column, recency_column, importance_column = st.columns(3)
with semantic_column:
    st.number_input(
        "Semantic Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_semantic_weight",
    )
with recency_column:
    st.number_input(
        "Recency Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_recency_weight",
    )
with importance_column:
    st.number_input(
        "Importance Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_importance_weight",
    )
weight_sum = (
    float(st.session_state.param_semantic_weight)
    + float(st.session_state.param_recency_weight)
    + float(st.session_state.param_importance_weight)
)
st.caption(f"Sum: {weight_sum:.2f}")

st.subheader("Recency")
st.number_input(
    "Half-Life Days",
    min_value=1.0,
    max_value=3650.0,
    step=1.0,
    format="%.0f",
    key="param_recency_half_life_days",
)
st.caption("Recency falls by half over this many days.")

st.subheader("Retrieval Cuts")
semantic_cut, weighted_cut, returned_cut = st.columns(3)
with semantic_cut:
    st.number_input(
        "Semantic Candidates",
        min_value=1,
        max_value=200,
        step=1,
        key="param_candidate_limit",
    )
with weighted_cut:
    st.number_input(
        "Weighted Candidates",
        min_value=1,
        max_value=200,
        step=1,
        key="param_weighted_limit",
    )
with returned_cut:
    st.number_input(
        "Returned Memories",
        min_value=1,
        max_value=200,
        step=1,
        key="param_result_limit",
    )
st.caption(
    "Semantic candidates, then the weighted pool, then the number returned. "
    "Each cut must be at least as large as the next."
)

st.subheader("Duplicate Clustering")
st.number_input(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    format="%.2f",
    key="param_duplicate_similarity",
)
st.caption(
    "Memories cluster when cosine similarity is greater than this value."
)

st.subheader("Token Budget")
st.number_input(
    "Content Token Limit",
    min_value=1,
    max_value=100000,
    step=1,
    key="param_token_budget",
)
st.caption(
    "Counts memory content only. Compression runs when the total is over this limit."
)

st.subheader("Atomic Score Weights")
predicate_column, list_column, clause_column = st.columns(3)
coordination_column, value_column, length_column = st.columns(3)
with predicate_column:
    st.number_input(
        "Predicate Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_predicate_weight",
    )
with list_column:
    st.number_input(
        "List Item Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_list_weight",
    )
with clause_column:
    st.number_input(
        "Clause Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_clause_weight",
    )
with coordination_column:
    st.number_input(
        "Coordination Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_coordination_weight",
    )
with value_column:
    st.number_input(
        "Value Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_value_weight",
    )
with length_column:
    st.number_input(
        "Length Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_length_weight",
    )
atomic_weight_sum = (
    float(st.session_state.param_predicate_weight)
    + float(st.session_state.param_list_weight)
    + float(st.session_state.param_clause_weight)
    + float(st.session_state.param_coordination_weight)
    + float(st.session_state.param_value_weight)
    + float(st.session_state.param_length_weight)
)
st.caption(f"Sum: {atomic_weight_sum:.2f}")
st.caption("These weights must sum to 1.")
st.number_input(
    "Atomic Score Threshold",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    format="%.2f",
    key="param_atomic_score_threshold",
)
st.caption(
    "A score at or above this value is sent to the atomic checker. "
    "A lower score continues to memory comparison."
)

st.subheader("Memory Comparison")
similarity_column, limit_column = st.columns(2)
with similarity_column:
    st.number_input(
        "Related Similarity",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_related_similarity",
    )
with limit_column:
    st.number_input(
        "Related Memories",
        min_value=1,
        max_value=200,
        step=1,
        key="param_related_limit",
    )
st.caption(
    "A new memory is compared with at most this many stored memories "
    "whose similarity is greater than the threshold."
)

st.subheader("Maintenance")
st.number_input(
    "Memory Token Capacity",
    min_value=1,
    max_value=10_000_000,
    step=100,
    key="param_token_capacity",
)
st.caption(
    "Maintenance runs when stored memory tokens exceed this capacity. "
    "A memory is tagged Expired after 60 days without a create, "
    "update, retrieval, or reinforcement. Its status stays unchanged."
)
st.number_input(
    "Dedup Similarity",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    format="%.2f",
    key="param_dedup_similarity",
)
st.caption(
    "Maintenance groups memories when cosine similarity is greater than this value."
)
importance_dedup, recency_dedup, half_life_dedup = st.columns(3)
with importance_dedup:
    st.number_input(
        "Dedup Importance Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_dedup_importance_weight",
    )
with recency_dedup:
    st.number_input(
        "Dedup Recency Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_dedup_recency_weight",
    )
with half_life_dedup:
    st.number_input(
        "Dedup Half-Life Days",
        min_value=1.0,
        max_value=3650.0,
        step=1.0,
        format="%.0f",
        key="param_dedup_half_life_days",
    )
dedup_weight_sum = (
    float(st.session_state.param_dedup_importance_weight)
    + float(st.session_state.param_dedup_recency_weight)
)
st.caption(f"Sum: {dedup_weight_sum:.2f}")
st.caption(
    "These weights choose which duplicate to keep. Recency ages from "
    "reinforced_at using this half-life."
)

st.subheader("Entity Resolution")
levenshtein_column, jaccard_column, embedding_column = st.columns(3)
with levenshtein_column:
    st.number_input(
        "Levenshtein Similarity",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_levenshtein_similarity",
    )
with jaccard_column:
    st.number_input(
        "Jaccard Similarity",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_jaccard_similarity",
    )
with embedding_column:
    st.number_input(
        "Embedding Similarity",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="param_embedding_similarity",
    )
st.caption(
    "A new name is sent to the entity judge when any one of these "
    "is at or above its threshold. These values apply to the next "
    "Build Graph."
)

read_error = None
write_error = None
try:
    drafted_read = ReadPathSettings(
        semantic_weight=float(st.session_state.param_semantic_weight),
        recency_weight=float(st.session_state.param_recency_weight),
        importance_weight=float(st.session_state.param_importance_weight),
        recency_half_life_days=float(
            st.session_state.param_recency_half_life_days
        ),
        candidate_limit=int(st.session_state.param_candidate_limit),
        weighted_limit=int(st.session_state.param_weighted_limit),
        result_limit=int(st.session_state.param_result_limit),
        duplicate_similarity=float(
            st.session_state.param_duplicate_similarity
        ),
        token_budget=int(st.session_state.param_token_budget),
    )
except ValueError as exc:
    read_error = str(exc)
    drafted_read = None
try:
    drafted_write = WritePathSettings(
        predicate_weight=float(st.session_state.param_predicate_weight),
        list_weight=float(st.session_state.param_list_weight),
        clause_weight=float(st.session_state.param_clause_weight),
        coordination_weight=float(
            st.session_state.param_coordination_weight
        ),
        value_weight=float(st.session_state.param_value_weight),
        length_weight=float(st.session_state.param_length_weight),
        related_similarity=float(
            st.session_state.param_related_similarity
        ),
        related_limit=int(st.session_state.param_related_limit),
        atomic_score_threshold=float(
            st.session_state.param_atomic_score_threshold
        ),
    )
except ValueError as exc:
    write_error = str(exc)
    drafted_write = None
try:
    drafted_maintenance = MaintenanceSettings(
        token_capacity=int(st.session_state.param_token_capacity),
        similarity_threshold=float(st.session_state.param_dedup_similarity),
        importance_weight=float(
            st.session_state.param_dedup_importance_weight
        ),
        recency_weight=float(st.session_state.param_dedup_recency_weight),
        half_life_days=float(st.session_state.param_dedup_half_life_days),
    )
except ValueError as exc:
    st.error(str(exc))
    drafted_maintenance = None
else:
    st.session_state.maintenance_settings = drafted_maintenance

entity_error = None
try:
    drafted_entity = EntityResolutionSettings(
        levenshtein_threshold=float(
            st.session_state.param_levenshtein_similarity
        ),
        jaccard_threshold=float(st.session_state.param_jaccard_similarity),
        embedding_threshold=float(
            st.session_state.param_embedding_similarity
        ),
    )
except ValueError as exc:
    entity_error = str(exc)
    drafted_entity = None
else:
    st.session_state.entity_resolution_settings = drafted_entity

if read_error:
    st.error(read_error)
if write_error:
    st.error(write_error)
if entity_error:
    st.error(entity_error)
if (
    drafted_read is not None
    and drafted_write is not None
    and drafted_entity is not None
    and read_error is None
    and write_error is None
    and entity_error is None
):
    st.session_state.read_path_settings = drafted_read
    st.session_state.write_path_settings = drafted_write
    st.caption(
        "Retrieval, write, and maintenance values apply to the next message. "
        "Entity thresholds apply to the next Build Graph."
    )
