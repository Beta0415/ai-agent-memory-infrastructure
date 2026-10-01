from __future__ import annotations

import streamlit as st

from ai_agent_memory_infrastructure.memory_api import ReadPathSettings

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

st.title("Parameter Configuration")
if st.button("Back To Chat"):
    st.switch_page("app.py")

st.subheader("Weights")
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

try:
    drafted = ReadPathSettings(
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
    st.error(str(exc))
else:
    st.session_state.read_path_settings = drafted
    st.caption("These values apply to the next message.")
