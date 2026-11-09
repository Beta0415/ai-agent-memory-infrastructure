from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from ai_agent_memory_infrastructure.memory_graph_rag import (
    GraphRagResult,
    GraphRagSettings,
    OpenAIGraphRagAnswerer,
    run_graph_rag,
)
from ai_agent_memory_infrastructure.memory_reranker import OpenAIMemoryReranker
from ai_agent_memory_infrastructure.memory_timing import (
    RuntimeTrace,
    runtime_trace,
)
from ai_agent_memory_infrastructure.postgres_memory_store import (
    PostgresMemoryStore,
)

load_dotenv()

st.set_page_config(
    page_title="Graph RAG",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def _graph_store(database_url: str, embedding_model: str) -> PostgresMemoryStore:
    return PostgresMemoryStore(
        database_url=database_url,
        api_key=os.environ["OPENAI_API_KEY"],
        embedding_model=embedding_model,
    )


def _remember(trace: RuntimeTrace) -> None:
    traces = st.session_state.setdefault("runtime_traces", [])
    if not isinstance(traces, list):
        traces = []
        st.session_state.runtime_traces = traces
    traces.insert(0, trace)
    del traces[40:]


st.title("Graph RAG")
st.caption(
    "A read-only demo of the stored memory graph. "
    "The main chat does not see this result."
)

if "graph_rag_question" not in st.session_state:
    st.session_state.graph_rag_question = ""
question = st.text_input(
    "Ask the Memory Graph",
    key="graph_rag_question",
)
final_top_k, frontier_column = st.columns(2)
with final_top_k:
    final_limit = st.number_input(
        "Final Top K",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )
with frontier_column:
    frontier_cap = st.number_input(
        "Frontier Cap",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="How many of the best passing edges are followed from each entity.",
    )
max_edges, similarity_column = st.columns(2)
with max_edges:
    edge_cap = st.number_input(
        "Max Edges Per Seed",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
    )
with similarity_column:
    hop_similarity = st.number_input(
        "Hop Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="An edge continues the walk only when its memory is at least this similar to the question.",
    )
st.caption("The walk stops after 3 hops. A broad entity is not entered.")

if st.button("Search Graph", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            store = _graph_store(
                os.getenv(
                    "DATABASE_URL",
                    "postgresql://localhost:5432/agent_memory",
                ),
                os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
            reranker = OpenAIMemoryReranker(
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.getenv("MEMORY_RERANKER_MODEL", "gpt-5-nano"),
            )
            answerer = OpenAIGraphRagAnswerer(
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.getenv("OPENAI_MODEL", "gpt-5.6-sol"),
            )
            with runtime_trace("Graph RAG", "Search Graph") as trace:
                st.session_state.graph_rag_result = run_graph_rag(
                    store,
                    question,
                    reranker=reranker,
                    answerer=answerer,
                    settings=GraphRagSettings(
                        final_top_k=int(final_limit),
                        max_edges_per_seed=int(edge_cap),
                        frontier_cap=int(frontier_cap),
                        min_hop_similarity=float(hop_similarity),
                    ),
                )
            _remember(trace)
        except Exception as exc:
            st.session_state.graph_rag_error = str(exc)

error = st.session_state.get("graph_rag_error")
if error:
    st.error(error)

result = st.session_state.get("graph_rag_result")
if isinstance(result, GraphRagResult):
    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Seed Entities")
    if result.seed_entities:
        for seed in result.seed_entities:
            label = "Broad, Explicit" if seed.broad else "Explicit"
            if not seed.explicit:
                label = "Semantic Match"
            st.markdown(f"- **{seed.name}** · `{seed.entity_id}` · {label}")
    else:
        st.caption("No Seed Entities")

    st.subheader("Graph Paths")
    if result.paths:
        for path in result.paths:
            st.markdown(
                f"`{path.source_name}` --{path.relation}--> `{path.target_name}`"
            )
            st.caption(f"Memory {path.memory_id} · Hop {path.hop}")
    else:
        st.caption("No Graph Paths")

    st.subheader("Supporting Memories")
    if result.retrieved_memories:
        for memory in result.retrieved_memories:
            score = f"Similarity {memory.semantic_similarity:.3f}"
            if memory.rerank_score is not None:
                score = f"{score} · Rerank {memory.rerank_score:.3f}"
            with st.expander(f"{memory.memory_id} · {score}"):
                st.write(memory.content)
    else:
        st.caption("No Supporting Memories")
