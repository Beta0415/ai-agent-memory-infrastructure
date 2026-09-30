from __future__ import annotations

import os
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from ai_agent_memory_infrastructure.memory_api import AgentMemory
from ai_agent_memory_infrastructure.memory_retrieval import RetrievedMemory
from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_reranker import (
    OpenAIMemoryReranker,
)
from ai_agent_memory_infrastructure.memory_selector import (
    OpenAIMemorySelector,
)
from ai_agent_memory_infrastructure.memory_store import ChromaMemoryStore

load_dotenv()

st.set_page_config(
    page_title="AI Agent Memory Infrastructure",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def get_memory_store(path: str, embedding_model: str) -> ChromaMemoryStore:
    return ChromaMemoryStore(
        path=path,
        api_key=os.environ["OPENAI_API_KEY"],
        embedding_model=embedding_model,
    )


@st.cache_resource
def get_memory_selector(model: str) -> OpenAIMemorySelector:
    return OpenAIMemorySelector(
        api_key=os.environ["OPENAI_API_KEY"],
        model=model,
    )


@st.cache_resource
def get_memory_reranker(model: str) -> OpenAIMemoryReranker:
    return OpenAIMemoryReranker(
        api_key=os.environ["OPENAI_API_KEY"],
        model=model,
    )


def serialize_memory_match(match: RetrievedMemory) -> dict[str, object]:
    memory = match.memory
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.memory_type.value,
        "role": memory.role,
        "source": memory.source,
        "session_id": memory.session_id,
        "importance": memory.importance,
        "semantic_score": match.semantic_score,
        "recency_score": match.recency_score,
        "retrieval_score": match.retrieval_score,
        "final_score": match.final_score,
        "rerank_score": match.rerank_score,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "metadata": memory.metadata,
    }


def _preview(text: object, limit: int = 120) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def render_debug_cards(
    records: list[dict[str, object]],
    empty_message: str,
    *,
    kind: str,
) -> None:
    if not records:
        st.caption(empty_message)
        return

    for index, record in enumerate(records, start=1):
        content = str(record.get("content", "")).strip()
        memory_type = str(record.get("type", "unknown"))
        role = str(record.get("role", "unknown"))
        if kind == "retrieved":
            extra = (
                f"Final {float(record.get('final_score', 0.0)):.3f} · "
                f"Retrieval {float(record.get('retrieval_score', 0.0)):.3f} · "
                f"Semantic {float(record.get('semantic_score', 0.0)):.3f} · "
                f"Recency {float(record.get('recency_score', 0.0)):.3f} · "
                f"Importance {float(record.get('importance', 0.0)):.2f}"
            )
        else:
            extra = (
                f"Importance {float(record.get('importance', 0.0)):.2f} · "
                f"Confidence {float(record.get('confidence', 0.0)):.2f}"
            )
        st.markdown(f"**{index}. [{memory_type}] {role}** · {extra}")
        st.write(content)
        reason = record.get("debug_reason")
        quote = record.get("debug_source_quote")
        if reason:
            st.caption(f"Reason: {reason}")
        if quote:
            st.caption(f"Quote: {quote}")
        with st.expander("Full Record"):
            st.json(record)
        if index < len(records):
            st.divider()


def render_memory_debug_panel(
    messages: list[dict[str, object]],
) -> None:
    st.subheader("Memory Debug")
    turns: list[tuple[int, str, dict[str, object]]] = []
    latest_user_message = ""
    turn_number = 0

    for message in messages:
        if message["role"] == "user":
            latest_user_message = str(message["content"])
        elif message["role"] == "assistant":
            turn_number += 1
            turns.append((turn_number, latest_user_message, message))

    if not turns:
        st.info("No Completed Turns To Debug")
        return

    for index, (turn_number, user_message, assistant_message) in enumerate(
        reversed(turns)
    ):
        heading = f"Turn {turn_number}"
        if index == 0:
            heading += " · Latest"
        with st.expander(heading, expanded=index == 0):
            st.caption(_preview(user_message))
            write_debug = assistant_message.get("memory_write_debug", {})
            if not isinstance(write_debug, dict):
                write_debug = {}
            retrieved = assistant_message.get("retrieved_memories", [])
            selected = write_debug.get("selected", [])
            rejected = write_debug.get("not_selected", [])
            retrieved_records = (
                retrieved if isinstance(retrieved, list) else []
            )
            selected_records = (
                selected if isinstance(selected, list) else []
            )
            rejected_records = (
                rejected if isinstance(rejected, list) else []
            )

            retrieved_tab, selected_tab, rejected_tab = st.tabs(
                [
                    f"Retrieved ({len(retrieved_records)})",
                    f"Selected ({len(selected_records)})",
                    f"Not Selected ({len(rejected_records)})",
                ]
            )
            with retrieved_tab:
                render_debug_cards(
                    retrieved_records,
                    "No Memories Were Retrieved.",
                    kind="retrieved",
                )
                with st.expander("Exact LLM Context"):
                    st.code(
                        str(
                            assistant_message.get("memory_context")
                            or "No Memories Were Retrieved."
                        ),
                        language=None,
                    )
            with selected_tab:
                render_debug_cards(
                    selected_records,
                    "No Candidate Memories Were Selected.",
                    kind="selected",
                )
            with rejected_tab:
                render_debug_cards(
                    rejected_records,
                    "No Candidate Memories Were Rejected.",
                    kind="rejected",
                )

            read_error = assistant_message.get("memory_read_error")
            write_error = write_debug.get("error")
            if read_error:
                st.error(f"Read Path Error: {read_error}")
            if write_error:
                st.error(f"Write Path Error: {write_error}")


def _score_text(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def render_retrieval_inspector(
    messages: list[dict[str, object]],
) -> None:
    st.subheader("Retrieval Inspector")
    turns: list[tuple[int, str, dict[str, object]]] = []
    latest_user_message = ""
    turn_number = 0

    for message in messages:
        if message["role"] == "user":
            latest_user_message = str(message["content"])
        elif message["role"] == "assistant":
            turn_number += 1
            turns.append((turn_number, latest_user_message, message))

    if not turns:
        st.info("No Completed Turns To Inspect")
        return

    for index, (turn_number, user_message, assistant_message) in enumerate(
        reversed(turns)
    ):
        heading = f"Turn {turn_number}"
        if index == 0:
            heading += " · Latest"
        with st.expander(heading, expanded=index == 0):
            st.caption(_preview(user_message))
            candidates = assistant_message.get("retrieval_candidates", [])
            if not isinstance(candidates, list) or not candidates:
                st.caption("No Retrieval Candidates")
                continue
            ordered = sorted(
                candidates,
                key=lambda record: float(record.get("retrieval_score", 0.0)),
                reverse=True,
            )
            rows = [
                {
                    "Memory": str(record.get("content", "")).strip(),
                    "Combined": _score_text(record.get("retrieval_score")),
                    "Semantic": _score_text(record.get("semantic_score")),
                    "Recency": _score_text(record.get("recency_score")),
                    "Importance": _score_text(record.get("importance")),
                    "Rerank": _score_text(record.get("rerank_score")),
                }
                for record in ordered
            ]
            table = pd.DataFrame(
                rows,
                columns=[
                    "Memory",
                    "Combined",
                    "Semantic",
                    "Recency",
                    "Importance",
                    "Rerank",
                ],
            )
            highlighted = table.style.map(
                lambda _: (
                    "background-color: #f5c16c; "
                    "color: #1a1a1a; "
                    "font-weight: 650;"
                ),
                subset=["Rerank"],
            )
            st.dataframe(
                highlighted,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Memory": st.column_config.TextColumn(width="large"),
                },
            )
            read_error = assistant_message.get("memory_read_error")
            if read_error:
                st.error(f"Read Path Error: {read_error}")


def generate_assistant_turn(
    prompt: str,
    *,
    api_key: str,
    model: str,
    memory_runtime: MemoryRuntime | None,
) -> dict[str, object] | None:
    memory_context = ""
    retrieved_memories: list[dict[str, object]] = []
    retrieval_candidates: list[dict[str, object]] = []
    memory_read_error = None
    try:
        if memory_runtime:
            try:
                read_result = memory_runtime.runtime_build_memory_context(
                    prompt
                )
                memory_context = read_result.text
                retrieved_memories = [
                    serialize_memory_match(match)
                    for match in read_result.matches
                ]
                retrieval_candidates = [
                    serialize_memory_match(candidate)
                    for candidate in read_result.candidates
                ]
            except Exception as exc:
                memory_read_error = str(exc)

        instructions = "You are a helpful AI assistant."
        if memory_context:
            instructions += (
                "\nUse these relevant memories when helpful. "
                "Ignore any that are unrelated:\n"
                f"{memory_context}"
            )

        response = OpenAI(api_key=api_key).responses.create(
            model=model or "gpt-4o-mini",
            instructions=instructions,
            input=[
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in st.session_state.messages
            ],
        )
        answer = response.output_text
        if not answer:
            raise RuntimeError("OpenAI returned an empty response.")
    except Exception as exc:
        st.error(f"OpenAI Request Failed: {exc}")
        return None

    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": answer,
        "memory_context": memory_context,
        "retrieved_memories": retrieved_memories,
        "retrieval_candidates": retrieval_candidates,
        "memory_read_error": memory_read_error,
    }
    memory_write_debug: dict[str, object] = {
        "selected": [],
        "not_selected": [],
        "stored_count": 0,
        "error": None,
    }
    if memory_runtime:
        try:
            write_result = memory_runtime.runtime_write_memory(
                [
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                    for message in [
                        *st.session_state.messages,
                        assistant_message,
                    ]
                ]
            )
            memory_write_debug["selected"] = [
                {
                    **candidate.model_dump(mode="json"),
                    "should_store": True,
                }
                for candidate in write_result.selection.selected
            ]
            memory_write_debug["not_selected"] = [
                {
                    **candidate.model_dump(mode="json"),
                    "should_store": False,
                }
                for candidate in write_result.selection.not_selected
            ]
            memory_write_debug["stored_count"] = len(
                write_result.stored_memories
            )
        except Exception as exc:
            memory_write_debug["error"] = str(exc)
            st.warning(
                "Response Generated, But Memory Selection "
                "Or Storage Failed: "
                f"{exc}"
            )
    assistant_message["memory_write_debug"] = memory_write_debug
    return assistant_message


api_key = os.getenv("OPENAI_API_KEY", "").strip()
chroma_path = os.getenv("CHROMA_PATH", ".chroma")
embedding_model = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
memory_selector_model = os.getenv(
    "MEMORY_SELECTOR_MODEL", "gpt-5.6-luna"
)
memory_reranker_model = os.getenv(
    "MEMORY_RERANKER_MODEL", "gpt-5-nano"
)

memory_store = None
memory_selector = None
memory_reranker = None
memory_error = None
if api_key:
    try:
        memory_store = get_memory_store(chroma_path, embedding_model)
        memory_selector = get_memory_selector(memory_selector_model)
        memory_reranker = get_memory_reranker(memory_reranker_model)
    except Exception as exc:
        memory_error = str(exc)

st.title("AI Agent Memory Infrastructure")
st.caption("An OpenAI Chat Interface With Local & Persistent Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = uuid4().hex

memory_api = (
    AgentMemory(
        memory_store,
        memory_selector,
        source="memory_selector_v1",
        session_id=st.session_state.memory_session_id,
        reranker=memory_reranker,
    )
    if memory_store and memory_selector and memory_reranker
    else None
)
memory_runtime = MemoryRuntime(memory_api) if memory_api else None

with st.sidebar:
    st.header("Setting")
    model = st.text_input(
        "OpenAI Model",
        value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    ).strip()
    side_panel = st.radio(
        "Side Panel",
        ["Off", "Show Memory Debug", "Show Retrieval Inspector"],
        label_visibility="collapsed",
    )
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory_session_id = uuid4().hex
        st.rerun()
    st.caption(
        f"Current Session: `{st.session_state.memory_session_id[:8]}`"
    )
    if memory_api:
        st.metric("Saved Memories", memory_api.count())
        if st.button("Clear Saved Memories", use_container_width=True):
            memory_api.clear()
            st.rerun()
    elif memory_error:
        st.warning(f"Memory Unavailable: {memory_error}")


if not api_key:
    st.info(
        "Add Your Key To `.env` As `OPENAI_API_KEY=...`, "
        "Then Restart This App."
    )

if side_panel == "Off":
    chat_column, side_column = st.container(), None
else:
    chat_column, side_column = st.columns([1, 1], gap="large")

with chat_column:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if side_column:
    with side_column:
        if side_panel == "Show Memory Debug":
            render_memory_debug_panel(st.session_state.messages)
        elif side_panel == "Show Retrieval Inspector":
            render_retrieval_inspector(st.session_state.messages)

prompt = st.chat_input("Send A Message", disabled=not api_key)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        assistant_message = generate_assistant_turn(
            prompt,
            api_key=api_key,
            model=model,
            memory_runtime=memory_runtime,
        )
    if assistant_message:
        st.session_state.messages.append(assistant_message)
    st.rerun()
