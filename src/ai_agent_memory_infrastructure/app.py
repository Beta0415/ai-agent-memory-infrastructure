from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from ai_agent_memory_infrastructure.memory_api import AgentMemory
from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_selector import (
    OpenAIMemorySelector,
)
from ai_agent_memory_infrastructure.memory_store import (
    ChromaMemoryStore,
    MemoryMatch,
)

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


def serialize_memory_match(match: MemoryMatch) -> dict[str, object]:
    memory = match.memory
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.memory_type.value,
        "role": memory.role,
        "source": memory.source,
        "session_id": memory.session_id,
        "importance": memory.importance,
        "similarity_score": match.similarity_score,
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
                f"Similarity {float(record.get('similarity_score', 0.0)):.3f}"
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


def generate_assistant_turn(
    prompt: str,
    *,
    api_key: str,
    model: str,
    memory_runtime: MemoryRuntime | None,
) -> dict[str, object] | None:
    memory_context = ""
    retrieved_memories: list[dict[str, object]] = []
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

memory_store = None
memory_selector = None
memory_error = None
if api_key:
    try:
        memory_store = get_memory_store(chroma_path, embedding_model)
        memory_selector = get_memory_selector(memory_selector_model)
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
    )
    if memory_store and memory_selector
    else None
)
memory_runtime = MemoryRuntime(memory_api) if memory_api else None

with st.sidebar:
    st.header("Setting")
    model = st.text_input(
        "OpenAI Model",
        value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    ).strip()
    show_memory_debug = st.checkbox("Show Memory Debug")
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

if show_memory_debug:
    chat_column, debug_column = st.columns([1, 1], gap="large")
else:
    chat_column = st.container()
    debug_column = None

with chat_column:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if debug_column:
    with debug_column:
        render_memory_debug_panel(st.session_state.messages)

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
