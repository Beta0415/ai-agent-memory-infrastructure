from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from ai_agent_memory_infrastructure.memory_runtime import MemoryRuntime
from ai_agent_memory_infrastructure.memory_store import ChromaMemoryStore

load_dotenv()


@st.cache_resource
def get_memory_store(path: str, embedding_model: str) -> ChromaMemoryStore:
    return ChromaMemoryStore(
        path=path,
        api_key=os.environ["OPENAI_API_KEY"],
        embedding_model=embedding_model,
    )


api_key = os.getenv("OPENAI_API_KEY", "").strip()
chroma_path = os.getenv("CHROMA_PATH", ".chroma")
embedding_model = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

memory_store = None
memory_error = None
if api_key:
    try:
        memory_store = get_memory_store(chroma_path, embedding_model)
    except Exception as exc:
        memory_error = str(exc)

st.set_page_config(
    page_title="AI Agent Memory Infrastructure", page_icon="🧠"
)
st.title("AI Agent Memory Infrastructure")
st.caption("An OpenAI Chat Interface With Local & Persistent Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = uuid4().hex

memory_runtime = (
    MemoryRuntime(
        memory_store,
        source="streamlit_chat",
        session_id=st.session_state.memory_session_id,
    )
    if memory_store
    else None
)

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
    if memory_store:
        st.metric("Saved Memories", memory_store.count())
        if st.button("Clear Saved Memories", use_container_width=True):
            memory_store.clear()
            st.rerun()
    elif memory_error:
        st.warning(f"Memory Unavailable: {memory_error}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if (
            show_memory_debug
            and message["role"] == "assistant"
            and "memory_context" in message
        ):
            with st.expander("Retrieved Memory Context"):
                st.code(
                    message["memory_context"]
                    or "No Memories Were Retrieved.",
                    language=None,
                )
        st.markdown(message["content"])

if not api_key:
    st.info(
        "Add Your Key To `.env` As `OPENAI_API_KEY=...`, "
        "Then Restart This App."
    )

prompt = st.chat_input("Send A Message", disabled=not api_key)
if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                memory_context = ""
                if memory_runtime:
                    try:
                        memory_context = (
                            memory_runtime.runtime_build_memory_context(
                                prompt
                            ).text
                        )
                    except Exception as exc:
                        st.warning(f"Memory Recall Failed: {exc}")
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
            else:
                if show_memory_debug:
                    with st.expander("Retrieved Memory Context"):
                        st.code(
                            memory_context
                            or "No Memories Were Retrieved.",
                            language=None,
                        )
                st.markdown(answer)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "memory_context": memory_context,
                    }
                )
                if memory_runtime:
                    try:
                        memory_runtime.runtime_save_exchange_to_memory(
                            prompt, answer
                        )
                    except Exception as exc:
                        st.warning(
                            "Response Generated, But Memory Was Not Saved: "
                            f"{exc}"
                        )
