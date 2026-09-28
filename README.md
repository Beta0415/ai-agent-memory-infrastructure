# AI Agent Memory Infrastructure

Python 3.13 project managed with [uv](https://docs.astral.sh/uv/).

## Run the chat interface

1. Put your OpenAI API key in `.env`:

   ```dotenv
   OPENAI_API_KEY=your-api-key-here
   ```

2. Start Streamlit:

   ```bash
   uv run streamlit run src/ai_agent_memory_infrastructure/app.py
   ```

The visible conversation is kept in Streamlit session state. Each successful
user/assistant exchange is also embedded with `text-embedding-3-small` and
stored locally in ChromaDB under `.chroma/`. Before each response, the app
retrieves up to four semantically relevant memories.

Both `.env` and `.chroma/` are excluded from Git.

## Memory API

The core API in `memory_store.py` provides:

- `memory_add`
- `memory_search`
- `memory_get`
- `memory_update`
- `memory_delete`

Each memory stores its ID, original content, type, role, source, session ID,
importance, creation/update timestamps, and optional metadata. Search results
also include a cosine similarity score.

Supported types are `conversation`, `fact`, `preference`, `goal`, and
`instruction`.

`MemoryRuntime.runtime_build_memory_context()` automatically builds retrieved
context before each response, and
`MemoryRuntime.runtime_save_exchange_to_memory()` saves the resulting exchange
afterward. These functions are not exposed to the language model as callable
tools yet.

“Clear Conversation” removes the active OpenAI transcript but retains the
current session ID. “New Session” clears that transcript and creates a new
session ID. Both controls preserve the Chroma memory store.

Enable “Show Memory Debug” in the sidebar to display the exact retrieved
memory text inserted into the OpenAI instructions for each assistant response.
