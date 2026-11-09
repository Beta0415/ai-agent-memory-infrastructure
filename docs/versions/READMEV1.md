# AI Agent Memory Infrastructure — V1

V1 adds LLM-based memory selection before embedding. Instead of embedding
every raw user and assistant message, the write path extracts atomic candidate
memories and stores only candidates approved by the selector.

The project uses Python 3.13 and is managed with
[uv](https://docs.astral.sh/uv/).

## Models

- Main chat model: `gpt-4o-mini`
- Memory selector: `gpt-5.6-luna`
- Embedding model: `text-embedding-3-small`
- Vector store: local ChromaDB

## Setup

1. Copy `.env.example` to `.env`.
2. Add an OpenAI API key:
  ```dotenv
   OPENAI_API_KEY=your-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   MEMORY_SELECTOR_MODEL=gpt-5.6-luna
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   CHROMA_PATH=.chroma
  ```
3. Install the locked dependencies:
  ```bash
   uv sync
  ```
4. Start the interface:
  ```bash
   uv run streamlit run src/ai_agent_memory_infrastructure/app.py
  ```

The interface is available at `http://localhost:8501`.

## Two Memory Paths

The architecture is divided into a Write Path and a Read Path:

```text
User message
  → Read Path
  → Main LLM response
  → Write Path
```



### Write Path

Information goes into memory. It answers:

> What from the current interaction should be remembered, and how should it
> be stored?

The current V1 Write Path contains:

1. A runtime trigger after a successful main-model response.
2. The public `memory_add(conversation)` operation.
3. GPT-5.6 Luna structured memory extraction.
4. Atomic candidate content generation.
5. Memory-type classification.
6. Importance and confidence scoring.
7. A selected or not-selected decision.
8. Filtering out the not-selected list.
9. Runtime-generated IDs, timestamps, session data, and metadata.
10. Content embedding with `text-embedding-3-small`.
11. Persistent storage in ChromaDB.



### Read Path

Information comes out of memory and is prepared for the main LLM. It answers:

> Given what the user is asking now, which existing memories should the model
> know about, and how should they be presented?

The current preliminary Read Path contains:

1. The current user message as the retrieval query.
2. Query embedding with `text-embedding-3-small`.
3. Chroma cosine-similarity search across all sessions.
4. Top-4 memory retrieval.
5. Reconstruction of complete memory records and similarity scores.
6. Formatting into compact memory-context lines.
7. Injection through the OpenAI Responses API `instructions` parameter.

V1 changes the Write Path. The Read Path remains unchanged from V0.

## V1 Write Path

The runtime decides when to write memory. The public `memory_add()` operation
decides what becomes memory:

```text
runtime_write_memory()
  → AgentMemory.memory_add()
      → OpenAIMemorySelector.extract_memory()
      → Structured selected and not-selected lists
      → Keep selected, discard not-selected
      → ChromaMemoryStore.write_memory()
          → Validation
          → Embedding
          → Metadata encoding
          → Chroma collection.add()
```

After every successful main-model response, the runtime passes the current
conversation to `memory_add()`. The memory selector sees raw transcript
messages in this form:

```json
[
  {"role": "user", "content": "I prefer concise answers."},
  {"role": "assistant", "content": "Understood."}
]
```

The selector returns structured output:

```json
{
  "selected": [
    {
      "content": "The user prefers concise answers.",
      "type": "preference",
      "importance": 0.8,
      "confidence": 0.99,
      "role": "user",
      "debug_source_quote": "I prefer concise answers.",
      "debug_reason": "This is a durable communication preference."
    }
  ],
  "not_selected": [
    {
      "content": "The assistant acknowledged the preference.",
      "type": "episode",
      "importance": 0.1,
      "confidence": 0.9,
      "role": "assistant",
      "debug_source_quote": "Understood.",
      "debug_reason": "A generic acknowledgement has no future value."
    }
  ]
}
```

Only `selected` candidates are embedded and persisted. `not_selected`
candidates and `debug_reason` remain transient and are shown in Memory Debug.

The selector audits the complete latest user/assistant turn. Every meaningful
statement goes into exactly one list. Non-durable content must appear in
`not_selected` rather than being omitted. Both lists are empty only when the
latest turn contains no meaningful content. Secret values are redacted from
rejected debug candidates.

### Selector Storage Policy

Importance measures future utility and consequence, not certainty:

- `0.0–0.2`: trivial or transient
- `0.3–0.4`: limited future usefulness
- `0.5–0.6`: useful recurring context
- `0.7–0.8`: stable preference, goal, project, decision, or constraint
- `0.9–1.0`: identity-critical fact, safety requirement, or persistent
  instruction

Confidence is separate: it measures how strongly the conversation supports
the candidate.

Typical information to store:

- Long-term preferences
- Projects
- Goals
- Decisions
- Constraints
- Important facts
- Repeated interests
- Long-running tasks
- Relationships between important entities
- Persistent instructions
- Confirmed corrections

Typical information not to store:

- Greetings
- Small talk
- Temporary statements
- One-off questions
- Information whose only relevance is already obvious from the current
  conversation and has no future value
- Random transient details
- Filler or generic assistant responses
- Speculation or unconfirmed claims
- Passwords, API keys, access tokens, credentials, or other secrets

## Memory Types

- `fact`: stable information presented as true
- `preference`: a user preference or recurring choice
- `goal`: an objective or desired future outcome
- `instruction`: persistent guidance for future behavior
- `episode`: something that happened at a specific point in time

Legacy V0 records with type `conversation` are interpreted as `episode`
records when read.

## Stored Memory Record

Only the candidate `content` is embedded. Python and the selector attach the
remaining fields to the stored record:

```json
{
  "id": "generated-memory-id",
  "content": "The user prefers concise answers.",
  "memory_type": "preference",
  "role": "user",
  "source": "memory_selector_v1",
  "session_id": "current-session-id",
  "importance": 0.8,
  "created_at": "generated-timestamp",
  "updated_at": "generated-timestamp",
  "metadata": {
    "confidence": 0.99,
    "debug_source_quote": "I prefer concise answers.",
    "exchange_id": "generated-exchange-id",
    "selector_model": "gpt-5.6-luna"
  }
}
```

IDs, timestamps, source, session ID, and exchange ID are generated by Python.
`debug_source_quote` and selector confidence are LLM-generated diagnostic
metadata. Rejected candidates are not persisted.

## Memory API

The public memory API is implemented by `AgentMemory`:

- `memory_add(conversation)`
- `memory_search(query, ...)`
- `memory_get(memory_id)`
- `memory_update(memory_id, ...)`
- `memory_delete(memory_id)`

Administrative operations:

- `count()`
- `clear()`

Internal write functions:

- `extract_memory()`: calls GPT-5.6 Luna and returns structured candidates
- `write_memory()`: validates, embeds, encodes, and inserts one approved record

The main chat model cannot call these functions. The Python runtime invokes
them automatically.

## Read Path

Retrieval is unchanged from V0:

1. Embed the current user prompt.
2. Search Chroma using cosine similarity.
3. Return up to four complete memory records.
4. Format the matches as memory context.
5. Pass the formatted context through the OpenAI Responses API
  `instructions` parameter.

The main model receives lines such as:

```text
- [preference; user; similarity=0.910] The user prefers concise answers.
```

There is currently no similarity threshold, reranking, deduplication,
summarization, or token-budget management.

## Sessions And Interface

- Streamlit keeps the active transcript in `st.session_state`.
- Each API request resends the active transcript; OpenAI does not retain it.
- `Clear Conversation` clears the visible transcript but keeps the session ID.
- `New Session` clears the transcript and generates a new session ID.
- Chroma memory is shared across sessions.
- `Clear Saved Memories` deletes all persisted Chroma memories.
- `Show Memory Debug` opens a panel beside the conversation. Each turn has
  three tabs: retrieved memories from the Read Path, candidates selected for
  storage by the Write Path, and candidates the selector rejected. Only the
  chosen tab is shown. The exact context supplied to the main model is under
  Retrieved.

The runtime searches across all sessions. Session IDs are stored as metadata
but are not currently used as retrieval filters.

## Local Data And Security

- `.env` contains the API key and is excluded from Git.
- `.chroma/` contains local memory data and is excluded from Git.
- `.venv/` contains reproducible local dependencies and is excluded from Git.
- `.env.example`, `pyproject.toml`, and `uv.lock` are safe to commit.

Never store passwords, API keys, access tokens, or other credentials as
memory. The selector is instructed to reject secrets, but deterministic secret
filtering can be added in a later version.

## Tests

Run:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover memory lifecycle operations, selector filtering, runtime
writing, retrieval context, and V0 `conversation`-to-`episode` compatibility.

## Current Limitations

- One additional selector-model call is made after each successful response.
- Candidate writes are not transactional.
- Semantic deduplication is not implemented.
- Selector confidence is diagnostic rather than guaranteed truth.
- The main model has no direct memory tools.
- Ordered transcripts are not persisted outside Streamlit session state.
