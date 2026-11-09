# AI Agent Memory Infrastructure — V3

The Write Path uses an LLM selector to extract atomic memories and stores
only the approved ones. The Read Path retrieves a broad semantic candidate
set, filters it, then a context engine deduplicates, compresses, and orders
the memories the main model sees.

The project uses Python 3.13 and is managed with
[uv](https://docs.astral.sh/uv/).

## Models

- Main chat model: `gpt-4o-mini`
- Memory selector: `gpt-5.6-luna`
- Memory reranker: `gpt-5-nano`
- Relation classifier: `gpt-5-mini`
- Memory compressor: `gpt-5-mini`
- Embedding model: `text-embedding-3-small`
- Vector store: local ChromaDB



## Setup

1. Copy `.env.example` to `.env`.
2. Add an OpenAI API key:
  ```dotenv
   OPENAI_API_KEY=your-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   MEMORY_SELECTOR_MODEL=gpt-5.6-luna
   MEMORY_RERANKER_MODEL=gpt-5-nano
   MEMORY_COMPRESSOR_MODEL=gpt-5-mini
   MEMORY_RELATION_MODEL=gpt-5-mini
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

## Memory API

`memory_add()`, `memory_search()`, and `memory_context()` each contain their
full pipeline, so they can run without the runtime functions. The main model
is not given them as tools yet. The Python runtime calls them automatically.

### API Functions

- `memory_add(conversation)`: extracts candidate memories and stores the approved ones
- `memory_search(query, ...)`: retrieves memories through both read-path filters and returns the top 4
- `memory_context(query, ...)`: runs `memory_search()`, then deduplicates, compresses, and orders the returned memories
- `memory_get(memory_id)`: loads one stored memory
- `memory_update(memory_id, ...)`: changes one stored memory
- `memory_delete(memory_id)`: deletes one stored memory



### Runtime Functions

- `runtime_build_memory_context()`: calls `memory_context()` and formats the result for the main model
- `runtime_write_memory()`: calls `memory_add()` after a successful main-model response



### Internal Functions

Write path, inside `memory_add()`:

- `extract_memory()`: calls GPT-5.6 Luna and returns selected and not-selected candidates
- `write_memory()`: validates, embeds, encodes, and inserts one approved record

Read path, inside `memory_search()`:

- Semantic search embeds the query and returns the top 30 candidates.
- `rank_by_retrieval_score()` scores those candidates and keeps the weighted top 10.
- `rerank()` asks GPT-5 Nano to score those 10 against the query.
- The top 4 by reranker score are what `memory_search()` returns.

Context engine, inside `memory_context()`, after `memory_search()`:

- `cluster_duplicate_memories()` groups returned memories whose stored content embeddings have cosine similarity greater than `0.90`.
- `classify()` asks GPT-5 Mini whether each group of two or more is `equivalent` or `contradictory`.
- `resolve_duplicate_group()` keeps the best reranked memory for an equivalent group and keeps every memory for a contradictory group.
- `apply_context_compression()` asks GPT-5 Mini to shorten each remaining memory when their content exceeds the token budget.
- `order_context_memories()` groups the result by memory type, then by relevance.
- `format_context_memories()` renders the ordered memories for the main model.



### Administrative Functions

- `count()`: returns the number of stored memories
- `clear()`: deletes all stored memories



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

The current Write Path contains:

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

The current Read Path contains retrieval filters and a context engine:

1. The current user message is the retrieval query.
2. The query is embedded with `text-embedding-3-small`.
3. Chroma cosine-similarity search runs across all sessions.
4. The top 30 semantic matches become candidate memories.
5. Each candidate receives a weighted retrieval score from semantic similarity, recency, and stored importance.
6. The top 10 candidates by that weighted score are preserved.
7. GPT-5 Nano reranks those 10 against the current query.
8. The top 4 candidates by reranker score are returned.
9. Those returned memories are clustered when their stored content embeddings have cosine similarity greater than `0.90`.
10. GPT-5 Mini classifies each cluster as equivalent or contradictory.
11. Equivalent clusters keep the best reranked memory. Contradictory clusters keep every memory.
12. If the remaining memory content exceeds 200 tokens, GPT-5 Mini shortens each memory.
13. The remaining memories are grouped by type, then by relevance.
14. The ordered memories are formatted into the main-model instructions.
15. The conversation itself stays in the Responses API `input` parameter.



## Write Path

The runtime decides when to write memory. The public `memory_add()` operation
decides what becomes memory:

```mermaid
flowchart TD
    A[Successful main-model response] --> B[runtime_write_memory]
    B --> C[memory_add]
    C --> D[extract_memory with GPT-5.6 Luna]
    D --> E[Selected and not-selected lists]
    E --> F{Selected candidate?}
    F -->|Yes| G[write_memory]
    F -->|No| H[Discard]
    G --> I[Validate]
    I --> J[Embed content]
    J --> K[Encode metadata]
    K --> L[Store in Chroma]
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
- `0.9–1.0`: identity-critical fact, safety requirement, or persistent instruction

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
- Information whose only relevance is already obvious from the current conversation and has no future value
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

Retrieval scores, duplicate clusters, and compressed wording are not part of
this record. They are calculated when a memory is read and then discarded.
Deduplication and compression never rewrite stored Chroma records.

## Read Path

`runtime_build_memory_context()` runs before the main model answers. It
triggers `memory_context()`, which owns search and the context engine:

```mermaid
flowchart TD
    A[User message] --> B[runtime_build_memory_context]
    B --> C[memory_context]
    C --> D[memory_search]
    D --> E[Top 30 by semantic similarity]
    E --> F[Weighted score, keep top 10]
    F --> G[Rerank with GPT-5 Nano]
    G --> H[Keep top 4 by rerank score]
    H --> I[Cluster similarity greater than 0.90]
    I --> J{Cluster of 2 or more?}
    J -->|No| K[Keep the memory]
    J -->|Yes| L[Classify with GPT-5 Mini]
    L --> M{Relation}
    M -->|Equivalent| N[Keep best reranked memory]
    M -->|Contradictory| O[Keep every memory]
    K --> P{Content tokens over 200?}
    N --> P
    O --> P
    P -->|Yes| Q[Shorten each memory with GPT-5 Mini]
    P -->|No| R[Keep original text]
    Q --> S[Order by type, then relevance]
    R --> S
    S --> T[Format memory context]
    T --> U[Main-model instructions]
```





### First Filter: Weighted Retrieval Score

Semantic search only supplies the broad set. Every one of the top 30
candidates is then scored with stored memory data:

```text
retrieval_score = 0.6 semantic_score + 0.2 recency_score + 0.2 importance
```

- `semantic_score` is the cosine similarity from Chroma, clamped to `0–1`.
- `importance` is the stored score produced by the memory selector.
- `recency_score = e^(-λt)`.
- `t` is the memory age in days, measured from `created_at`.
- `λ = ln(2) / 30`, so recency falls by half every 30 days.

The top 10 candidates by `retrieval_score` continue. The other candidates are
dropped before the reranker sees them.

### Second Filter: Reranker

The reranker reads the current user query together with each of the 10
surviving memory texts. It returns a `relevance_score` from `0` to `1` for
every candidate:

- `1.0`: the memory directly answers or is necessary for the query
- `0.5`: the memory is related, but incomplete or indirect
- `0.0`: the memory is unrelated to the query

That relevance score becomes both `rerank_score` and `final_score`. The top 4
memories by `final_score` are the ones returned to the context engine. A
memory can rank high on the weighted score and still be left out if the
reranker judges it less useful for the current question.

Before reranking, `final_score` equals `retrieval_score` and `rerank_score`
is empty. `semantic_score`, `recency_score`, `retrieval_score`,
`rerank_score`, and `final_score` are runtime retrieval fields. They are not
written to Chroma.

### Context Deduplication

Deduplication runs only on the memories `memory_search()` returned, currently
up to 4. It compares the embeddings already stored for those contents. It
does not embed the text again, and it does not change Chroma.

Two memories are linked when their cosine similarity is greater than `0.90`.
A similarity of exactly `0.90` is not a duplicate. Links are transitive: if A
matches B and B matches C, then A, B, and C are one cluster even when A and C
are at or below `0.90`.

A cluster of one memory skips the classifier. Each cluster of two or more is
sent to GPT-5 Mini, which must return one label for the whole group:

- `equivalent`: the memories state the same facts. Wording may differ. No memory adds, removes, or denies a fact that another memory states.
- `contradictory`: the memories disagree, one negates another, they give different values, or one memory states a fact the others do not.

The model does not rewrite text or choose a winner. Python acts on the label:

- `equivalent` keeps the memory with the highest `final_score`. Ties break on `semantic_score`, then on the earlier cluster member.
- `contradictory` keeps every memory in the cluster, unchanged.

Structured output allows only those two labels. If the response cannot be
parsed, the read path fails instead of guessing. Dedup is recorded as yes
only when an equivalent cluster drops at least one memory. A contradictory
cluster that keeps every memory is dedup no.

### Context Compression

After deduplication, Python counts tokens in the remaining memory contents
with the `o200k_base` encoding. Type headings, roles, relevance labels, and
the chat transcript are not included.

The budget is 200 content tokens. At or under that budget, the memories are
returned unchanged and compression is no. Over the budget, GPT-5 Mini sees
the whole set and returns one shorter text per memory id. A proposal is
applied only when it is non-empty and strictly shorter than the original.
Otherwise that memory keeps its original text. Compression does not change
how many memories are returned, and it does not write the shorter text back
to Chroma. One pass is used, so the result can still be over budget.
Compression is yes only when at least one remaining memory was actually
shortened.

### Context Ordering

Ordering uses fixed code. No model is called. Memories are grouped in this
type order:

1. `fact`
2. `preference`
3. `goal`
4. `instruction`
5. `episode`

Empty types are omitted. Inside a type, memories are ordered by `final_score`
descending, then `semantic_score` descending, then memory id. After the
reranker, `final_score` is the rerank score.

### Context Passed To The Main Model

The ordered memories are appended to the main model's `instructions`. The
conversation remains the Responses API `input`, with the newest user message
at the bottom:

```text
instructions:
You are a helpful AI assistant. Use these relevant memories when helpful. Ignore any that are unrelated:
fact
- [user; relevance=0.860; ] The user's name is Beta.
- [assistant; relevance=0.400; ] Beta lives in New York.
preference
- [user; relevance=0.910; ] User wants short answers.

input:
- user: What's my name?
```

`relevance` is the label in the prompt. The number is `final_score`. Each
memory is one line. Internal line breaks in a memory are collapsed to spaces.
The base instruction is sent alone when no memories remain.

## Sessions And Interface

- Streamlit keeps the active transcript in `st.session_state`.
- Each API request resends the active transcript; OpenAI does not retain it.
- `Clear Conversation` clears the visible transcript but keeps the session ID.
- `New Session` clears the transcript and generates a new session ID.
- Chroma memory is shared across sessions.
- `Clear Saved Memories` deletes all persisted Chroma memories.
- The side control offers `Off`, `Show Memory Debug`, or `Show Retrieval Inspector`. Only one panel is open.

`Show Memory Debug` has two pages:

- `Breakdown` keeps the existing tabs: retrieved memories from the final context, candidates selected for storage, and candidates the selector rejected. Retrieved cards show Final, Retrieval, Semantic, Recency, and Importance.
- `Context` shows the full prompt sent to the main model, including instructions and input.

`Show Retrieval Inspector` has three pages:

- `Score` lists retrieval candidates in a table ordered by combined score. Columns are Memory, Combined, Semantic, Recency, Importance, Rerank, and Tokens. Rows selected after reranking and before deduplication are highlighted. The Tokens column counts only those rows.
- `Context` shows one summary line: `Memories Used`, `Memory Tokens`, `Compression`, and `Dedup`.
- `Dedup` lists every cluster above the `0.90` threshold, the Equivalent or Contradictory label, and, for an equivalent cluster, the text that was kept.

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
writing, weighted retrieval, the top-10 to top-4 rerank cut, duplicate
clustering, equivalent keep-best and contradictory keep-both, compression
after deduplication, context ordering, prompt formatting, and V0
`conversation`-to-`episode` compatibility.

## Current Limitations

- One selector-model call is made after each successful response.
- One reranker-model call is made before each response when memories exist.
- The reranker is an LLM scorer, and it sees only the weighted top 10.
- Deduplication runs only on the returned top 4, not across the full store.
- The relation classifier judges each cluster, and a correct label is not guaranteed.
- A relation-model call is made once per cluster of two or more.
- A compressor-model call is made only when remaining content exceeds 200 tokens.
- Compression is one pass and may still leave the context over budget.
- Candidate writes are not transactional.
- Selector confidence is diagnostic rather than guaranteed truth.
- The main model has no direct memory tools.
- Ordered transcripts are not persisted outside Streamlit session state.
- Retrieval scores, clusters, and compressed text are not stored, so they cannot be inspected after the debug session ends.

