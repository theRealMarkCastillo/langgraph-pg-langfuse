# LangGraph + Postgres + Langfuse demos

[![CI](https://github.com/theRealMarkCastillo/langgraph-pg-langfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/theRealMarkCastillo/langgraph-pg-langfuse/actions/workflows/ci.yml)

A set of small, runnable LangGraph (Python) demos that all:

- persist graph state to **PostgreSQL** (`langgraph-checkpoint-postgres`, plus `PostgresStore` for long-term memory, with optional pgvector semantic search)
- send traces to a **self-hosted Langfuse** stack via the Langfuse Python SDK
- call any **OpenAI-compatible** endpoint, or run fully **offline** with a scripted model

| Demo | Shows |
| --- | --- |
| [`demos.chat`](demos/chat.py) | Thread state in Postgres, conditional routing, token streaming, **time travel / forking** |
| [`demos.support_agent`](demos/support_agent.py) | **Human-in-the-loop**: tool-calling agent pauses for refund approval across processes; approve / reject / edit; fork a past decision; decision recorded as a **Langfuse score** |
| [`demos.durable`](demos/durable.py) | **Durable execution**: `RetryPolicy` on a flaky step, a hard crash mid-pipeline, resume without re-running finished steps |
| [`demos.memory`](demos/memory.py) | **Long-term memory** across threads: `PostgresStore`, scoped per user via runtime context; recalls all memories, or the most relevant ones via **pgvector** when an embedding model is set |
| [`demos.summarize`](demos/summarize.py) | **Long conversations**: fold old messages into a summary and delete them with `RemoveMessage` |
| [`demos.fanout`](demos/fanout.py) | **Parallel map-reduce** with the `Send` API and a reducer |
| [`demos.multi_agent`](demos/multi_agent.py) | **Supervisor + specialist subgraphs** using `Command(goto=...)`, subgraph streaming |
| [`demos.prompts`](demos/prompts.py) | **Langfuse prompt management**: system prompt fetched from Langfuse, versions linked to generations |
| [`demos.experiments`](demos/experiments.py) | **Langfuse datasets + experiments**: score the multi-agent router on labelled examples |

**Presenting?** Follow [RUNBOOK.md](RUNBOOK.md), a timed demo script with preflight
(`scripts/preflight.sh`), reset (`scripts/reset.sh`), talk track and Langfuse click paths.

## Quickstart

**Prerequisites:** Docker with Compose v2, [uv](https://docs.astral.sh/uv/), and about 4 GB of free RAM
for the Langfuse stack. uv installs Python 3.12 for you if needed.

```bash
git clone https://github.com/theRealMarkCastillo/langgraph-pg-langfuse.git && cd langgraph-pg-langfuse
```

```bash
cp .env.example .env
```

```bash
docker compose -f langfuse/docker-compose.yml up -d && docker compose up -d --wait
```

```bash
uv run python -m demos.support_agent -t t1 "Order A100 arrived broken, I want a refund"
```

That runs offline. To use a real model, set `OPENAI_BASE_URL`, `OPENAI_MODEL` and (if needed)
`OPENAI_API_KEY` in `.env`; see [LLM configuration](#llm-configuration).
`./scripts/preflight.sh` checks that everything is up.

## Layout

| Path | What |
| --- | --- |
| `docker-compose.yml` | Postgres 18 + pgvector for LangGraph state and memory, on `localhost:5433` |
| `langfuse/docker-compose.yml` | Official Langfuse v4 compose file (unmodified), web UI on `localhost:3000` |
| `langfuse/.env` | Headless init: creates org, project `langgraph-demo`, API keys and a login user |
| `.env.example` | Template for `.env` (git-ignored): Postgres URI, Langfuse keys/URL, LLM endpoint |
| `demos/common.py` | Shared wiring: checkpointer, store, LLM/embeddings factory, Langfuse tracing, stream printer |
| `scripts/` | `preflight.sh`, `reset.sh` and `smoke_test.sh` (the offline end-to-end test CI runs) |

## Langfuse UI

Langfuse UI: <http://localhost:3000>, login `demo@example.com` / `demo-password-123`, project **LangGraph Demo**.
Each demo run groups its traces into a Langfuse **session** named after the thread id.

## LLM configuration

Set these in `.env` to use a real model; leave `OPENAI_BASE_URL` unset to run offline.

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `OPENAI_BASE_URL` | yes | — | e.g. `http://localhost:11434/v1` (Ollama), vLLM, LiteLLM, OpenAI |
| `OPENAI_MODEL` | yes | — | model name the endpoint expects; needs tool calling for `support_agent` |
| `OPENAI_API_KEY` | no | `not-needed` | many local servers ignore it |
| `OPENAI_TEMPERATURE` | no | `0.2` | |
| `OPENAI_MAX_TOKENS` | no | `300` | |
| `OPENAI_TIMEOUT` | no | `60` | seconds |
| `OPENAI_MAX_RETRIES` | no | `2` | |
| `OPENAI_EMBEDDING_MODEL` | no | — | turns on pgvector semantic recall in `demos.memory` |
| `OPENAI_EMBEDDING_BASE_URL` | no | `OPENAI_BASE_URL` | embeddings can use another provider than chat |
| `OPENAI_EMBEDDING_API_KEY` | no | `OPENAI_API_KEY` | |
| `OPENAI_EMBEDDING_DIMS` | no | auto-detected | vector size; only needed if the probe call is unwanted |

**Offline mode:** each demo passes a small deterministic script to `ScriptedChatModel`
([`demos/common.py`](demos/common.py)) that decides replies and tool calls, so every flow
(including human-in-the-loop) runs without an API and still produces realistic Langfuse traces.
Its answers are canned; it shows the *mechanics*, not model quality.

To run offline while `.env` points at an endpoint, override it per command:

```bash
OPENAI_BASE_URL= uv run python -m demos.chat
```

## Walkthroughs

### Human-in-the-loop (`demos.support_agent`)

```bash
uv run python -m demos.support_agent -t t1 "Order A100 arrived broken, I want a refund"
```

The agent calls `lookup_order` and `refund_policy` on its own, then proposes `issue_refund`.
That routes to `human_approval`, which calls `interrupt()`: the pending approval is saved in
Postgres and **the process exits**. Later, from any process:

```bash
uv run python -m demos.support_agent -t t1 --status
```

```bash
uv run python -m demos.support_agent -t t1 --resume approve
```

Or `--resume reject --reason "outside policy"`, or `--resume edit --amount 20`.
The resume **joins the original Langfuse trace** (its id is stored in the interrupt payload),
and the decision is saved as a categorical `human_approval` score on that trace.

Re-decide an old pause (the checkpoint id is printed when it pauses). This creates a new branch,
and the original outcome is kept:

```bash
uv run python -m demos.support_agent -t t1 --resume reject --checkpoint <checkpoint_id>
```

### Time travel (`demos.chat`)

```bash
uv run python -m demos.chat -t c1
```

```bash
uv run python -m demos.chat -t c1 --history
```

```bash
uv run python -m demos.chat -t c1 --fork <checkpoint_id> "Actually, call me Bob. What's my name?"
```

### Durable execution (`demos.durable`)

```bash
uv run python -m demos.durable -t job1 --crash-at summarize
```

```bash
uv run python -m demos.durable -t job1 --status
```

```bash
uv run python -m demos.durable -t job1 --resume
```

`enrich` fails twice and is retried by its `RetryPolicy`. The crash kills the process with
`os._exit`; `extract` and `enrich` were already checkpointed (`durability="sync"`), so the
resume runs only `summarize` and `publish`. Like any real crash, the killed run's
not-yet-flushed spans never reach Langfuse; only the resumed run is traced.

### Long-term memory (`demos.memory`)

```bash
uv run python -m demos.memory -u ada -t monday "I prefer email over phone calls. I live in Lisbon."
```

```bash
uv run python -m demos.memory -u ada -t tuesday "How should you contact me about my order?"
```

```bash
uv run python -m demos.memory -u ada --list
```

A different thread recalls Ada's preference; another user (`-u bob`) sees nothing.

**Embeddings are optional.** Without `OPENAI_EMBEDDING_MODEL`, recall loads all of the user's
memories (fine for the handful a user typically has, and works with any chat model). With it,
`put()` embeds each memory into pgvector and recall returns the three most relevant by cosine
similarity, printing the scores. That works even when chat runs on a provider without an
embeddings API, for example:

```bash
OPENAI_EMBEDDING_MODEL=text-embedding-3-small OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1 \
OPENAI_EMBEDDING_API_KEY=sk-... uv run python -m demos.memory -u ada -t tuesday "Where do I live?"
```

pgvector columns have a fixed size, so switching embedding models (or turning embeddings on
after memories were stored with a different model) stops with a message to run `./scripts/reset.sh`.

### The rest

```bash
uv run python -m demos.summarize -t long1
```

```bash
uv run python -m demos.fanout "home espresso"
```

```bash
uv run python -m demos.multi_agent "I was charged twice and the app crashes when I log in"
```

```bash
uv run python -m demos.prompts "Do you ship to Canada?"
```

```bash
uv run python -m demos.prompts --publish "You are a pirate-themed assistant for {{company}}. Tone: {{tone}}."
```

```bash
uv run python -m demos.experiments --run-name my-model-v1
```

`demos.prompts` creates prompt `support-assistant` v1 on first run; `--publish` makes a new
version the `production` one, and `--version N` pins an older one. `demos.experiments` upserts
dataset `support-routing`; compare runs under **Datasets → support-routing** in Langfuse.
The offline router is keyword-based and misses the "card was declined" item on purpose, so the
experiment shows a failing case.

## How the wiring works

- **State:** `PostgresSaver` checkpoints after every step, keyed by `config["configurable"]["thread_id"]`.
  Subgraphs inherit it and get their own `checkpoint_ns`.
- **Memory:** nodes reach the `PostgresStore` as `runtime.store`, and the user id arrives via `context=`.
  With an embedding model, the store gets `index={"dims", "embed", "fields": ["text"]}`: `put()` embeds
  and `search(query=...)` ranks by pgvector cosine similarity (HNSW index). Without one, it's a plain
  key-value store.
- **Tracing:** `traced()` in `common.py` opens a Langfuse root span (or continues a trace id) inside
  `propagate_attributes(session_id=..., user_id=..., tags=...)`, and yields a LangChain
  `CallbackHandler` for the graph config. Every node, LLM call and tool call nests under it.
  `flush()` runs before exit because the scripts are short-lived.
- **Streaming:** `stream_run()` uses `stream_mode=["updates", "messages"]`. LLM calls tagged
  `nostream` (routers, extractors, parallel workers) are traced but not printed.

## Inspect state directly

```bash
docker compose exec postgres psql -U langgraph -c "select thread_id, checkpoint_ns, count(*) from checkpoints group by 1, 2"
```

```bash
docker compose exec postgres psql -U langgraph -c "select prefix, value from store"
```

## Tear down

```bash
docker compose down -v
```

```bash
docker compose -f langfuse/docker-compose.yml down -v
```

All credentials and keys here are local demo values. Change the `# CHANGEME` values in
`langfuse/docker-compose.yml` before exposing anything beyond localhost.

## Development

```bash
uv sync
```

```bash
uv run ruff check . && uv run ruff format --check .
```

```bash
./scripts/smoke_test.sh
```

The smoke test runs every demo offline against the LangGraph Postgres, with tracing disabled, and
checks the outcomes (pause and resume, forks, crash recovery, memory isolation and so on). CI runs the
same test plus the lint on every push and pull request. Issues and pull requests are welcome; please
keep each demo self-contained and runnable offline.

## License

[MIT](LICENSE). `langfuse/docker-compose.yml` is © ClickHouse, Inc. (Langfuse), MIT-licensed;
see [langfuse/README.md](langfuse/README.md).
