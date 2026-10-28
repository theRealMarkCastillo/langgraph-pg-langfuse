# Demo runbook

A presenter's script for the LangGraph + Postgres + Langfuse demos: about **35 minutes**
for the full run, or the **15-minute** core path (Acts 1, 2, 3). Commands run from the repo root.
For what each demo does internally, see [README.md](README.md).

## Before the audience arrives (T-15 min)

1. **Start the infrastructure** (skip if already running):

   ```bash
   docker compose -f langfuse/docker-compose.yml up -d
   ```

   ```bash
   docker compose up -d --wait
   ```

2. **Reset to a clean slate.** `--all` also wipes Langfuse, so the prompt starts at v1 and
   the dashboards are empty. It takes about a minute.

   ```bash
   ./scripts/reset.sh --all
   ```

3. **Preflight.** Every line should be ✔:

   ```bash
   ./scripts/preflight.sh
   ```

   If the LLM line fails and you can't fix the endpoint, present **offline**: prefix every
   command with `OPENAI_BASE_URL=`. Every flow still works, with canned answers.

4. **Warm up the model** (the first call can be slow), then reset again so the run starts clean:

   ```bash
   uv run python -m demos.chat -t warmup "hello"
   ```

   ```bash
   ./scripts/reset.sh
   ```

5. **Set up the screen:** a terminal with large font on the left, and a browser on the right
   at <http://localhost:3000>, logged in as `demo@example.com` / `demo-password-123`, project
   **LangGraph Demo**. Keep a second terminal tab open for `psql`.

> Thread ids below (`t1`, `job1`, ...) assume a fresh reset. Re-running an act needs a new id
> or another `./scripts/reset.sh`.

---

## Act 1: State lives in Postgres (4 min)

**Point:** LangGraph saves a checkpoint after every step, so a conversation survives the process.

```bash
uv run python -m demos.chat -t t1
```

- Three turns run, with tokens streaming. In turn 3 it answers "What's my name?" from the stored history.
- **Show Postgres:**

  ```bash
  docker compose exec postgres psql -U langgraph -c "select thread_id, count(*) from checkpoints group by 1"
  ```

- **Time travel.** List the checkpoints, copy the `checkpoint_id` at `step=3` (just after turn 1), and branch from it:

  ```bash
  uv run python -m demos.chat -t t1 --history
  ```

  ```bash
  uv run python -m demos.chat -t t1 --fork <checkpoint_id> "Actually, call me Bob. What's my name?"
  ```

  Say: "We rewound to after the first message and took a different path. The original history is still in the database."

**Langfuse:** **Sessions** → `t1`. One session holds every turn, rendered as a chat transcript.
Open a trace: the **Tree** tab shows `classify` → `answer` and the LLM generation with its input
and output, and the **Graph** tab draws the LangGraph structure that ran.

## Act 2: Human-in-the-loop (8 min), the centrepiece

**Point:** An agent can pause for a human, wait in the database for hours, and resume from a different process.

```bash
uv run python -m demos.support_agent -t t2 "Order A100 arrived broken, I want a refund"
```

- The agent calls `lookup_order` and `refund_policy` on its own, then proposes `issue_refund` and **stops**.
  Point at "⏸ Waiting for a human" and the **process has exited**.
- Show that the pending request is durable:

  ```bash
  uv run python -m demos.support_agent -t t2 --status
  ```

- Resume as the human. Pick one:

  ```bash
  uv run python -m demos.support_agent -t t2 --resume approve
  ```

  ```bash
  uv run python -m demos.support_agent -t t2 --resume edit --amount 50
  ```

- **Re-decide the past:** copy the "paused at checkpoint" id from the first command and reject it.
  This makes a new branch; the approved outcome still exists.

  ```bash
  uv run python -m demos.support_agent -t t2 --resume reject --reason "needs a photo of the damage" --checkpoint <checkpoint_id>
  ```

**Langfuse:**
- **Sessions** → `t2`: the original run and every resume sit in **one trace**. The resume joined
  the trace whose id was stored with the interrupt.
- Open the trace: its header carries a `human_approval: …` badge, and the **Scores** tab lists
  each decision (approve / edit / reject) with your reason as the comment. In the **Tree**, each
  resume is a second `support-agent` root under the same trace.
- **Scores** in the sidebar: filter on `human_approval` to see every human decision across runs.

## Act 3: Crash recovery (4 min)

**Point:** If the process dies, finished work isn't repeated.

```bash
uv run python -m demos.durable -t job1 --crash-at summarize
```

- `enrich` fails twice ("pricing API timed out") and the `RetryPolicy` recovers it. Then 💥 the process is killed.

```bash
uv run python -m demos.durable -t job1 --status
```

```bash
uv run python -m demos.durable -t job1 --resume
```

- Only `summarize` and `publish` run. Point at the step log: `enrich: ok after 3 attempts`
  appears once, so it wasn't redone.

**Langfuse:** **Tracing**, filter on name `durable-pipeline`. Only the resumed run is there, because the
killed process died before sending its spans. Say: "Postgres kept the work; tracing lost the
crashed process, just as it would in production."

## Act 4: Long-term memory (4 min)

**Point:** Checkpoints are per conversation; the store is per user, across all of their conversations.

```bash
uv run python -m demos.memory -u ada -t monday "I prefer email over phone calls. I live in Lisbon."
```

```bash
uv run python -m demos.memory -u ada -t tuesday "How should you contact me about my order?"
```

- A new thread recalls "prefers email". The first line says which recall mode is active:
  **all memories** by default, or **semantic search (pgvector)** with similarity scores when
  `OPENAI_EMBEDDING_MODEL` is set. If you have an embedding endpoint, use semantic mode and point
  at the scores: the relevant memory ranks first.
- Show the isolation between users:

  ```bash
  uv run python -m demos.memory -u bob -t wed "How should you contact me about my order?"
  ```

  ```bash
  uv run python -m demos.memory -u ada --list
  ```

**Langfuse:** **Users** → `ada`: every trace for this user, across threads.

## Act 5: Agent patterns (5 min)

```bash
uv run python -m demos.multi_agent "I was charged twice and the app crashes when I log in"
```

- The supervisor routes to **billing**, then **tech**. Indented lines come from inside each specialist's subgraph.

```bash
uv run python -m demos.fanout "home espresso"
```

- Three research branches run in parallel. Point at the last line: the research phase took about
  as long as the *slowest* branch, while the *combined* work is roughly three times that.

```bash
uv run python -m demos.summarize -t long1
```

- Watch "messages in state" drop from 6 to 2 once older turns fold into the summary.

**Langfuse:** open a `fanout-research` trace. On the **Timeline** tab, the three `research` spans overlap.

## Act 6: Langfuse beyond tracing (6 min)

**Prompt management.** The system prompt lives in Langfuse, not in the code.

```bash
uv run python -m demos.prompts "Do you ship to Canada?"
```

```bash
uv run python -m demos.prompts --publish "You are a pirate-themed assistant for {{company}}. Tone: {{tone}}." "Do you ship to Canada?"
```

```bash
uv run python -m demos.prompts --version 1 "Do you ship to Canada?"
```

- The behaviour changes with no code change, and `--version 1` pins the old version.
- **Langfuse:** **Prompts** → `support-assistant`: versions, labels, and the generations linked to each version.

**Experiments.** Evaluate the router on a labelled dataset.

Run a keyword-only baseline (offline), then the real model:

```bash
OPENAI_BASE_URL= uv run python -m demos.experiments --run-name keyword-baseline
```

```bash
uv run python -m demos.experiments --run-name live-model
```

- The terminal prints accuracy and any misses (✘). The keyword baseline misses "My card was
  declined at checkout" (about 89%); a capable LLM typically gets all of them.
- **Langfuse:** **Datasets** → `support-routing` → **Runs**: compare the two runs side by side,
  then click into the failing item's trace. The same flow works for comparing models or prompt versions.

## Wrap-up talking points (1 min)

- **Postgres** stores short-term state (checkpoints), human-in-the-loop pauses and long-term memory: one durable, queryable database.
- **Langfuse** provides traces grouped by session and user, human decisions as scores, versioned prompts and experiments.
- The model is swappable through `OPENAI_BASE_URL`; the whole demo also runs offline.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Connection error` / `OpenAIConnectionError` | Endpoint down or wrong `OPENAI_BASE_URL`. Run preflight, or present offline with `OPENAI_BASE_URL=` |
| `Thread 'job1' already ran` | Use a new `-t` id, or `./scripts/reset.sh` |
| `No checkpoint '...' in thread` | Typo in the copied id; rerun with `--history` (chat) or `--status` (support agent) |
| `Nothing is waiting for approval` | The thread was already resumed. Start a new thread, or use `--checkpoint` to re-decide an old pause |
| Traces don't appear in Langfuse | Wait 5-10 s and refresh; ingestion is asynchronous. Check that `langfuse-langfuse-worker-1` is running (`docker compose -f langfuse/docker-compose.yml ps`) |
| Pirate replies when you expect normal ones | v2 is still `production`. Use `--version 1`, publish the original text again, or `./scripts/reset.sh --all` |
| Memory demo stops with "Stored memories use vector(N)…" | The embedding model changed since memories were stored. Run `./scripts/reset.sh` |
| Embedding errors while chat works | Some chat providers have no embeddings API. Set `OPENAI_EMBEDDING_BASE_URL`/`_API_KEY` to one that does, or unset `OPENAI_EMBEDDING_MODEL` |
| Langfuse login fails after `reset.sh --all` | It's still booting; wait about 30 s. The user is recreated from `langfuse/.env` |
| Agent never calls tools (online) | The model or endpoint doesn't support tool calling. Pick a tool-capable model |

## After the demo

```bash
docker compose down
```

```bash
docker compose -f langfuse/docker-compose.yml down
```

Add `-v` to either command to delete its data as well.
