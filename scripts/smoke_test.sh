#!/usr/bin/env bash
# Offline end-to-end check of every demo that doesn't need the Langfuse API.
# Needs only the LangGraph Postgres (docker compose up -d --wait). Used by CI; safe to run locally.
set -euo pipefail
cd "$(dirname "$0")/.."

export OPENAI_BASE_URL=          # scripted offline model
export LANGFUSE_TRACING_ENABLED=false
export POSTGRES_URI=${POSTGRES_URI:-postgresql://langgraph:langgraph@localhost:5433/langgraph?sslmode=disable}
export LANGFUSE_BASE_URL=${LANGFUSE_BASE_URL:-http://localhost:3000}
RUN=smoke-$(date +%s)-$RANDOM    # unique thread ids, so reruns don't collide

pass=0
expect() {  # expect <description> <pattern> <command...>
  local desc=$1 pattern=$2; shift 2
  local out
  if ! out=$("$@" 2>&1); then
    if [[ "$desc" != *"(crash expected)"* ]]; then
      echo "✘ $desc: command failed"; echo "$out" | tail -20; exit 1
    fi
  fi
  if grep -qE -- "$pattern" <<<"$out"; then
    echo "✔ $desc"; pass=$((pass + 1))
  else
    echo "✘ $desc: expected /$pattern/"; echo "$out" | tail -20; exit 1
  fi
}
demo() { uv run --frozen python -m "demos.$1" "${@:2}"; }

expect "chat remembers across turns" "your name is Ada" demo chat -t "$RUN-chat"
CP=$(demo chat -t "$RUN-chat" --history | awk '$2=="3"{print $4}' | cut -d= -f2)
expect "chat forks from an earlier checkpoint" "your name is Bob" \
  demo chat -t "$RUN-chat" --fork "$CP" "Actually, call me Bob. What's my name?"

expect "support agent pauses for approval" "Waiting for a human" \
  demo support_agent -t "$RUN-sup" "Order A100 arrived broken, I want a refund"
PAUSED=$(demo support_agent -t "$RUN-sup" --status | awk '/paused at/{print $4}')
expect "support agent resumes on approve" 'Refunded \$249.00' demo support_agent -t "$RUN-sup" --resume approve
expect "support agent fork re-decides as reject" "wasn't able to issue" \
  demo support_agent -t "$RUN-sup" --resume reject --reason smoke --checkpoint "$PAUSED"
demo support_agent -t "$RUN-sup2" "Order A100 arrived broken" >/dev/null
expect "support agent edit changes the amount" 'Refunded \$20.00' demo support_agent -t "$RUN-sup2" --resume edit --amount 20

expect "durable pipeline crashes (crash expected)" "simulated crash" \
  env PYTHONUNBUFFERED=1 uv run --frozen python -m demos.durable -t "$RUN-job" --crash-at summarize
expect "durable pipeline resumes at the next step" "next step\(s\): \('summarize',\)" demo durable -t "$RUN-job" --status
expect "durable resume finishes without redoing work" "enrich: ok after 3 attempts', 'summarize: done', 'publish: posted'" \
  demo durable -t "$RUN-job" --resume

demo memory -u "$RUN" -t mon "I prefer email over phone calls. I live in Lisbon." >/dev/null
expect "memory recalls across threads" "recalled from other threads: \['User said: I prefer email" \
  demo memory -u "$RUN" -t tue "How should you contact me about my order?"
expect "memory is isolated per user" "recalled from other threads: nothing" \
  demo memory -u "$RUN-other" -t wed "How should you contact me about my order?"
demo memory -u "$RUN" --forget >/dev/null

expect "summarize trims state" "messages in state: 2   summary: The user talked about" demo summarize -t "$RUN-sum"
expect "fanout runs branches in parallel" "3 research branches ran in parallel" demo fanout -t "$RUN-fan"
expect "multi-agent routes to both specialists" "handled by: \['billing', 'tech'\]" demo multi_agent -t "$RUN-multi"

echo "All $pass checks passed."
