#!/usr/bin/env bash
# Checks everything the demo needs is up. Run from the repo root: ./scripts/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ok()   { printf "  \033[32m✔\033[0m %s\n" "$1"; }
bad()  { printf "  \033[31m✘\033[0m %s\n" "$1"; FAILED=1; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
FAILED=0

echo "Config"
[ -f .env ] && ok ".env present" || bad ".env missing -> cp .env.example .env"

echo "Infrastructure"
docker info >/dev/null 2>&1 && ok "Docker is running" || bad "Docker is not running"

if docker compose exec -T postgres pg_isready -U langgraph -d langgraph >/dev/null 2>&1; then
  ok "LangGraph Postgres on localhost:5433"
else
  bad "LangGraph Postgres down -> docker compose up -d --wait"
fi

if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/api/public/health)" = "200" ]; then
  ok "Langfuse web on http://localhost:3000"
else
  bad "Langfuse down -> docker compose -f langfuse/docker-compose.yml up -d (takes ~1 min)"
fi

if curl -s -o /dev/null -w '%{http_code}' -u pk-lf-local-demo:sk-lf-local-demo \
     http://localhost:3000/api/public/projects | grep -q 200; then
  ok "Langfuse API keys accepted"
else
  bad "Langfuse rejected the API keys in .env"
fi

echo "LLM"
BASE_URL=${OPENAI_BASE_URL-$(grep -sE '^OPENAI_BASE_URL=' .env | cut -d= -f2-)}
if [ -z "$BASE_URL" ]; then
  warn "OPENAI_BASE_URL unset -> demos run OFFLINE with the scripted model"
elif curl -s -m 5 -o /dev/null "$BASE_URL/models"; then
  ok "LLM endpoint reachable: $BASE_URL"
else
  bad "LLM endpoint unreachable: $BASE_URL (start it, or prefix commands with OPENAI_BASE_URL=)"
fi

echo "Python"
uv run python -c "import demos.common" >/dev/null 2>&1 && ok "uv environment and demos import" \
  || bad "demos fail to import -> uv sync"

[ "$FAILED" = 0 ] && echo "Ready." || { echo "Fix the ✘ items above."; exit 1; }
