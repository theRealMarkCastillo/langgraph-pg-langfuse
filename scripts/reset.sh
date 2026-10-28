#!/usr/bin/env bash
# Wipes demo data so a run starts clean. Run from the repo root.
#   ./scripts/reset.sh            LangGraph state + memories only (seconds)
#   ./scripts/reset.sh --all      also Langfuse traces, scores, prompts, datasets (~1 min)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Resetting LangGraph Postgres (checkpoints + long-term memory)..."
docker compose down -v
docker compose up -d --wait

if [ "${1:-}" = "--all" ]; then
  echo "Resetting Langfuse (re-creates org, project, keys and login user on boot)..."
  docker compose -f langfuse/docker-compose.yml down -v
  docker compose -f langfuse/docker-compose.yml up -d
  printf "Waiting for Langfuse"
  for _ in $(seq 1 90); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -u pk-lf-local-demo:sk-lf-local-demo \
          http://localhost:3000/api/public/projects)" = "200" ]; then
      echo " up."; break
    fi
    printf "."; sleep 2
  done
fi

echo "Done. Run ./scripts/preflight.sh next."
