"""Durable execution: retries, hard crashes, and resuming from the last checkpoint.

A 4-step pipeline: extract -> enrich -> summarize -> publish.
- `enrich` fails twice with a transient error; a RetryPolicy retries it in-process.
- `--crash-at summarize` kills the process (os._exit) when that step starts, like a
  container being OOM-killed. Steps that finished are already checkpointed in Postgres.
- `--resume` continues the same thread: finished steps are NOT re-run.

    uv run python -m demos.durable -t job1 --crash-at summarize   # dies mid-pipeline
    uv run python -m demos.durable -t job1 --status               # shows what's next
    uv run python -m demos.durable -t job1 --resume               # finishes the job
"""

import argparse
import operator
import os
import time
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from demos.common import flush, make_llm, postgres_checkpointer, reply, sessions_url, stream_run, traced


class State(TypedDict, total=False):
    region: str
    rows: list[dict]
    summary: str
    published_to: str
    log: Annotated[list[str], operator.add]  # appended to by every step


@dataclass
class Context:
    """Run-scoped settings passed via `context=` (not stored in the checkpoint)."""

    crash_at: str | None = None


llm = make_llm(
    lambda messages, tools: reply("Sales rose 12% week over week, led by espresso machines; grinders were flat.")
)

enrich_attempts = 0


def extract(state: State) -> dict:
    time.sleep(0.3)
    rows = [{"sku": "espresso", "units": 112}, {"sku": "grinder", "units": 40}, {"sku": "frother", "units": 65}]
    return {"rows": rows, "log": [f"extract: {len(rows)} rows for {state['region']}"]}


def enrich(state: State) -> dict:
    global enrich_attempts
    enrich_attempts += 1
    if enrich_attempts <= 2:
        print(f"    enrich attempt {enrich_attempts}: pricing API timed out (transient)")
        raise ConnectionError("pricing API timed out")
    prices = {"espresso": 249.0, "grinder": 89.5, "frother": 35.0}
    rows = [{**r, "revenue": r["units"] * prices[r["sku"]]} for r in state["rows"]]
    return {"rows": rows, "log": [f"enrich: ok after {enrich_attempts} attempts"]}


def maybe_crash(step: str, runtime: Runtime[Context]) -> None:
    if runtime.context and runtime.context.crash_at == step:
        print(f"    💥 simulated crash at '{step}' (process killed, nothing cleaned up)", flush=True)
        os._exit(1)  # skips exception handling and Langfuse flush: as abrupt as a real crash


def summarize(state: State, runtime: Runtime[Context]) -> dict:
    maybe_crash("summarize", runtime)
    msg = llm.invoke([SystemMessage("Summarise this sales data in one sentence."), HumanMessage(str(state["rows"]))])
    return {"summary": msg.content, "log": ["summarize: done"]}


def publish(state: State, runtime: Runtime[Context]) -> dict:
    maybe_crash("publish", runtime)
    time.sleep(0.3)
    return {"published_to": f"#sales-{state['region']}", "log": ["publish: posted"]}


def build_graph(checkpointer):
    g = StateGraph(State, context_schema=Context)
    g.add_node("extract", extract)
    g.add_node(
        "enrich", enrich, retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.2, retry_on=ConnectionError)
    )
    g.add_node("summarize", summarize)
    g.add_node("publish", publish)
    g.add_edge(START, "extract")
    g.add_edge("extract", "enrich")
    g.add_edge("enrich", "summarize")
    g.add_edge("summarize", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--thread", default="durable-demo")
    parser.add_argument("--region", default="emea")
    parser.add_argument("--crash-at", choices=["summarize", "publish"])
    parser.add_argument("--resume", action="store_true", help="continue the thread from its last checkpoint")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config = {"configurable": {"thread_id": args.thread}}

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        state = graph.get_state(config)
        if args.status or args.resume:
            print(f"[{args.thread}] completed steps: {state.values.get('log', [])}")
            print(f"[{args.thread}] next step(s): {state.next or 'none - job finished'}")
            if args.status or not state.next:
                return
        elif state.values:
            raise SystemExit(f"Thread {args.thread!r} already ran; use --resume/--status or a new -t")

        # Input None = "continue from the checkpoint"; a dict = start a new run.
        inp = None if args.resume else {"region": args.region}
        with traced("durable-pipeline", session_id=args.thread, input=inp or "resume", tags=["durable"]) as (
            span,
            handler,
        ):
            # durability="sync": each step's checkpoint is written before the next step starts,
            # so a crash can never lose a step that already reported success.
            stream_run(
                graph,
                inp,
                {**config, "callbacks": [handler]},
                context=Context(crash_at=args.crash_at),
                durability="sync",
            )
            final = graph.get_state(config).values
            span.update(output=final.get("summary"))
        print(f"\n✔ {final.get('summary')}\n  published to {final.get('published_to')}")
        print(f"  step log: {final.get('log')}")

    flush()
    print(f"Traces: {sessions_url()}")


if __name__ == "__main__":
    main()
