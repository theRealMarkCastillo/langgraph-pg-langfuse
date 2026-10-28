"""Parallel fan-out / map-reduce with the Send API.

plan -> research x N (in parallel, one Send per subtopic) -> write
Each parallel `research` branch returns notes; a reducer (operator.add) merges
them. In Langfuse the research spans overlap in time on the trace timeline.

    uv run python -m demos.fanout "home espresso"
"""

import argparse
import operator
import time
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from demos.common import (
    flush,
    last_human_text,
    make_llm,
    postgres_checkpointer,
    reply,
    sessions_url,
    stream_run,
    traced,
)


class State(TypedDict, total=False):
    topic: str
    subtopics: list[str]
    notes: Annotated[list[str], operator.add]  # parallel branches append here
    research_seconds: Annotated[list[float], operator.add]  # how long each branch took
    brief: str


class ResearchInput(TypedDict):
    """What each parallel branch receives: its own slice, not the whole state."""

    topic: str
    subtopic: str


def offline(messages, tools):
    prompt = str(messages[0].content)
    if prompt.startswith("List"):
        return reply("equipment\ncosts\ntechnique")
    if prompt.startswith("Write"):
        return reply("Brief: " + " ".join(n.split(": ", 1)[1] for n in last_human_text(messages).splitlines()))
    sub = last_human_text(messages)
    return reply(f"{sub}: key facts about {sub} gathered by a parallel worker.")


llm = make_llm(offline)
quiet_llm = llm.with_config(tags=["nostream"])  # parallel branches would interleave tokens


def plan(state: State) -> dict:
    out = quiet_llm.invoke(
        [
            SystemMessage("List exactly 3 short subtopics to research, one per line, no numbering."),
            HumanMessage(state["topic"]),
        ]
    ).content
    return {"subtopics": [s.strip("-• 0123456789.").strip() for s in out.splitlines() if s.strip()][:3]}


def fan_out(state: State) -> list[Send]:
    # One Send per subtopic: LangGraph runs these `research` tasks concurrently.
    return [Send("research", {"topic": state["topic"], "subtopic": s}) for s in state["subtopics"]]


def research(inp: ResearchInput) -> dict:
    started = time.time()
    time.sleep(1.0)  # simulate a slow lookup; the three run at the same time
    note = quiet_llm.invoke(
        [
            SystemMessage(f"Give one sentence of useful facts about this aspect of {inp['topic']}."),
            HumanMessage(inp["subtopic"]),
        ]
    ).content
    if not note.startswith(inp["subtopic"]):
        note = f"{inp['subtopic']}: {note}"
    return {"notes": [note], "research_seconds": [time.time() - started]}


def write(state: State) -> dict:
    brief = llm.invoke(
        [SystemMessage("Write a 3-sentence brief from these notes."), HumanMessage("\n".join(state["notes"]))]
    ).content
    return {"brief": brief}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("write", write)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["research"])
    g.add_edge("research", "write")  # waits for every parallel branch
    g.add_edge("write", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topic", nargs="?", default="home espresso")
    parser.add_argument("-t", "--thread", default=None, help="defaults to a fresh id per run")
    args = parser.parse_args()
    thread = args.thread or f"fanout-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread}}

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        print(f"[{thread}] topic: {args.topic}")
        with traced("fanout-research", session_id=thread, input=args.topic, tags=["fanout"]) as (span, handler):
            stream_run(graph, {"topic": args.topic}, {**config, "callbacks": [handler]})
            final = graph.get_state(config).values
            span.update(output=final["brief"])
        took = final["research_seconds"]
        print(f"\n✔ {final['brief']}")
        # In parallel, the research phase takes about as long as its slowest branch, not the sum.
        print(
            f"  {len(took)} research branches ran in parallel: slowest {max(took):.1f}s, "
            f"combined {sum(took):.1f}s of work"
        )

    flush()
    print(f"Traces: {sessions_url()}")


if __name__ == "__main__":
    main()
