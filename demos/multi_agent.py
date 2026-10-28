"""Multi-agent: a supervisor routes to specialist sub-agents with Command(goto=...).

supervisor -> billing | tech -> supervisor -> ... -> FINISH
Each specialist is its own compiled subgraph (draft -> review). Subgraphs share the
parent's checkpointer; their checkpoints live under their own namespace, and with
subgraphs=True streaming shows which agent each step came from.

    uv run python -m demos.multi_agent "I was charged twice and the app crashes on login"
"""

import argparse
import re
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

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

SPECIALISTS = {
    "billing": "charges, refunds, invoices, payments and subscriptions",
    "tech": "app crashes, login problems, bugs and errors",
}
KEYWORDS = {
    "billing": r"charg|refund|invoice|bill|payment|subscription|price",
    "tech": r"crash|login|log in|bug|error|freez|password|app",
}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Specialists that already answered. Deliberately no append-reducer: a subgraph returns its
    # whole state (including the `handled` it inherited), which an append would double-count.
    handled: list[str]


def offline(messages, tools):
    prompt = str(messages[0].content)
    question = last_human_text(messages)
    if prompt.startswith("You route"):
        done = re.search(r"Already handled: (.*)", prompt).group(1)
        todo = [n for n, kw in KEYWORDS.items() if re.search(kw, question, re.I) and n not in done]
        return reply(todo[0] if todo else "FINISH")
    if prompt.startswith("Review"):
        return reply(str(messages[-1].content) + " Anything else I can help with?")
    team = "billing" if "billing" in prompt else "tech"
    fix = (
        "I've flagged the duplicate charge; the extra payment will be refunded within 5 days."
        if team == "billing"
        else "Please update to the latest version and clear the app cache; that fixes the login crash."
    )
    return reply(f"[{team}] {fix}")


llm = make_llm(offline)
router_llm = llm.with_config(tags=["nostream"])


def supervisor(state: State) -> Command[Literal["billing", "tech", "__end__"]]:
    menu = "\n".join(f"- {name}: {desc}" for name, desc in SPECIALISTS.items())
    choice = (
        router_llm.invoke(
            [
                SystemMessage(
                    f"You route customer messages to specialists:\n{menu}\n"
                    f"Already handled: {', '.join(state.get('handled', [])) or 'none'}\n"
                    "Reply with exactly one word: the next specialist still needed, or FINISH."
                ),
                HumanMessage(last_human_text(state["messages"])),
            ]
        )
        .content.strip()
        .lower()
    )
    nxt = next((name for name in SPECIALISTS if name in choice and name not in state.get("handled", [])), None)
    return Command(goto=nxt or END)


def make_specialist(name: str, focus: str):
    """A small sub-agent: draft an answer, then self-review it."""

    def draft(state: State) -> dict:
        return {
            "messages": [
                llm.invoke(
                    [
                        SystemMessage(
                            f"You are the {name} specialist ({focus}). Answer only the {name} part "
                            "of the customer's message in one or two sentences."
                        ),
                        HumanMessage(last_human_text(state["messages"])),
                    ]
                )
            ]
        }

    def review(state: State) -> dict:
        polished = llm.invoke(
            [
                SystemMessage("Review this support reply for tone and clarity; return the improved reply only."),
                state["messages"][-1],
            ]
        )
        # Same id -> replaces the draft rather than adding a second message.
        return {
            "messages": [AIMessage(content=polished.content, id=state["messages"][-1].id)],
            "handled": [*state.get("handled", []), name],
        }

    sub = StateGraph(State)
    sub.add_node("draft", draft)
    sub.add_node("review", review)
    sub.add_edge(START, "draft")
    sub.add_edge("draft", "review")
    sub.add_edge("review", END)
    return sub.compile()  # inherits the parent's checkpointer


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    for name, focus in SPECIALISTS.items():
        g.add_node(name, make_specialist(name, focus))
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    return g.compile(checkpointer=checkpointer)


def run(graph, text: str, thread: str, handler=None, show: bool = True) -> dict:
    config = {"configurable": {"thread_id": thread}, "callbacks": [handler] if handler else []}
    inp = {"messages": [HumanMessage(text)], "handled": []}
    if show:
        stream_run(graph, inp, config, subgraphs=True)
    else:
        graph.invoke(inp, config)
    return graph.get_state(config).values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", default="I was charged twice and the app crashes when I log in")
    parser.add_argument("-t", "--thread", default=None, help="defaults to a fresh id per run")
    args = parser.parse_args()
    thread = args.thread or f"multi-{uuid.uuid4().hex[:6]}"

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        print(f"[{thread}] customer: {args.message}")
        with traced("multi-agent", session_id=thread, input=args.message, tags=["multi-agent"]) as (span, handler):
            values = run(graph, args.message, thread, handler)
            span.update(output=[m.content for m in values["messages"] if m.type == "ai"])
        print(f"\n✔ handled by: {values['handled']}")
        for m in values["messages"]:
            if m.type == "ai":
                print(f"  {m.content}")

    flush()
    print(f"Traces: {sessions_url()}")


if __name__ == "__main__":
    main()
