"""Basics: Postgres-backed conversation state, routing, streaming, time travel.

Each run continues the conversation identified by --thread. LangGraph checkpoints
the state to Postgres after every node, so later runs see the full history.

    uv run python -m demos.chat                          # scripted 3-turn conversation
    uv run python -m demos.chat -t t1 "Hi there"         # one turn, tokens streamed
    uv run python -m demos.chat -t t1 --history          # checkpoints stored in Postgres
    uv run python -m demos.chat -t t1 --fork <checkpoint_id> "Actually, call me Bob"
"""

import argparse
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    turns: int


def offline(messages, tools):
    text = last_human_text(messages)
    names = [
        m.group(1)
        for msg in messages
        if msg.type == "human"
        if (m := re.search(r"(?:name is|call me) (\w+)", str(msg.content), re.I))
    ]
    if "name" in text.lower() and "?" in text:
        name = names[-1] if names else "something you haven't told me yet"
        return reply(f"You told me your name is {name}. I read that from the history stored in Postgres.")
    if "?" in text:
        earlier = sum(1 for m in messages if m.type in ("human", "ai")) - 1
        return reply(
            f"Good question. I can see all {earlier} earlier messages in this thread, "
            "because every step is checkpointed to Postgres."
        )
    return reply(f"Nice to meet you{', ' + names[-1] if names else ''}! I'll remember that.")


llm = make_llm(offline)


def classify(state: State) -> dict:
    """Cheap rule-based router so the trace shows a branch decision."""
    text = state["messages"][-1].content.lower()
    intent = "question" if text.rstrip().endswith("?") else "chat"
    return {"intent": intent, "turns": state.get("turns", 0) + 1}


def route(state: State) -> Literal["answer", "chat"]:
    return "answer" if state["intent"] == "question" else "chat"


def answer(state: State) -> dict:
    system = SystemMessage("Answer the user's question concisely using the conversation so far.")
    return {"messages": [llm.invoke([system, *state["messages"]])]}


def chat(state: State) -> dict:
    system = SystemMessage("You are a friendly assistant. Reply in one or two sentences.")
    return {"messages": [llm.invoke([system, *state["messages"]])]}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("answer", answer)
    g.add_node("chat", chat)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route)
    g.add_edge("answer", END)
    g.add_edge("chat", END)
    return g.compile(checkpointer=checkpointer)


def run_turn(graph, thread_id: str, text: str, checkpoint_id: str | None = None) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        # Time travel: start from an older checkpoint. The new steps form a branch;
        # the original history after that checkpoint is kept, not overwritten.
        config["configurable"]["checkpoint_id"] = checkpoint_id
        # An unknown checkpoint_id silently behaves like an empty thread, so check first.
        if not graph.get_state(config).values:
            raise SystemExit(f"No checkpoint {checkpoint_id!r} in thread {thread_id!r} (see --history)")
    print(f"\n[{thread_id}] user: {text}")
    with traced("chat-turn", session_id=thread_id, input=text, tags=["chat"]) as (span, handler):
        stream_run(graph, {"messages": [HumanMessage(text)]}, {**config, "callbacks": [handler]})
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        span.update(output=state.values["messages"][-1].content)
    print(f"  turn {state.values['turns']}, {len(state.values['messages'])} messages in Postgres")


def print_history(graph, thread_id: str) -> None:
    for snap in graph.get_state_history({"configurable": {"thread_id": thread_id}}):
        msgs = snap.values.get("messages", [])
        last = f"{msgs[-1].type}: {str(msgs[-1].content)[:50]}" if msgs else ""
        print(
            f"step={snap.metadata.get('step'):>3}  next={snap.next or ('END',)!s:<16} "
            f"checkpoint_id={snap.config['configurable']['checkpoint_id']}  {last}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="user message for a single turn")
    parser.add_argument("-t", "--thread", default="chat-demo", help="conversation/thread id")
    parser.add_argument("--history", action="store_true", help="list stored checkpoints and exit")
    parser.add_argument("--fork", metavar="CHECKPOINT_ID", help="branch the thread from this checkpoint")
    args = parser.parse_args()

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        if args.history:
            print_history(graph, args.thread)
            return
        messages = (
            [args.message]
            if args.message
            else [
                "Hi! My name is Ada and I'm testing LangGraph.",
                "Where is our conversation state stored?",
                "What's my name?",
            ]
        )
        for text in messages:
            run_turn(graph, args.thread, text, args.fork)

    flush()
    print(f"\nTraces: {sessions_url()}")


if __name__ == "__main__":
    main()
