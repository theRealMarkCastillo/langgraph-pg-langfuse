"""Managing a long conversation: summarise old messages and delete them from state.

After each reply, if the thread holds more than MAX_MESSAGES, older messages are
folded into a running `summary` and removed with RemoveMessage. The checkpoint in
Postgres stays small, while the model still gets the gist via the summary.

    uv run python -m demos.summarize -t long1          # scripted 6-turn conversation
    uv run python -m demos.summarize -t long1 "And what was the first thing I asked?"
"""

import argparse
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
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

MAX_MESSAGES = 6  # summarise once the thread holds more than this
KEEP_LAST = 2  # newest messages kept verbatim


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str


def offline_respond(messages, tools):
    summary = re.search(r"Summary of earlier conversation: (.+)", str(messages[0].content))
    text = last_human_text(messages)
    if "first" in text.lower() and summary:
        return reply(f"From my summary of our earlier chat: {summary.group(1)}")
    return reply(f"Got it: {text.rstrip('?.!')}.")


def offline_summarize(messages, tools):
    prior = re.search(r"Existing summary: (.*)", str(messages[0].content))
    topics = [str(m.content).rstrip("?.!") for m in messages[1:] if m.type == "human"]
    parts = ([prior.group(1)] if prior and prior.group(1) != "(none)" else []) + topics
    return reply("The user talked about: " + "; ".join(p.removeprefix("The user talked about: ") for p in parts))


respond_llm = make_llm(offline_respond)
summarize_llm = make_llm(offline_summarize).with_config(tags=["nostream"])


def respond(state: State) -> dict:
    system = "You are a concise assistant. Reply in one sentence."
    if state.get("summary"):
        system += f"\nSummary of earlier conversation: {state['summary']}"
    return {"messages": [respond_llm.invoke([SystemMessage(system), *state["messages"]])]}


def should_summarize(state: State) -> Literal["summarize", "__end__"]:
    return "summarize" if len(state["messages"]) > MAX_MESSAGES else END


def summarize(state: State) -> dict:
    old = state["messages"][:-KEEP_LAST]
    summary = summarize_llm.invoke(
        [
            SystemMessage(
                "Update the running summary with the new messages. Be brief.\n"
                f"Existing summary: {state.get('summary') or '(none)'}"
            ),
            *old,
        ]
    ).content
    # RemoveMessage(id) is handled by the add_messages reducer: those messages leave the state.
    return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old]}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_node("summarize", summarize)
    g.add_edge(START, "respond")
    g.add_conditional_edges("respond", should_summarize)
    g.add_edge("summarize", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?")
    parser.add_argument("-t", "--thread", default="summarize-demo")
    args = parser.parse_args()
    config = {"configurable": {"thread_id": args.thread}}

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        messages = (
            [args.message]
            if args.message
            else [
                "I'm planning a trip to Japan.",
                "I want to see Kyoto temples.",
                "Budget is about $3000.",
                "I'll travel in April.",
                "I like ramen.",
                "What was the first thing I asked about?",
            ]
        )
        for text in messages:
            print(f"\n[{args.thread}] user: {text}")
            with traced("summarize-turn", session_id=args.thread, input=text, tags=["summarize"]) as (_, handler):
                stream_run(graph, {"messages": [HumanMessage(text)]}, {**config, "callbacks": [handler]})
            values = graph.get_state(config).values
            print(f"  messages in state: {len(values['messages'])}   summary: {values.get('summary') or '-'}")

    flush()
    print(f"\nTraces: {sessions_url()}")


if __name__ == "__main__":
    main()
