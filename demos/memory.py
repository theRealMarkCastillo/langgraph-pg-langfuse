"""Long-term memory across threads: PostgresStore + pgvector semantic search.

Checkpoints are per-thread (short-term). The store is per-user and shared by all
of that user's threads (long-term). Each turn:
  recall   -> vector-search the user's memories for ones relevant to the message
  respond  -> answer with those memories in the prompt
  remember -> extract durable facts from the message and save them

    uv run python -m demos.memory -u ada -t monday  "I prefer email over phone calls"
    uv run python -m demos.memory -u ada -t tuesday "How should you contact me about my order?"
    uv run python -m demos.memory -u ada --list
"""

import argparse
import re
import uuid
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from demos.common import (
    flush,
    last_human_text,
    make_llm,
    postgres_checkpointer,
    postgres_store,
    reply,
    sessions_url,
    stream_run,
    traced,
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    recalled: list[str]


@dataclass
class Context:
    user_id: str


def offline_respond(messages, tools):
    memories = re.findall(r"^- (.+)$", str(messages[0].content), re.M)
    if not memories:
        return reply("Thanks! I don't know anything about you yet, but I'll remember what you tell me.")
    return reply(f"Based on what I remember about you ({'; '.join(memories)}), I'll keep that in mind.")


def offline_extract(messages, tools):
    text = last_human_text(messages)
    facts = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\b(I prefer|I like|I love|I hate|I'm|I am|I live|I work|my \w+ is)\b", s, re.I)
        and not s.strip().endswith("?")
    ]
    return reply("\n".join(f"User said: {f.rstrip('.!')}" for f in facts) or "NONE")


respond_llm = make_llm(offline_respond)
# Internal call: keep its tokens out of the user-facing stream (still traced in Langfuse).
extract_llm = make_llm(offline_extract).with_config(tags=["nostream"])


def namespace(runtime: Runtime[Context]) -> tuple[str, str]:
    return ("memories", runtime.context.user_id)


def recall(state: State, runtime: Runtime[Context]) -> dict:
    hits = runtime.store.search(namespace(runtime), query=last_human_text(state["messages"]), limit=3)
    return {"recalled": [h.value["text"] for h in hits if (h.score or 0) > 0.2]}


def respond(state: State, runtime: Runtime[Context]) -> dict:
    known = "\n".join(f"- {m}" for m in state["recalled"]) or "(nothing yet)"
    system = SystemMessage(
        f"You are a helpful assistant. Things you remember about this user:\n{known}\n"
        "Use them when relevant. Reply in one or two sentences."
    )
    return {"messages": [respond_llm.invoke([system, *state["messages"]])]}


def remember(state: State, runtime: Runtime[Context]) -> dict:
    out = extract_llm.invoke(
        [
            SystemMessage(
                "Extract durable facts or preferences about the user from their message, one per line, "
                "phrased in third person. If there are none, reply exactly NONE."
            ),
            HumanMessage(last_human_text(state["messages"])),
        ]
    ).content.strip()
    if out != "NONE":
        for fact in filter(None, (line.strip("-• ").strip() for line in out.splitlines())):
            # put() embeds the "text" field and writes it to pgvector.
            runtime.store.put(namespace(runtime), str(uuid.uuid4()), {"text": fact})
            print(f"    saved memory: {fact}")
    return {}


def build_graph(checkpointer, store):
    g = StateGraph(State, context_schema=Context)
    g.add_node("recall", recall)
    g.add_node("respond", respond)
    g.add_node("remember", remember)
    g.add_edge(START, "recall")
    g.add_edge("recall", "respond")
    g.add_edge("respond", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer, store=store)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?")
    parser.add_argument("-u", "--user", default="ada", help="user id: memories are scoped per user")
    parser.add_argument("-t", "--thread", default="memory-demo", help="conversation id: history is per thread")
    parser.add_argument("--list", action="store_true", help="list this user's stored memories")
    parser.add_argument("--forget", action="store_true", help="delete this user's memories")
    args = parser.parse_args()

    with postgres_checkpointer() as checkpointer, postgres_store() as store:
        ns = ("memories", args.user)
        if args.list or args.forget:
            items = store.search(ns, limit=100)
            for item in items:
                print(f"- {item.value['text']}  ({item.updated_at:%Y-%m-%d %H:%M})")
                if args.forget:
                    store.delete(ns, item.key)
            print(f"{len(items)} memories {'deleted' if args.forget else 'stored'} for {args.user}")
            return

        graph = build_graph(checkpointer, store)
        messages = [args.message] if args.message else ["I prefer email over phone calls. I live in Lisbon."]
        for text in messages:
            print(f"[user={args.user} thread={args.thread}] {text}")
            with traced("memory-turn", session_id=args.thread, user_id=args.user, input=text, tags=["memory"]) as (
                span,
                handler,
            ):
                stream_run(
                    graph,
                    {"messages": [HumanMessage(text)]},
                    {"configurable": {"thread_id": args.thread}, "callbacks": [handler]},
                    context=Context(user_id=args.user),
                )
            recalled = graph.get_state({"configurable": {"thread_id": args.thread}}).values["recalled"]
            print(f"  recalled from other threads: {recalled or 'nothing'}")

    flush()
    print(f"Traces: {sessions_url()}")


if __name__ == "__main__":
    main()
