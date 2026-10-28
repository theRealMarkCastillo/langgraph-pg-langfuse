"""Langfuse prompt management: the system prompt lives in Langfuse, not in code.

The node fetches the prompt labelled `production`, so publishing a new version in
Langfuse (UI or `--publish`) changes behaviour without a deploy. Each generation
is linked to the exact prompt version it used, so Langfuse can compare versions.

    uv run python -m demos.prompts "Do you ship to Canada?"
    uv run python -m demos.prompts --publish "You are a pirate-themed assistant for {{company}}. Tone: {{tone}}."
    uv run python -m demos.prompts --version 1 "Do you ship to Canada?"   # pin an older version
"""

import argparse
import re
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from demos.common import flush, langfuse, make_llm, postgres_checkpointer, reply, sessions_url, stream_run, traced

PROMPT_NAME = "support-assistant"
V1_SYSTEM = "You are a helpful support assistant for {{company}}. Tone: {{tone}}. Answer in one or two sentences."


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def offline(messages, tools):
    system = str(messages[0].content)
    question = str(messages[-1].content)
    if "pirate" in system.lower():
        return reply(f"Arr, matey! Aye, we ship that far. ({question})")
    return reply(f"Yes, happy to help with that! ({question})")


llm = make_llm(offline)


def ensure_prompt() -> None:
    """Create v1 on first run so the demo is self-contained."""
    try:
        langfuse.get_prompt(PROMPT_NAME, type="chat", cache_ttl_seconds=0)
    except Exception:
        publish(V1_SYSTEM, "initial version")


def publish(system: str, message: str = "published from demos.prompts") -> None:
    p = langfuse.create_prompt(
        name=PROMPT_NAME,
        type="chat",
        labels=["production"],
        commit_message=message,
        prompt=[
            {"role": "system", "content": system},
            {"type": "placeholder", "name": "history"},
        ],  # filled with the conversation
        config={"company": "Bean There Coffee Co.", "tone": "warm and concise"},
    )
    print(f"published {PROMPT_NAME} v{p.version} with label 'production'")


def respond(state: State, config) -> dict:
    version = config["configurable"].get("prompt_version")
    # cache_ttl_seconds=0 so a freshly published version is picked up immediately in the demo.
    prompt = langfuse.get_prompt(
        PROMPT_NAME, type="chat", version=version, label=None if version else "production", cache_ttl_seconds=0
    )
    template = ChatPromptTemplate.from_messages(prompt.get_langchain_prompt())
    # This metadata is what links the generation to the prompt version in Langfuse.
    template = template.with_config(metadata={"langfuse_prompt": prompt})
    chain = template | llm
    print(f"    using {PROMPT_NAME} v{prompt.version}")
    return {"messages": [chain.invoke({**prompt.config, "history": state["messages"]})]}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?")
    parser.add_argument("-t", "--thread", default=None)
    parser.add_argument("--publish", metavar="SYSTEM_PROMPT", help="publish a new production version")
    parser.add_argument("--version", type=int, help="use this prompt version instead of 'production'")
    args = parser.parse_args()

    ensure_prompt()
    if args.publish:
        if not re.search(r"\{\{\w+\}\}", args.publish):
            print("note: no {{company}}/{{tone}} variables in the new prompt; that's fine")
        publish(args.publish)
        if not args.message:
            flush()
            return

    thread = args.thread or f"prompts-{uuid.uuid4().hex[:6]}"
    text = args.message or "Do you ship to Canada?"
    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        print(f"[{thread}] user: {text}")
        with traced("prompt-managed-turn", session_id=thread, input=text, tags=["prompts"]) as (span, handler):
            stream_run(
                graph,
                {"messages": [HumanMessage(text)]},
                {"configurable": {"thread_id": thread, "prompt_version": args.version}, "callbacks": [handler]},
            )

    flush()
    print(f"Prompt: {sessions_url().rsplit('/', 1)[0]}/prompts/{PROMPT_NAME}")


if __name__ == "__main__":
    main()
