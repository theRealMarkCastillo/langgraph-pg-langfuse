"""Human-in-the-loop support agent: tool calling + durable approval + Langfuse scores.

The agent can look up orders on its own, but any refund pauses the graph with
`interrupt()`. The pending approval lives in Postgres, so the process exits and
a human can approve/reject/edit later, from another process. The resume joins
the same Langfuse trace, and the human's decision is recorded as a score on it.

    uv run python -m demos.support_agent -t t1 "Order A100 arrived broken, I want a refund"
    uv run python -m demos.support_agent -t t1 --status
    uv run python -m demos.support_agent -t t1 --resume approve
    uv run python -m demos.support_agent -t t1 --resume reject --reason "outside policy"
    uv run python -m demos.support_agent -t t1 --resume edit --amount 20
    # Time travel: re-decide an earlier pause (prints its checkpoint id) -> new branch
    uv run python -m demos.support_agent -t t1 --resume reject --checkpoint <id>
"""

import argparse
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from demos.common import (
    flush,
    langfuse,
    last_human_text,
    make_llm,
    postgres_checkpointer,
    reply,
    sessions_url,
    stream_run,
    traced,
)

ORDERS = {
    "A100": {"item": "Espresso machine", "total": 249.00, "status": "delivered", "days_since_delivery": 3},
    "B200": {"item": "Coffee grinder", "total": 89.50, "status": "delivered", "days_since_delivery": 45},
    "C300": {"item": "Milk frother", "total": 35.00, "status": "in transit", "days_since_delivery": None},
}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by id (e.g. A100). Returns item, total, status and days since delivery."""
    return ORDERS.get(order_id.upper(), {"error": f"no order {order_id}"})


@tool
def refund_policy() -> str:
    """Return the store's refund policy."""
    return "Full refunds within 30 days of delivery for damaged or defective items. Otherwise store credit."


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund money to the customer. Requires human approval before it runs."""
    return f"Refunded ${amount:.2f} for order {order_id.upper()} ({reason})."


TOOLS = [lookup_order, refund_policy, issue_refund]
SENSITIVE = {"issue_refund"}


def offline(messages, tools):
    """Scripted agent: look up -> check policy -> propose refund -> summarise."""
    last = messages[-1]
    order = re.search(r"\b([A-Z]\d{3})\b", last_human_text(messages), re.I)
    order_id = order.group(1).upper() if order else None
    if last.type == "human":
        if order_id:
            return reply("", [{"name": "lookup_order", "args": {"order_id": order_id}}])
        return reply("Happy to help! Which order number is this about?")
    if last.type == "tool" and last.name == "lookup_order":
        if "error" in str(last.content):
            return reply(f"I couldn't find order {order_id}. Could you double-check the number?")
        return reply("", [{"name": "refund_policy", "args": {}}])
    if last.type == "tool" and last.name == "refund_policy":
        info = ORDERS[order_id]
        return reply(
            "",
            [
                {
                    "name": "issue_refund",
                    "args": {"order_id": order_id, "amount": info["total"], "reason": "item arrived damaged"},
                }
            ],
        )
    if last.type == "tool" and last.name == "issue_refund":
        if str(last.content).startswith("Refunded"):
            return reply(f"Done! {last.content} You'll see it in 3-5 business days.")
        return reply("I'm sorry, I wasn't able to issue that refund. A teammate will follow up by email.")
    return reply("Is there anything else I can help with?")


llm = make_llm(offline).bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent(state: State) -> dict:
    system = SystemMessage(
        "You are a customer-support agent for a coffee-gear store. Use tools to look up orders "
        "and the refund policy before proposing a refund. Call issue_refund when a refund is warranted."
    )
    return {"messages": [llm.invoke([system, *state["messages"]])]}


def route_after_agent(state: State) -> Literal["tools", "human_approval", "__end__"]:
    calls = getattr(state["messages"][-1], "tool_calls", [])
    if not calls:
        return END
    return "human_approval" if any(c["name"] in SENSITIVE for c in calls) else "tools"


def human_approval(state: State) -> Command[Literal["tools", "agent"]]:
    """Pause for a human. interrupt() persists the payload in Postgres and stops the run;
    on resume, this node re-runs from the top and interrupt() returns the human's decision."""
    ai: AIMessage = state["messages"][-1]
    call = next(c for c in ai.tool_calls if c["name"] in SENSITIVE)
    decision = interrupt(
        {
            "question": f"Approve {call['name']}({call['args']})?",
            "tool_call": call,
            "trace_id": langfuse.get_current_trace_id(),  # lets the resume join this trace
        }
    )

    if decision["action"] == "approve":
        return Command(goto="tools")
    if decision["action"] == "edit":
        edited = [
            {**c, "args": {**c["args"], **decision["args"]}} if c["id"] == call["id"] else c for c in ai.tool_calls
        ]
        # Same message id -> add_messages replaces the proposal instead of appending.
        return Command(goto="tools", update={"messages": [AIMessage(content=ai.content, tool_calls=edited, id=ai.id)]})
    # Rejected: answer every pending tool call so the conversation stays valid.
    reason = decision.get("reason") or "no reason given"
    return Command(
        goto="agent",
        update={
            "messages": [
                ToolMessage(f"Rejected by human reviewer: {reason}", tool_call_id=c["id"], name=c["name"])
                for c in ai.tool_calls
            ]
        },
    )


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("human_approval", human_approval)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def report(graph, config) -> None:
    state = graph.get_state(config)
    if state.interrupts:
        pending = state.interrupts[0].value
        print(f"\n⏸  Waiting for a human: {pending['question']}")
        print(f"   paused at checkpoint {state.config['configurable']['checkpoint_id']}")
        print("   resume with: --resume approve | reject [--reason ...] | edit --amount N")
    elif state.values:
        print(f"\n✔ done: {state.values['messages'][-1].content}")
    else:
        print("\n(no conversation on this thread yet)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="customer message")
    parser.add_argument("-t", "--thread", default="support-demo")
    parser.add_argument("--status", action="store_true", help="show whether the thread is waiting for approval")
    parser.add_argument("--resume", choices=["approve", "reject", "edit"], help="answer the pending approval")
    parser.add_argument("--reason", help="rejection reason")
    parser.add_argument("--amount", type=float, help="edited refund amount")
    parser.add_argument("--checkpoint", help="resume from this (older) paused checkpoint -> forks the thread")
    args = parser.parse_args()

    config = {"configurable": {"thread_id": args.thread}}
    if args.checkpoint:
        config["configurable"]["checkpoint_id"] = args.checkpoint

    with postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        if args.status or not (args.message or args.resume):
            report(graph, config)
            return

        if args.resume:
            pending = graph.get_state(config).interrupts
            if not pending:
                raise SystemExit("Nothing is waiting for approval on this thread/checkpoint.")
            payload = pending[0].value
            decision = {"action": args.resume, "reason": args.reason}
            if args.resume == "edit":
                if args.amount is None:
                    raise SystemExit("--resume edit needs --amount")
                decision["args"] = {"amount": args.amount}
            if args.checkpoint:
                # That checkpoint already holds the earlier decision's result, and resuming it would
                # replay that. Branch first: a no-op update "as agent" makes a new child checkpoint
                # that routes to human_approval again, so the new decision really runs.
                config = graph.update_state(config, None, as_node="agent")
                print(f"[{args.thread}] forked to checkpoint {config['configurable']['checkpoint_id']}")
            print(f"[{args.thread}] human decision: {decision}")
            with traced(
                "support-agent",
                session_id=args.thread,
                trace_id=payload["trace_id"],
                input=decision,
                tags=["support", "resume"],
            ) as (span, handler):
                stream_run(graph, Command(resume=decision), {**config, "callbacks": [handler]})
            # The decision becomes a score on the original trace: filter/analyse approvals in Langfuse.
            langfuse.create_score(
                name="human_approval",
                value=args.resume,
                data_type="CATEGORICAL",
                trace_id=payload["trace_id"],
                comment=args.reason,
            )
        else:
            print(f"[{args.thread}] customer: {args.message}")
            with traced("support-agent", session_id=args.thread, input=args.message, tags=["support"]) as (
                span,
                handler,
            ):
                stream_run(graph, {"messages": [HumanMessage(args.message)]}, {**config, "callbacks": [handler]})

        config["configurable"].pop("checkpoint_id", None)  # report on the thread's latest state
        report(graph, config)

    flush()
    print(f"Traces: {sessions_url()}")


if __name__ == "__main__":
    main()
