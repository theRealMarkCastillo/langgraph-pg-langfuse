"""Shared wiring for every demo: Postgres checkpointer/store, LLM, embeddings, Langfuse."""

import json
import logging
import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from langfuse import get_client, propagate_attributes

load_dotenv()

if not os.getenv("POSTGRES_URI"):
    raise SystemExit("POSTGRES_URI is not set. Create your config first:  cp .env.example .env")
POSTGRES_URI = os.environ["POSTGRES_URI"]
ONLINE = bool(os.getenv("OPENAI_BASE_URL"))


# --- LLM -------------------------------------------------------------------

Script = Callable[[list[BaseMessage], list[str]], AIMessage]
"""Offline behaviour: (messages, bound tool names) -> the AIMessage to return."""


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for a real model so every demo runs offline.

    Each demo passes a small `script` function that decides the reply (including
    tool calls). Generations still flow through LangChain callbacks, so Langfuse
    traces look like they would with a real model.
    """

    script: Script
    tool_names: list[str] = []
    model_name: str = "offline-scripted"

    @property
    def _llm_type(self) -> str:
        return "offline-scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        names = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script(list(messages), self.tool_names)
        msg.usage_metadata = {
            "input_tokens": sum(len(str(m.content)) // 4 for m in messages),
            "output_tokens": max(1, len(str(msg.content)) // 4),
            "total_tokens": 0,
        }
        msg.usage_metadata["total_tokens"] = msg.usage_metadata["input_tokens"] + msg.usage_metadata["output_tokens"]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._generate(messages).generations[0].message
        words = re.findall(r"\S+\s*", str(msg.content))
        for word in words:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk
        # Tool calls + usage ride on the final chunk.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                    for i, tc in enumerate(msg.tool_calls)
                ],
                usage_metadata=msg.usage_metadata,
                chunk_position="last",
            )
        )


def make_llm(offline: Script) -> BaseChatModel:
    """A ChatOpenAI client for OPENAI_BASE_URL, or the scripted offline model."""
    if ONLINE:
        from langchain_openai import ChatOpenAI

        # Any OpenAI-compatible /v1/chat/completions endpoint (vLLM, Ollama, LiteLLM, OpenAI, ...).
        return ChatOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.getenv("OPENAI_API_KEY") or "not-needed",  # many local servers ignore it
            model=os.environ["OPENAI_MODEL"],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "300")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )
    return ScriptedChatModel(script=offline)


def reply(text: str, tool_calls: list[dict] | None = None) -> AIMessage:
    """Build an AIMessage for offline scripts. tool_calls: [{"name", "args"}]."""
    calls = [
        {
            "name": c["name"],
            "args": c["args"],
            "id": f"call_{c['name']}_{abs(hash(str(c['args']))) % 10**8}",
            "type": "tool_call",
        }
        for c in tool_calls or []
    ]
    return AIMessage(content=text, tool_calls=calls)


def last_human_text(messages: Sequence[AnyMessage]) -> str:
    return next((str(m.content) for m in reversed(messages) if m.type == "human"), "")


# --- Embeddings ------------------------------------------------------------


def make_embeddings() -> tuple[Embeddings, int] | None:
    """(embeddings, dims) when OPENAI_EMBEDDING_MODEL is set, else None (no vector search).

    Embeddings may use a different endpoint than chat (e.g. DeepSeek chat + OpenAI or
    Ollama embeddings): OPENAI_EMBEDDING_BASE_URL / OPENAI_EMBEDDING_API_KEY fall back
    to OPENAI_BASE_URL / OPENAI_API_KEY.
    """
    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if not model:
        return None
    base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise SystemExit("OPENAI_EMBEDDING_MODEL is set but neither OPENAI_EMBEDDING_BASE_URL nor OPENAI_BASE_URL is.")
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        base_url=base_url,
        api_key=os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed",
        model=model,
        check_embedding_ctx_length=False,  # tiktoken lookup breaks on non-OpenAI model names
    )
    dims = os.getenv("OPENAI_EMBEDDING_DIMS")
    # Unset -> ask the model once; pgvector needs the exact size up front.
    return embeddings, int(dims) if dims else len(embeddings.embed_query("dimension probe"))


# --- Postgres --------------------------------------------------------------


@contextmanager
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    """Short-term state: one checkpoint per graph step, keyed by thread_id."""
    with PostgresSaver.from_conn_string(POSTGRES_URI) as saver:
        saver.setup()  # creates tables on first run; no-op afterwards
        yield saver


@contextmanager
def postgres_store() -> Iterator[PostgresStore]:
    """Long-term memory shared across threads.

    With an embedding model configured, values are embedded on put() and searchable by
    meaning (pgvector). Without one, it's a plain key-value store (no vectors).
    """
    configured = make_embeddings()
    index = None
    if configured:
        embeddings, dims = configured
        _check_vector_dims(dims)
        index = {"dims": dims, "embed": embeddings, "fields": ["text"]}
    with PostgresStore.from_conn_string(POSTGRES_URI, index=index) as store:
        store.setup()
        yield store


def _check_vector_dims(dims: int) -> None:
    """pgvector columns have a fixed size; switching embedding models needs a reset."""
    import psycopg

    with psycopg.connect(POSTGRES_URI) as conn:
        row = conn.execute(
            "select format_type(atttypid, atttypmod) from pg_attribute "
            "where attrelid = to_regclass('store_vectors') and attname = 'embedding'"
        ).fetchone()
    if row and row[0] != f"vector({dims})":
        raise SystemExit(
            f"Stored memories use {row[0]} but the embedding model returns vector({dims}). "
            "Switching embedding models needs a fresh store: ./scripts/reset.sh"
        )


# --- Langfuse --------------------------------------------------------------

if not ONLINE:
    # The scripted model has no provider model name; silence Langfuse's parse warning.
    logging.getLogger("langfuse").setLevel(logging.ERROR)

langfuse = get_client()


@contextmanager
def traced(
    name: str,
    *,
    session_id: str,
    trace_id: str | None = None,
    tags: Sequence[str] = (),
    input: Any = None,
    user_id: str | None = None,
) -> Iterator[tuple[Any, CallbackHandler]]:
    """Open a Langfuse trace (or continue `trace_id`) and yield (root span, LangChain handler).

    Pass the handler in the graph config's `callbacks`; every node, LLM call and
    tool call then nests under this root span. `session_id` groups traces in the UI.
    """
    with propagate_attributes(session_id=session_id, user_id=user_id, tags=["langgraph", *tags]):
        with langfuse.start_as_current_observation(
            name=name,
            as_type="agent",
            input=input,
            trace_context={"trace_id": trace_id} if trace_id else None,
        ) as span:
            yield span, CallbackHandler()


def flush() -> None:
    """Short-lived scripts must flush before exit or the last spans are lost."""
    langfuse.flush()


def sessions_url() -> str:
    return f"{os.environ['LANGFUSE_BASE_URL']}/project/langgraph-demo/sessions"


# --- Output helpers --------------------------------------------------------


def stream_run(graph, inp: Any, config: dict, *, subgraphs: bool = False, **kwargs) -> None:
    """Stream a run, printing node updates and LLM tokens as they arrive."""
    streaming_tokens = False
    for part in graph.stream(inp, config, stream_mode=["updates", "messages"], subgraphs=subgraphs, **kwargs):
        ns, (mode, chunk) = (part[0], part[1:]) if subgraphs else ((), part)
        prefix = "  " * len(ns)
        if mode == "messages":
            token, meta = chunk
            if isinstance(token, AIMessageChunk) and token.content:
                if not streaming_tokens:
                    print(f"{prefix}  ~ {meta.get('langgraph_node')}: ", end="")
                    streaming_tokens = True
                print(token.content, end="", flush=True)
            continue
        if streaming_tokens:
            print()
            streaming_tokens = False
        for node, update in chunk.items():
            where = f"{'/'.join(n.split(':')[0] for n in ns)}/" if ns else ""
            if node == "__interrupt__":
                print(f"{prefix}⏸  interrupted: {update[0].value.get('question', update[0].value)}")
            else:
                print(f"{prefix}▸ {where}{node}{describe_update(update)}")


def describe_update(update: Any) -> str:
    if not isinstance(update, dict):
        return ""
    bits = []
    for msg in update.get("messages", []) if isinstance(update.get("messages"), list) else []:
        if getattr(msg, "tool_calls", None):
            bits.append("calls " + ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls))
        elif msg.type == "tool":
            bits.append(f"tool result: {str(msg.content)[:80]}")
        elif msg.type == "remove":
            bits.append("remove msg")
    keys = [k for k in update if k != "messages"]
    if keys:
        bits.append("sets " + ", ".join(keys))
    return f"  ({'; '.join(bits)})" if bits else ""
