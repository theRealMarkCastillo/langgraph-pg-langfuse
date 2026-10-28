"""Langfuse datasets + experiments: evaluate the multi-agent router on labelled examples.

Creates (or updates) the dataset `support-routing`, runs the supervisor graph on
every item, and scores each run. Results show up in Langfuse under
Datasets -> support-routing -> Runs, where runs (e.g. different models or prompt
versions) can be compared side by side.

    uv run python -m demos.experiments
    uv run python -m demos.experiments --run-name "llama3.1-v2-prompt"
"""

import argparse
import uuid
from datetime import datetime

from langfuse.langchain import CallbackHandler

from demos import multi_agent
from demos.common import flush, langfuse, postgres_checkpointer
from langfuse import Evaluation

DATASET = "support-routing"
ITEMS = [
    ("I was charged twice this month", ["billing"]),
    ("Can I get a refund for my subscription?", ["billing"]),
    ("Where can I download last month's invoice?", ["billing"]),
    ("The app crashes every time I open it", ["tech"]),
    ("I can't log in, it says invalid password", ["tech"]),
    ("I was charged twice and now the app won't load", ["billing", "tech"]),
    ("My payment failed and I get an error on checkout", ["billing", "tech"]),
    ("Thanks, that's all!", []),
    ("My card was declined at checkout", ["billing"]),  # no obvious keyword: a harder case
]


def ensure_dataset() -> None:
    langfuse.create_dataset(name=DATASET, description="Customer messages labelled with the specialists needed")
    for i, (text, expected) in enumerate(ITEMS):
        # A stable id makes this an upsert, so re-running doesn't duplicate items.
        langfuse.create_dataset_item(
            dataset_name=DATASET, id=f"routing-{i}", input={"message": text}, expected_output={"specialists": expected}
        )


def route_exact(*, input, output, expected_output, **kwargs) -> Evaluation:
    ok = sorted(output["specialists"]) == sorted(expected_output["specialists"])
    return Evaluation(
        name="route_exact",
        value=1.0 if ok else 0.0,
        comment=None if ok else f"expected {expected_output['specialists']}, got {output['specialists']}",
    )


def route_recall(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Partial credit: share of needed specialists that were actually consulted."""
    need = set(expected_output["specialists"])
    got = set(output["specialists"])
    return Evaluation(name="route_recall", value=1.0 if not need else len(need & got) / len(need))


def accuracy(*, item_results, **kwargs) -> Evaluation:
    scores = [e.value for r in item_results for e in r.evaluations if e.name == "route_exact"]
    return Evaluation(name="route_accuracy", value=sum(scores) / len(scores) if scores else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    ensure_dataset()
    dataset = langfuse.get_dataset(DATASET)

    with postgres_checkpointer() as checkpointer:
        graph = multi_agent.build_graph(checkpointer)

        def task(*, item, **kwargs) -> dict:
            # run_experiment opens a trace per item; the handler nests the graph under it.
            values = multi_agent.run(
                graph, item.input["message"], f"exp-{uuid.uuid4().hex[:8]}", CallbackHandler(), show=False
            )
            return {
                "specialists": values["handled"],
                "replies": [m.content for m in values["messages"] if m.type == "ai"],
            }

        result = dataset.run_experiment(
            name="router-eval",
            run_name=args.run_name or f"router-{datetime.now():%Y%m%d-%H%M%S}",
            description="Does the supervisor consult exactly the right specialists?",
            task=task,
            evaluators=[route_exact, route_recall],
            run_evaluators=[accuracy],
            max_concurrency=4,
        )

    print(result.format())
    for r in result.item_results:
        for e in r.evaluations:
            if e.name == "route_exact" and e.value < 1:
                print(f"  ✘ {r.item.input['message']!r}: {e.comment}")
    flush()


if __name__ == "__main__":
    main()
