from pprint import pprint

from langsmith import Client

from evals.dataset import DATASET_NAME, EXAMPLES, ensure_dataset
from evals.judges import (
    LLM_EVALUATORS,
    jev_weather_choice,
    jev_weather_does_pass,
    jev_weather_quality,
)
from weather_agent.agent import agent


EVALUATORS = {
    "jev": (jev_weather_quality, jev_weather_does_pass, jev_weather_choice),
    **LLM_EVALUATORS,
}


def _content(message) -> str:
    content = message.get("content", "") if isinstance(message, dict) else message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def target(inputs: dict) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
    messages = result["messages"]
    tool_calls = [
        call
        for message in messages
        for call in (
            message.get("tool_calls", [])
            if isinstance(message, dict)
            else getattr(message, "tool_calls", [])
        )
    ]
    return {
        "answer": _content(messages[-1]),
        "tool_calls": [call["name"] for call in tool_calls],
        "evidence": [
            _content(message)
            for message in messages
            if (isinstance(message, dict) and message.get("role") == "tool")
            or getattr(message, "type", None) == "tool"
        ],
    }


def run() -> None:
    client = Client()
    dataset, created = ensure_dataset(client)
    print(f"{'Created' if created else 'Using existing'} dataset: {DATASET_NAME}")
    results = client.evaluate(
        target,
        data=dataset.id,
        evaluators=[evaluator for group in EVALUATORS.values() for evaluator in group],
        experiment_prefix="weather-agent",
        max_concurrency=1,
    )
    print(results)


def run_local() -> None:
    scores = []
    for index, example in enumerate(EXAMPLES, start=1):
        outputs = target(example["inputs"])
        evaluations = {
            judge: {
                "quality": quality(example["inputs"], outputs, example["outputs"]),
                "does_pass": does_pass(example["inputs"], outputs, example["outputs"]),
                "choice": choice(example["inputs"], outputs, example["outputs"]),
            }
            for judge, (quality, does_pass, choice) in EVALUATORS.items()
        }
        scores.extend(
            evaluation["quality"]["score"] for evaluation in evaluations.values()
        )

        print(f"\nExample {index}/{len(EXAMPLES)}: {example['inputs']['question']}")
        pprint(
            {
                "answer": outputs["answer"],
                "tool_calls": outputs["tool_calls"],
                "evaluations": evaluations,
            },
            sort_dicts=False,
        )

    print(f"\nAverage score: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    run()
