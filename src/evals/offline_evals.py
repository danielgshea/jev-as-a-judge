from pprint import pprint

from langsmith import Client

from evals.dataset import DATASET_NAME, EXAMPLES, ensure_dataset
from evals.judges import jev_weather_quality
from weather_agent.agent import agent


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
        evaluators=[jev_weather_quality],
        experiment_prefix="weather-agent",
        max_concurrency=1,
    )
    print(results)


def run_local() -> None:
    scores = []
    for index, example in enumerate(EXAMPLES, start=1):
        outputs = target(example["inputs"])
        evaluation = jev_weather_quality(
            example["inputs"], outputs, example["outputs"]
        )
        scores.append(evaluation["score"])

        print(f"\nExample {index}/{len(EXAMPLES)}: {example['inputs']['question']}")
        pprint(
            {
                "answer": outputs["answer"],
                "tool_calls": outputs["tool_calls"],
                "evaluation": evaluation,
            },
            sort_dicts=False,
        )

    print(f"\nAverage score: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    run()
