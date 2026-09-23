import json
import os

from dotenv import load_dotenv
from langsmith import Client

from evals.datasets.cases import CASES, ORACLE_BY_QUESTION
from evals.judges.state import judge_state_json
from weather_agent.agent import create_weather_agent


DATASET_NAME = "weather-agent-recorded-context-safe-v1"
GENERATOR_MODEL = "openai:gpt-5.5"
MAX_STATE_CHARS = 4_000


def _content(message) -> str:
    content = message.get("content", "") if isinstance(message, dict) else message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _tool_calls(message) -> list:
    return (
        message.get("tool_calls", [])
        if isinstance(message, dict)
        else getattr(message, "tool_calls", [])
    )


def _evidence(message) -> list[dict[str, str]]:
    if not (
        (isinstance(message, dict) and message.get("role") == "tool")
        or getattr(message, "type", None) == "tool"
    ):
        return []
    content = _content(message)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def recorded_response(result: dict) -> dict:
    messages = result["messages"]
    return {
        "answer": _content(messages[-1]),
        "tool_calls": [call["name"] for message in messages for call in _tool_calls(message)],
        "evidence": [card for message in messages for card in _evidence(message)],
    }


def parse_example(example) -> dict:
    expected_behavior = example.outputs["expected_behavior"]
    return {
        "question": example.inputs["question"],
        "recorded_response": example.inputs["recorded_response"],
        "expected_behavior": expected_behavior,
        "oracle": example.outputs["oracle"],
        "metadata": {
            "category": example.metadata["category"],
            "generator_model": example.metadata["generator_model"],
            "state_size": example.metadata["state_size"],
        },
    }


def build_examples(agent) -> list[dict]:
    examples = []
    for case in CASES:
        response = recorded_response(
            agent.invoke({"messages": [{"role": "user", "content": case["question"]}]})
        )
        inputs = {"question": case["question"], "recorded_response": response}
        outputs = {
            "expected_behavior": case["expected_behavior"],
            "oracle": ORACLE_BY_QUESTION[case["question"]],
        }
        state_size = len(judge_state_json(inputs, response, outputs))
        if state_size >= MAX_STATE_CHARS:
            raise ValueError(f"{case['question']!r}: judge state is {state_size} characters")
        examples.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "metadata": {
                    "category": case["category"],
                    "generator_model": GENERATOR_MODEL,
                    "state_size": state_size,
                },
            }
        )
    return examples


def create_recorded_dataset(client: Client, agent) -> None:
    if next(client.list_datasets(dataset_name=DATASET_NAME, limit=1), None):
        raise RuntimeError(f"Dataset already exists: {DATASET_NAME}")
    examples = build_examples(agent)
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Context-safe recorded GPT-5.5 weather-agent responses.",
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    actual = sorted(
        (parse_example(example) for example in client.list_examples(dataset_id=dataset.id)),
        key=lambda example: example["question"],
    )
    expected = sorted(
        (
            {
                "question": example["inputs"]["question"],
                "recorded_response": example["inputs"]["recorded_response"],
                "expected_behavior": example["outputs"]["expected_behavior"],
                "oracle": example["outputs"]["oracle"],
                "metadata": example["metadata"],
            }
            for example in examples
        ),
        key=lambda example: example["question"],
    )
    if actual != expected:
        raise RuntimeError("Uploaded examples differ from the generated examples")


def main() -> None:
    load_dotenv(override=True)
    os.environ["LANGSMITH_GATEWAY"] = "true"
    create_recorded_dataset(
        Client(),
        create_weather_agent(
            model=GENERATOR_MODEL,
            tavily_api_key=os.environ["TAVILY_API_KEY"],
        ),
    )
    print(f"Created and verified dataset: {DATASET_NAME}")


if __name__ == "__main__":
    main()
