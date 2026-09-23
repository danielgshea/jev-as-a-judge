import argparse
import json

from dotenv import load_dotenv

from evals.config import DECISION_JUDGES, LLM_JUDGES
from evals.datasets.recorded import DATASET_NAME
from evals.reliability.statistics import JUDGES, summarize


def recorded_target(inputs: dict) -> dict:
    return inputs["recorded_response"]


def _judge_metric(key: str) -> tuple[str, str] | None:
    judge, separator, metric = key.partition("_weather_")
    return (judge, metric) if separator and judge in JUDGES else None


def _evaluators() -> dict:
    from evals.judges.llm import build_llm_evaluators
    from evals.judges.system_one import DECISION_EVALUATORS

    return {**DECISION_EVALUATORS, **build_llm_evaluators()}


def _records(results, questions: list[str]) -> list[dict]:
    case_by_question = {question: case for case, question in enumerate(questions)}
    records = []
    for row in results:
        record = {"case": case_by_question[row["example"].inputs["question"]]}
        evaluation_results = row["evaluation_results"]
        result_items = (
            evaluation_results.results
            if hasattr(evaluation_results, "results")
            else evaluation_results["results"]
        )
        for result in result_items:
            key = result.key if hasattr(result, "key") else result["key"]
            if not (judge_metric := _judge_metric(key)):
                continue
            judge, metric = judge_metric
            value = result.value if hasattr(result, "value") else result["value"]
            score = result.score if hasattr(result, "score") else result["score"]
            record.setdefault(judge, {})["choice" if metric == "outcome" else metric] = (
                value if metric == "outcome" else score
            )
        records.append(record)
    return records


def run_langsmith(trials: int, max_concurrency: int = 2) -> dict:
    from langsmith import Client
    from langsmith.evaluation import evaluate

    from evals.judges.system_one import verify_decision_models

    verify_decision_models()
    client = Client()
    examples = list(client.list_examples(dataset_name=DATASET_NAME))
    if not examples:
        raise RuntimeError(f"Dataset is empty or missing: {DATASET_NAME}")
    questions = sorted(example.inputs["question"] for example in examples)
    evaluators = _evaluators()
    results = evaluate(
        recorded_target,
        data=DATASET_NAME,
        evaluators=[evaluator for group in evaluators.values() for evaluator in group],
        num_repetitions=trials,
        client=client,
        experiment_prefix="judge-reliability-all",
        description="Compare Jev, SemIf, Luna, Terra, and Sonnet on recorded weather-agent responses.",
        metadata={
            "judge_models": {
                key: config.model
                for key, config in {**DECISION_JUDGES, **LLM_JUDGES}.items()
            }
        },
        max_concurrency=max_concurrency,
    )
    records = _records(results, questions)
    return {
        "experiment_name": results.experiment_name,
        "experiment_url": results.url,
        "trials": trials,
        "cases": len(examples),
        **summarize(records, len(examples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare judge reliability on recorded responses.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    load_dotenv(override=True)
    print(json.dumps(run_langsmith(args.trials, args.max_concurrency), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
