import argparse
import json
import os
import random
import statistics
from collections import Counter

from dotenv import load_dotenv


load_dotenv(override=True)

JUDGES = ("jev", "llm")
FROZEN_OUTPUT_KEY = "_frozen_output"
BOOTSTRAP_SAMPLES = 10_000


def _variance(values: list[float]) -> float:
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def _sample_variance(values: list[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _bootstrap_ci(values: list, statistic, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    samples = [
        statistic([values[rng.randrange(len(values))] for _ in values])
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return [None, None]
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _categorical_summary(values: list[str]) -> dict:
    counts = Counter(values)
    agreement = max(counts.values()) / len(values)
    return {
        "modal_choice": counts.most_common(1)[0][0],
        "agreement": agreement,
        "disagreement_rate": 1 - agreement,
    }


def _summarize(records: list[dict], case_count: int) -> dict:
    summary = {}
    case_values = {}
    for judge in JUDGES:
        summary[judge] = {}
        for metric in ("quality", "score"):
            values_by_case = [
                [record[judge][metric] for record in records if record["case"] == case]
                for case in range(case_count)
            ]
            case_values[judge, metric] = values_by_case
            case_means = [statistics.mean(values) for values in values_by_case]
            variances = [_sample_variance(values) for values in values_by_case]
            pooled_values = [value for values in values_by_case for value in values]
            summary[judge][metric] = {
                "mean": statistics.mean(pooled_values),
                "stddev": statistics.stdev(pooled_values),
                "mean_variance": statistics.mean(variances),
                "max_variance": max(variances),
                "mean_95_ci": _bootstrap_ci(case_means, statistics.mean, 101),
                "mean_variance_95_ci": _bootstrap_ci(variances, statistics.mean, 102),
            }

        choices_by_case = [
            [record[judge]["choice"] for record in records if record["case"] == case]
            for case in range(case_count)
        ]
        categorical = [_categorical_summary(values) for values in choices_by_case]
        disagreement_rates = [result["disagreement_rate"] for result in categorical]
        summary[judge]["choice"] = {
            "mean_disagreement_rate": statistics.mean(disagreement_rates),
            "max_disagreement_rate": max(disagreement_rates),
            "mean_disagreement_rate_95_ci": _bootstrap_ci(
                disagreement_rates, statistics.mean, 103
            ),
        }

    comparison = {}
    for metric in ("quality", "score"):
        jev_variance = summary["jev"][metric]["mean_variance"]
        llm_variance = summary["llm"][metric]["mean_variance"]
        jev_means = [statistics.mean(values) for values in case_values["jev", metric]]
        llm_means = [statistics.mean(values) for values in case_values["llm", metric]]
        jev_variances = [
            _sample_variance(values) for values in case_values["jev", metric]
        ]
        llm_variances = [
            _sample_variance(values) for values in case_values["llm", metric]
        ]
        mean_differences = [
            llm_mean - jev_mean for llm_mean, jev_mean in zip(llm_means, jev_means)
        ]
        variance_differences = [
            llm_variance - jev_variance
            for llm_variance, jev_variance in zip(llm_variances, jev_variances)
        ]
        comparison[metric] = {
            "jev_mean_variance": jev_variance,
            "llm_mean_variance": llm_variance,
            "llm_to_jev_variance_ratio": (
                llm_variance / jev_variance if jev_variance else None
            ),
            "mean_difference_llm_minus_jev": statistics.mean(mean_differences),
            "mean_difference_95_ci": _bootstrap_ci(
                mean_differences, statistics.mean, 201
            ),
            "variance_difference_llm_minus_jev": statistics.mean(variance_differences),
            "variance_difference_95_ci": _bootstrap_ci(
                variance_differences, statistics.mean, 202
            ),
            "variance_ratio_95_ci": _bootstrap_ci(
                list(zip(jev_variances, llm_variances)),
                lambda pairs: (
                    sum(llm for _, llm in pairs) / sum(jev for jev, _ in pairs)
                    if sum(jev for jev, _ in pairs)
                    else None
                ),
                203,
            ),
        }
    return {
        "analysis": {
            "variance_definition": "unbiased sample variance within each frozen case",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "confidence_level": 0.95,
            "bootstrap_unit": "frozen case",
        },
        "summary": summary,
        "comparison": comparison,
    }


def _evaluators() -> dict:
    from evals.judges import (
        jev_weather_choice,
        jev_weather_quality,
        jev_weather_score,
        llm_weather_choice,
        llm_weather_quality,
        llm_weather_score,
    )

    return {
        "jev": (jev_weather_quality, jev_weather_score, jev_weather_choice),
        "llm": (llm_weather_quality, llm_weather_score, llm_weather_choice),
    }


def _frozen_cases() -> list[tuple[dict, dict]]:
    from evals.dataset import EXAMPLES
    from evals.offline_evals import target

    return [(example, target(example["inputs"])) for example in EXAMPLES]


def _local_records(trials: int) -> tuple[list[dict], int]:
    evaluators = _evaluators()
    frozen_cases = _frozen_cases()
    records = []
    for trial in range(trials):
        for case, (example, outputs) in enumerate(frozen_cases):
            record = {"case": case, "trial": trial + 1}
            for judge in JUDGES:
                quality, score, choice = (
                    evaluator(example["inputs"], outputs, example["outputs"])
                    for evaluator in evaluators[judge]
                )
                record[judge] = {
                    "quality": quality["score"],
                    "score": score["score"],
                    "choice": choice["value"],
                }
            records.append(record)
    return records, len(frozen_cases)


def _frozen_target(inputs: dict) -> dict:
    return inputs[FROZEN_OUTPUT_KEY]


def _remote_examples(client) -> list:
    from evals.dataset import EXAMPLES, ensure_dataset
    from evals.offline_evals import target
    from langsmith.schemas import Example

    dataset, _ = ensure_dataset(client)
    existing = {
        example.inputs["question"]: example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    return [
        Example(
            id=existing[source["inputs"]["question"]].id,
            dataset_id=dataset.id,
            inputs={
                **source["inputs"],
                FROZEN_OUTPUT_KEY: target(source["inputs"]),
            },
            outputs=source["outputs"],
            metadata={"case": case, **(source.get("metadata") or {})},
        )
        for case, source in enumerate(EXAMPLES)
    ]


def run_langsmith(trials: int) -> dict:
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    examples = _remote_examples(client)
    evaluators = _evaluators()
    results = evaluate(
        _frozen_target,
        data=examples,
        evaluators=[evaluator for group in evaluators.values() for evaluator in group],
        num_repetitions=trials,
        client=client,
        experiment_prefix="judge-reliability",
        description="Compare Jev and LLM judge variance on frozen weather-agent outputs.",
        metadata={
            "judge_model": os.getenv("LLM_JUDGE_MODEL", "gpt-5.6-luna"),
        },
        max_concurrency=2,
    )
    records = []
    for row in results:
        record = {"case": row["example"].metadata["case"]}
        evaluation_results = row["evaluation_results"]
        result_items = (
            evaluation_results.results
            if hasattr(evaluation_results, "results")
            else evaluation_results["results"]
        )
        for result in result_items:
            key = result.key if hasattr(result, "key") else result["key"]
            judge, metric = key.split("_weather_", 1)
            value = result.value if hasattr(result, "value") else result["value"]
            score = result.score if hasattr(result, "score") else result["score"]
            record.setdefault(judge, {})["choice" if metric == "outcome" else metric] = (
                value if metric == "outcome" else score
            )
        records.append(record)
    return {
        "experiment_name": results.experiment_name,
        "experiment_url": results.url,
        "trials": trials,
        "cases": len(examples),
        **_summarize(records, len(examples)),
    }


def run_local(trials: int) -> dict:
    records, case_count = _local_records(trials)
    return {
        "trials": trials,
        "cases": case_count,
        **_summarize(records, case_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Jev and LLM judge reliability.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    result = run_local(args.trials) if args.local else run_langsmith(args.trials)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
