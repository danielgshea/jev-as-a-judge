import random
import statistics
from collections import Counter

from evals.config import DECISION_JUDGES, LLM_JUDGES


JUDGES = tuple((*DECISION_JUDGES, *LLM_JUDGES))
BOOTSTRAP_SAMPLES = 10_000


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
    return (
        [_percentile(samples, 0.025), _percentile(samples, 0.975)]
        if samples
        else [None, None]
    )


def _categorical_summary(values: list[str]) -> dict:
    counts = Counter(values)
    agreement = max(counts.values()) / len(values)
    return {"disagreement_rate": 1 - agreement}


def summarize(records: list[dict], case_count: int) -> dict:
    summary = {}
    case_values = {}
    for judge in JUDGES:
        summary[judge] = {}
        for metric in ("quality", "does_pass"):
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
                "stddev": statistics.stdev(pooled_values) if len(pooled_values) > 1 else 0.0,
                "mean_variance": statistics.mean(variances),
                "max_variance": max(variances),
                "mean_95_ci": _bootstrap_ci(case_means, statistics.mean, 101),
                "mean_variance_95_ci": _bootstrap_ci(variances, statistics.mean, 102),
            }
        disagreement_rates = [
            _categorical_summary(
                [record[judge]["choice"] for record in records if record["case"] == case]
            )["disagreement_rate"]
            for case in range(case_count)
        ]
        summary[judge]["choice"] = {
            "mean_disagreement_rate": statistics.mean(disagreement_rates),
            "max_disagreement_rate": max(disagreement_rates),
            "mean_disagreement_rate_95_ci": _bootstrap_ci(
                disagreement_rates, statistics.mean, 103
            ),
        }

    comparison = {judge: {} for judge in JUDGES[1:]}
    for judge in JUDGES[1:]:
        for metric in ("quality", "does_pass"):
            jev_variances = [
                _sample_variance(values) for values in case_values["jev", metric]
            ]
            judge_variances = [
                _sample_variance(values) for values in case_values[judge, metric]
            ]
            mean_differences = [
                statistics.mean(right) - statistics.mean(left)
                for left, right in zip(case_values["jev", metric], case_values[judge, metric])
            ]
            variance_differences = [
                right - left for left, right in zip(jev_variances, judge_variances)
            ]
            jev_variance = summary["jev"][metric]["mean_variance"]
            judge_variance = summary[judge][metric]["mean_variance"]
            comparison[judge][metric] = {
                "jev_mean_variance": jev_variance,
                "judge_mean_variance": judge_variance,
                "judge_to_jev_variance_ratio": judge_variance / jev_variance if jev_variance else None,
                "mean_difference_judge_minus_jev": statistics.mean(mean_differences),
                "mean_difference_95_ci": _bootstrap_ci(mean_differences, statistics.mean, 201),
                "variance_difference_judge_minus_jev": statistics.mean(variance_differences),
                "variance_difference_95_ci": _bootstrap_ci(variance_differences, statistics.mean, 202),
                "variance_ratio_95_ci": _bootstrap_ci(
                    list(zip(jev_variances, judge_variances)),
                    lambda pairs: (
                        sum(right for _, right in pairs) / sum(left for left, _ in pairs)
                        if sum(left for left, _ in pairs)
                        else None
                    ),
                    203,
                ),
            }
    return {
        "analysis": {
            "variance_definition": "unbiased sample variance within each recorded case",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "confidence_level": 0.95,
            "bootstrap_unit": "recorded case",
        },
        "summary": summary,
        "comparison": comparison,
    }
