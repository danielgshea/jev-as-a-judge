from evals.judge_reliability import _categorical_summary, _summarize, _variance


assert _variance([0.0, 1.0, 2.0]) == 2 / 3
assert abs(_categorical_summary(["answered", "answered", "poor"])["disagreement_rate"] - 1 / 3) < 1e-12

records = [
    {
        "case": case,
        "jev": {"quality": 0.5, "score": 0.5, "choice": "answered"},
        "llm": {"quality": 0.6, "score": 1.0, "choice": "answered"},
    }
    for case in range(5)
    for _ in range(3)
]
summary = _summarize(records, 5)
assert summary["analysis"]["bootstrap_samples"] == 10_000
assert len(summary["comparison"]["quality"]["variance_ratio_95_ci"]) == 2
