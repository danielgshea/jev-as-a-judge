import argparse
import json
import statistics

from dotenv import load_dotenv
from langsmith import Client

from evals.analysis.common import JUDGE_LABELS, NUMERIC_METRICS, grouped_runs, project_id, select_repetitions
from evals.config import DECISION_JUDGES, LLM_JUDGES


JUDGE_MODELS = {**DECISION_JUDGES, **LLM_JUDGES}
SEMIF_COST_ESTIMATE = {
    "basis": (
        "Midpoint of public Qwen3.5 9B rates used as a conservative proxy "
        "for SemIf's Qwen3.5 4B base"
    ),
    "input_usd_per_million": 0.135,
    "output_usd_per_million": 0.20,
    "range": {
        "input_usd_per_million": [0.10, 0.17],
        "output_usd_per_million": [0.15, 0.25],
    },
    "sources": [
        "https://openrouter.ai/api/v1/models",
        "https://www.together.ai/pricing",
    ],
    "token_estimate": "Serialized trace payload UTF-8 bytes divided by four",
}


def _source_run_id(feedback) -> str | None:
    source = feedback.feedback_source.model_dump()
    return source.get("metadata", {}).get("__run", {}).get("run_id")


def _feedback_keys(
    judges: list[str] | None,
    metrics: list[str] | None = None,
) -> list[str] | None:
    if judges is None and metrics is None:
        return None
    return [
        f"{judge}_weather_{metric}"
        for judge in (judges or JUDGE_LABELS)
        for metric in (metrics or (*NUMERIC_METRICS, "outcome"))
    ]


def _chunks(values: list[str], size: int = 100):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _trace_payload(run) -> tuple[dict, dict]:
    arguments = run.inputs or {}
    inputs = arguments.get("inputs", {})
    outputs = arguments.get("outputs", {})
    reference_outputs = arguments.get("reference_outputs", {})
    evidence = outputs.get("evidence", [])
    if evidence and all(isinstance(item, str) for item in evidence):
        evidence = "\n\n".join(evidence)[:4_000]
    state = {
        "user_question": inputs.get("question"),
        "expected_behavior": reference_outputs.get(
            "expected_behavior", reference_outputs
        ),
        "final_answer": outputs.get("answer"),
        "tool_calls": outputs.get("tool_calls", []),
        "search_evidence": evidence,
    }
    return state, run.outputs or {}


def _token_counts(judge: str, run) -> tuple[int | None, int | None, str | None]:
    if run is None:
        return None, None, None
    input_tokens = run.prompt_tokens
    output_tokens = run.completion_tokens
    if judge != "semif" or input_tokens or output_tokens:
        return input_tokens, output_tokens, "reported"
    traced_input, traced_output = _trace_payload(run)
    # ponytail: four bytes per token is approximate; use the Qwen tokenizer if billing precision matters.
    def estimate(payload: dict) -> int:
        size = len(json.dumps(payload, sort_keys=True).encode())
        return max(1, (size + 3) // 4)

    return (
        estimate(traced_input),
        estimate(traced_output),
        "estimated_from_trace_payload",
    )


def _cost(
    judge: str,
    reported_cost: float | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> tuple[float | None, list[float] | None, str | None]:
    if judge != "semif" or reported_cost:
        return reported_cost, None, "reported" if reported_cost is not None else None
    if input_tokens is None or output_tokens is None:
        return None, None, None
    estimate = SEMIF_COST_ESTIMATE
    midpoint = (
        input_tokens * estimate["input_usd_per_million"]
        + output_tokens * estimate["output_usd_per_million"]
    ) / 1_000_000
    low = (
        input_tokens * estimate["range"]["input_usd_per_million"][0]
        + output_tokens * estimate["range"]["output_usd_per_million"][0]
    ) / 1_000_000
    high = (
        input_tokens * estimate["range"]["input_usd_per_million"][1]
        + output_tokens * estimate["range"]["output_usd_per_million"][1]
    ) / 1_000_000
    return midpoint, [low, high], "estimated_qwen3.5_9b_proxy"


def _summarize(records: list[dict], repetitions: int) -> dict:
    summary = {}
    for judge in JUDGE_LABELS.values():
        judge_records = [record for record in records if record["judge"] == judge]
        if not judge_records:
            continue
        successful = [record for record in judge_records if record["status"] == "success"]
        complete = [
            record
            for record in successful
            if all(
                record[field] is not None
                for field in ("input_tokens", "output_tokens", "latency_seconds")
            )
        ]
        latency_records = [
            record for record in successful if record["latency_seconds"] is not None
        ]
        cost_records = [record for record in successful if record["total_cost"] is not None]
        estimated_cost_records = [
            record
            for record in cost_records
            if record.get("cost_source", "").startswith("estimated")
        ]
        summary[judge] = {
            "calls": len(judge_records),
            "successful": len(successful),
            "with_metrics": len(complete),
            "with_cost": len(cost_records),
            "with_estimated_cost": len(estimated_cost_records),
            "all_successful": len(successful) == len(judge_records),
            "all_metrics_present": len(complete) == len(judge_records),
            "average_input_tokens": statistics.fmean(record["input_tokens"] for record in complete) if complete else None,
            "average_output_tokens": statistics.fmean(record["output_tokens"] for record in complete) if complete else None,
            "average_latency_seconds": statistics.fmean(record["latency_seconds"] for record in latency_records) if latency_records else None,
            "latency_by_repetition_seconds": [
                statistics.fmean(values) if values else None
                for repetition in range(repetitions)
                if (
                    values := [
                        record["latency_seconds"]
                        for record in latency_records
                        if record["repetition"] == repetition
                    ]
                )
            ],
            "latency_seconds": [record["latency_seconds"] for record in latency_records],
            "total_cost": sum(record["total_cost"] for record in cost_records),
            "average_cost": statistics.fmean(record["total_cost"] for record in cost_records)
            if cost_records
            else None,
            "estimated_total_cost_range": (
                [
                    sum(record["cost_range"][0] for record in estimated_cost_records),
                    sum(record["cost_range"][1] for record in estimated_cost_records),
                ]
                if estimated_cost_records
                else None
            ),
            "cost_estimate": SEMIF_COST_ESTIMATE if estimated_cost_records else None,
            "token_sources": sorted(
                {
                    record["token_source"]
                    for record in successful
                    if record.get("token_source")
                }
            ),
            "models": sorted({record["model"] for record in complete if record["model"]}),
        }
    return summary


def query_metrics(
    experiment: str,
    judges: list[str] | None = None,
    metrics: list[str] | None = None,
    repetitions: int | None = None,
) -> dict:
    client = Client()
    experiment_id = project_id(client, experiment)
    run_groups, repetition_count = select_repetitions(
        grouped_runs(list(client.list_runs(project_id=experiment_id, is_root=True))), repetitions
    )
    target_runs = [run for group in run_groups for run in group]
    repetitions_by_run = {
        str(run.id): repetition
        for group in run_groups
        for repetition, run in enumerate(group)
    }
    feedback = [
        result
        for run_ids in _chunks([run.id for run in target_runs])
        for result in client.list_feedback(
            run_ids=run_ids,
            feedback_key=_feedback_keys(judges, metrics),
        )
    ]
    feedback = [
        result
        for result in feedback
        if "_weather_" in result.key
        and result.key.split("_weather_", 1)[0] in (judges or JUDGE_LABELS)
        and result.key.split("_weather_", 1)[1] in (metrics or (*NUMERIC_METRICS, "outcome"))
    ]
    source_run_ids = {
        source_run_id
        for result in feedback
        if (source_run_id := _source_run_id(result)) is not None
    }
    evaluator_runs = {
        str(run.id): run
        for run_ids in _chunks(sorted(source_run_ids))
        for run in client.list_runs(run_ids=run_ids, limit=len(run_ids))
    }
    records = []
    for result in feedback:
        judge_key = result.key.split("_weather_", 1)[0]
        run = evaluator_runs.get(str(_source_run_id(result)))
        input_tokens, output_tokens, token_source = _token_counts(judge_key, run)
        reported_cost = (
            float(run.total_cost) if run and run.total_cost is not None else None
        )
        total_cost, cost_range, cost_source = _cost(
            judge_key, reported_cost, input_tokens, output_tokens
        )
        records.append(
            {
                "judge": JUDGE_LABELS[judge_key],
                "key": result.key,
                "repetition": repetitions_by_run[str(result.run_id)],
                "status": run.status.lower() if run else "missing",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "token_source": token_source,
                "latency_seconds": run.latency if run else None,
                "total_cost": total_cost,
                "cost_range": cost_range,
                "cost_source": cost_source,
                "model": JUDGE_MODELS[judge_key].model,
            }
        )
    summary = _summarize(records, repetition_count)
    return {
        "experiment_id": experiment_id,
        "cases": len(run_groups),
        "repetitions": repetition_count,
        "target_runs": len(target_runs),
        "feedback_results": len(feedback),
        "all_successful": all(item["all_successful"] for item in summary.values()),
        "all_metrics_present": all(item["all_metrics_present"] for item in summary.values()),
        "total_cost": sum(item["total_cost"] for item in summary.values()),
        "judges": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LangSmith judge trace metrics.")
    parser.add_argument("experiment", help="Experiment name, project UUID, or comparison URL")
    parser.add_argument("--judges", nargs="+", choices=JUDGE_LABELS, help="Only include these judges")
    parser.add_argument("--metrics", nargs="+", choices=(*NUMERIC_METRICS, "outcome"), help="Only include these feedback metrics")
    parser.add_argument("--repetitions", type=int, help="Use this many repetitions per case")
    args = parser.parse_args()
    load_dotenv(override=True)
    print(json.dumps(query_metrics(args.experiment, args.judges, args.metrics, args.repetitions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
