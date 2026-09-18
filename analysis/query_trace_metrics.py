import argparse
import json
import statistics

from dotenv import load_dotenv
from langsmith import Client

from analysis.common import JUDGE_LABELS, NUMERIC_METRICS, grouped_runs, project_id, select_repetitions
from evals.model import LLM_JUDGES


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


def _summarize(records: list[dict]) -> dict:
    summary = {}
    for judge in JUDGE_LABELS.values():
        judge_records = [record for record in records if record["judge"] == judge]
        if not judge_records:
            continue
        successful = [record for record in judge_records if record["status"] == "success"]
        complete = [
            record
            for record in successful
            if all(record[field] is not None for field in ("input_tokens", "output_tokens", "latency_seconds"))
        ]
        cost_records = [record for record in successful if record["total_cost"] is not None]
        summary[judge] = {
            "calls": len(judge_records),
            "successful": len(successful),
            "with_metrics": len(complete),
            "with_cost": len(cost_records),
            "all_successful": len(successful) == len(judge_records),
            "all_metrics_present": len(complete) == len(judge_records),
            "average_input_tokens": statistics.fmean(record["input_tokens"] for record in complete) if complete else None,
            "average_output_tokens": statistics.fmean(record["output_tokens"] for record in complete) if complete else None,
            "average_latency_seconds": statistics.fmean(record["latency_seconds"] for record in complete) if complete else None,
            "total_cost": sum(record["total_cost"] for record in cost_records),
            "average_cost": statistics.fmean(record["total_cost"] for record in cost_records)
            if cost_records
            else None,
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
        records.append(
            {
                "judge": JUDGE_LABELS[judge_key],
                "key": result.key,
                "status": run.status.lower() if run else "missing",
                "input_tokens": run.prompt_tokens if run else None,
                "output_tokens": run.completion_tokens if run else None,
                "latency_seconds": run.latency if run else None,
                "total_cost": float(run.total_cost) if run and run.total_cost is not None else None,
                "model": LLM_JUDGES[judge_key].model if judge_key in LLM_JUDGES else "Jev",
            }
        )
    summary = _summarize(records)
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
