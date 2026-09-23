import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from evals.reliability.statistics import JUDGES, summarize


def archive(experiment_id: str, output: Path) -> None:
    client = Client()
    project = client.read_project(project_id=experiment_id)
    rows = list(
        client.get_experiment_results(project_id=experiment_id, preview=True)[
            "examples_with_runs"
        ]
    )
    records = []
    recorded_cases = []
    for case, row in enumerate(sorted(rows, key=lambda row: row.inputs["question"])):
        recorded_cases.append(
            {
                "inputs": {"question": row.inputs["question"]},
                "outputs": row.inputs["recorded_response"],
                "reference_outputs": row.outputs,
                "metadata": row.metadata,
            }
        )
        for trial, run in enumerate(row.runs, start=1):
            record = {"case": case, "trial": trial}
            for judge in JUDGES:
                stats = run.feedback_stats
                record[judge] = {
                    "quality": stats[f"{judge}_weather_quality"]["avg"],
                    "does_pass": stats[f"{judge}_weather_does_pass"]["avg"],
                    "choice": next(iter(stats[f"{judge}_weather_outcome"]["values"])),
                }
            records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "experiment": {
                    "id": experiment_id,
                    "name": project.name,
                    "started_at": project.start_time.isoformat(),
                    "metadata": project.metadata,
                },
                "recorded_cases": recorded_cases,
                "analysis": summarize(records, len(recorded_cases)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive a judge reliability benchmark.")
    parser.add_argument("experiment_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(override=True)
    archive(args.experiment_id, args.output)


if __name__ == "__main__":
    main()
