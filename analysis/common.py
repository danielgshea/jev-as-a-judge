import re
from collections import defaultdict

from evals.model import LLM_JUDGES


UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
NUMERIC_METRICS = ("quality", "does_pass")
JUDGE_LABELS = {"jev": "Jev", **{key: config.label for key, config in LLM_JUDGES.items()}}


def project_id(client, experiment: str) -> str:
    selected_session = re.search(rf"selectedSessions=({UUID_PATTERN})", experiment)
    if selected_session:
        return selected_session.group(1)
    if re.fullmatch(UUID_PATTERN, experiment):
        return experiment
    return str(client.read_project(project_name=experiment).id)


def select_repetitions(groups: list[list], repetitions: int | None) -> tuple[list[list], int]:
    if not groups or not all(groups):
        raise ValueError("Experiment has no complete repetitions.")
    available = min(len(group) for group in groups)
    count = repetitions or available
    if count < 1 or count > available:
        raise ValueError(f"Repetitions must be between 1 and {available}.")
    return [group[:count] for group in groups], count


def grouped_runs(runs: list) -> list[list]:
    groups = defaultdict(list)
    for run in runs:
        groups[str(run.reference_example_id)].append(run)
    return [sorted(group, key=lambda run: run.start_time or "") for group in groups.values()]
