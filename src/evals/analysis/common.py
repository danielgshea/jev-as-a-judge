import re
from collections import defaultdict

from evals.config import DECISION_JUDGES, LLM_JUDGES


UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
NUMERIC_METRICS = ("quality", "does_pass")
JUDGE_LABELS = {
    **{key: config.label for key, config in DECISION_JUDGES.items()},
    **{key: config.label for key, config in LLM_JUDGES.items()},
}
QUALITY_FIELDS = ("is_grounded", "matches_search_expectation", "is_useful")


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


def _oracle_labels(labels: list[dict], questions: list[str]) -> dict[str, dict]:
    by_question = {}
    for label in labels:
        question = label.get("question")
        if not isinstance(question, str) or question in by_question:
            raise ValueError("Each oracle label must have a unique question.")
        for field in (*QUALITY_FIELDS, "does_pass"):
            if label.get(field) not in (0, 1) or isinstance(label.get(field), bool):
                raise ValueError(f"{question!r}: {field} must be 0 or 1.")
        by_question[question] = label
    if set(by_question) != set(questions):
        raise ValueError("Oracle labels must match the archived recorded-case questions exactly.")
    return by_question


def _oracle_metric(data: dict, metric: str, judge: str, case_count: int) -> list[list[float]]:
    values = data["values"].get(metric, {}).get(judge)
    if not isinstance(values, list) or len(values) != case_count:
        raise ValueError(f"{judge}: incomplete {metric} values.")
    if not all(isinstance(case, list) and case for case in values):
        raise ValueError(f"{judge}: empty {metric} case values.")
    if not all(isinstance(value, (int, float)) for case in values for value in case):
        raise ValueError(f"{judge}: non-numeric {metric} value.")
    return values


def score_oracle(data: dict, recorded_cases: list[dict], oracle: dict, tolerance: float = 0.10) -> dict:
    questions = [case["inputs"]["question"] for case in recorded_cases]
    labels = _oracle_labels(oracle.get("labels", []), questions)
    report = {}
    for _, judge in data["judges"]:
        quality_cases = _oracle_metric(data, "quality", judge, len(questions))
        pass_cases = _oracle_metric(data, "does_pass", judge, len(questions))
        if [len(case) for case in quality_cases] != [len(case) for case in pass_cases]:
            raise ValueError(f"{judge}: quality and does_pass repetitions differ.")

        pass_correct = 0
        quality_errors = []
        cases = []
        for question, quality_values, pass_values in zip(questions, quality_cases, pass_cases):
            label = labels[question]
            quality_oracle = sum(label[field] for field in QUALITY_FIELDS) / len(QUALITY_FIELDS)
            errors = [abs(value - quality_oracle) for value in quality_values]
            pass_correct += sum(value == label["does_pass"] for value in pass_values)
            quality_errors.extend(errors)
            cases.append(
                {
                    "question": question,
                    "repetitions": len(pass_values),
                    "does_pass_oracle": label["does_pass"],
                    "does_pass_accuracy": sum(value == label["does_pass"] for value in pass_values)
                    / len(pass_values),
                    "quality_oracle": quality_oracle,
                    "quality_mae": sum(errors) / len(errors),
                    "quality_within_tolerance": sum(error <= tolerance + 1e-12 for error in errors)
                    / len(errors),
                }
            )
        predictions = sum(len(values) for values in pass_cases)
        report[judge] = {
            "predictions": predictions,
            "does_pass_accuracy": pass_correct / predictions,
            "quality_mae": sum(quality_errors) / len(quality_errors),
            "quality_within_tolerance": sum(error <= tolerance + 1e-12 for error in quality_errors)
            / len(quality_errors),
            "cases": cases,
        }
    return {"cases": len(questions), "quality_tolerance": tolerance, "judges": report}
