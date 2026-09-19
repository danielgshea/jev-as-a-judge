import argparse
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from analysis.common import JUDGE_LABELS, NUMERIC_METRICS, project_id, score_oracle, select_repetitions
from analysis.query_trace_metrics import query_metrics


BACKGROUND = "#030710"
BORDER = "#1A2740"
TEXT = "#FFFFFF"
MUTED_TEXT = "#C8DDF0"
COLORS = ("#7FC8FF", "#E3FF8F", "#B27D75", "#C78EAD", "#D5C3F7")


def _blend(start: str, end: str, amount: float) -> str:
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{round(a + (b - a) * amount):02X}" for a, b in zip(start_rgb, end_rgb))


def _text(x: float, y: float, value: str, size: int = 14, anchor: str = "start", color: str = TEXT) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Lausanne, Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="{color}">'
        f"{html.escape(value)}</text>"
    )


def _svg(width: int, height: int, parts: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
            *parts,
            "</svg>",
        ]
    )


def _axis(parts: list[str], left: float, right: float, top: float, bottom: float, maximum: float) -> None:
    for fraction in range(6):
        value = maximum * fraction / 5
        y = bottom - (bottom - top) * fraction / 5
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{BORDER}" stroke-width="1"/>'
        )
        parts.append(_text(left - 10, y + 5, f"{value:.3g}", 12, "end"))


def _horizontal_axis(parts: list[str], left: float, right: float, top: float, bottom: float, maximum: float) -> None:
    for fraction in range(6):
        value = maximum * fraction / 5
        x = left + (right - left) * fraction / 5
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
            f'stroke="{BORDER}" stroke-width="1"/>'
        )
        parts.append(_text(x, bottom + 20, f"{value:.3g}", 12, "middle"))


def _metric_value(run, judge: str, metric: str) -> float | None:
    stats = run.feedback_stats.get(f"{judge}_weather_{metric}")
    return stats["avg"] if stats else None


def _asset_dir(parent: Path, experiment_id: str) -> Path:
    return parent / experiment_id


def _cache_path(parent: Path, experiment: str, judges: list[str] | None, metrics: list[str] | None) -> Path:
    key = json.dumps([experiment, judges, metrics], separators=(",", ":"))
    return parent / ".cache" / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _read_cache(path: Path) -> tuple[dict, int, str, dict]:
    cached = json.loads(path.read_text(encoding="utf-8"))
    return cached["data"], cached["repetitions"], cached["experiment_id"], cached["trace_metrics"]


def _write_cache(path: Path, data: dict, repetitions: int, experiment_id: str, trace_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "data": data,
                "repetitions": repetitions,
                "experiment_id": experiment_id,
                "trace_metrics": trace_metrics,
            }
        ),
        encoding="utf-8",
    )


def _data(
    experiment: str,
    judge_keys: list[str] | None = None,
    metric_keys: list[str] | None = None,
) -> tuple[dict, int, str]:
    client = Client()
    experiment_id = project_id(client, experiment)
    rows = list(
        client.get_experiment_results(
            project_id=experiment_id, preview=True
        )["examples_with_runs"]
    )
    runs_by_case, count = select_repetitions(
        [sorted(row.runs, key=lambda run: run.start_time or "") for row in rows],
        None,
    )
    judges = [(key, JUDGE_LABELS[key]) for key in (judge_keys or JUDGE_LABELS)]
    metrics = [
        metric
        for metric in (metric_keys or NUMERIC_METRICS)
        if all(
            _metric_value(run, judge, metric) is not None
            for runs in runs_by_case
            for run in runs
            for judge, _ in judges
        )
    ]
    if not metrics:
        raise ValueError("Experiment has no complete numeric judge metrics.")
    return {
        "judges": judges,
        "metrics": metrics,
        "values": {
            metric: {
                label: [
                    [_metric_value(run, judge, metric) for run in runs]
                    for runs in runs_by_case
                ]
                for judge, label in judges
            }
            for metric in metrics
        },
    }, count, experiment_id


def _variance(values: list[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def _continuous_metrics(data: dict) -> list[str]:
    return [metric for metric in data["metrics"] if metric != "does_pass"]


def _legend(parts: list[str], judges: list[tuple[str, str]], y: float, width: float) -> None:
    for index, (_, label) in enumerate(judges):
        x = 40 + index * (width - 80) / len(judges)
        parts.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{COLORS[index]}"/>')
        parts.append(_text(x + 22, y, label, 12))


def _guide(parts: list[str], width: float, height: float, reading: str, meaning: str) -> None:
    parts.append(_text(width / 2, height - 34, f"How to read: {reading}", 11, "middle", MUTED_TEXT))
    parts.append(_text(width / 2, height - 16, f"What it means: {meaning}", 11, "middle", MUTED_TEXT))


def _relative_variance(data: dict) -> str:
    metrics = _continuous_metrics(data)
    width, height = 1050, 235 + 150 * len(metrics)
    left, right, top, bottom = 190, width - 45, 75, height - 130
    ratios = {
        metric: {
            label: (
                statistics.fmean(_variance(case) for case in data["values"][metric][label])
                / statistics.fmean(_variance(case) for case in data["values"][metric]["Jev"])
                if statistics.fmean(_variance(case) for case in data["values"][metric]["Jev"])
                else 0 if label == "Jev" else None
            )
            for _, label in data["judges"]
        }
        for metric in metrics
    }
    maximum = max((value for values in ratios.values() for value in values.values() if value is not None), default=1)
    maximum = max(1, maximum)
    parts = [
        _text(width / 2, 32, "Relative mean per-case variance", 22, "middle"),
        _text(width / 2, 56, "Jev = 1× baseline; lower is more consistent", 14, "middle"),
    ]
    _axis(parts, left, right, top, bottom, maximum)
    bar_width = (right - left) / (len(metrics) * (len(data["judges"]) + 1))
    for metric_index, metric in enumerate(metrics):
        center = left + (metric_index + 0.5) * (right - left) / len(metrics)
        parts.append(_text(center, bottom + 25, metric.replace("_", " ").title(), 13, "middle"))
        for judge_index, (_, label) in enumerate(data["judges"]):
            value = ratios[metric][label]
            x = center + (judge_index - (len(data["judges"]) - 1) / 2) * bar_width
            if value is None:
                parts.append(_text(x, bottom - 16, "n/a", 11, "middle"))
                continue
            bar_height = max(2, (bottom - top) * value / maximum)
            parts.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{bottom - bar_height:.1f}" '
                f'width="{bar_width - 3:.1f}" height="{bar_height:.1f}" fill="{COLORS[judge_index]}"/>'
            )
    _legend(parts, data["judges"], height - 70, width)
    _guide(
        parts, width, height,
        "compare each bar with the 1× Jev baseline; lower bars are steadier.",
        "bars above 1× are more variable than Jev; isolated high bars point to a metric-specific weakness.",
    )
    return _svg(width, height, parts)


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _metric_oscillation(data: dict, repetitions: int, metric: str) -> str:
    series = {
        label: [
            statistics.fmean(case[repetition] for case in data["values"][metric][label])
            for repetition in range(repetitions)
        ]
        for _, label in data["judges"]
    }
    low = min(value for values in series.values() for value in values)
    high = max(value for values in series.values() for value in values)
    padding = max((high - low) * 0.1, 0.02)
    minimum, maximum = max(0, low - padding), min(1, high + padding)
    if minimum == maximum:
        minimum, maximum = max(0, minimum - 0.02), min(1, maximum + 0.02)

    width, panel_height = 1000, 130
    height = 125 + panel_height * len(data["judges"])
    left, right = 260, width - 40
    metric_label = metric.replace("_", " ")
    title = (
        f"Evaluator-returned quality score oscillation across {repetitions} repetitions"
        if metric == "quality"
        else f"Judge {metric_label} oscillation across {repetitions} repetitions"
    )
    subtitle = (
        "Mean evaluator-returned quality score across five frozen cases; shared zoomed y-axis"
        if metric == "quality"
        else f"Mean {metric_label} across five frozen cases; shared zoomed y-axis"
    )
    parts = [
        _text(width / 2, 32, title, 22, "middle"),
        _text(width / 2, 56, subtitle, 14, "middle"),
    ]
    for judge_index, (_, label) in enumerate(data["judges"]):
        top = 82 + judge_index * panel_height
        bottom = top + 82
        for fraction in range(3):
            value = minimum + (maximum - minimum) * fraction / 2
            y = bottom - (bottom - top) * fraction / 2
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1"/>')
            parts.append(_text(left - 10, y + 4, f"{value:.2f}", 11, "end"))
        points = " ".join(
            f"{left + index * (right - left) / max(repetitions - 1, 1):.1f},"
            f"{bottom - (value - minimum) * (bottom - top) / (maximum - minimum):.1f}"
            for index, value in enumerate(series[label])
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[judge_index]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.append(_text(left - 50, (top + bottom) / 2 + 5, label, 12, "end"))
        if judge_index == len(data["judges"]) - 1:
            for fraction in range(6):
                index = round(fraction * (repetitions - 1) / 5)
                x = left + index * (right - left) / max(repetitions - 1, 1)
                parts.append(_text(x, bottom + 22, str(index + 1), 12, "middle"))
    _guide(
        parts, width, height,
        "each point is the mean across frozen cases for one repetition; panels share one zoomed scale.",
        "a flat line is repeatable; frequent swings are noisy; isolated spikes or dips are one-off judgement changes.",
    )
    return _svg(width, height, parts)


def _cost_and_latency(metrics: dict) -> str:
    values = [
        (label, metrics["judges"][label])
        for _, label in JUDGE_LABELS.items()
        if label in metrics["judges"]
        and metrics["judges"][label]["average_cost"] is not None
        and metrics["judges"][label]["average_latency_seconds"] is not None
    ]
    costs = [item["average_cost"] for _, item in values]
    latencies = [item["average_latency_seconds"] for _, item in values]
    log_min, log_max = math.log10(min(costs)), math.log10(max(costs))
    if log_min == log_max:
        log_min -= 0.5
        log_max += 0.5
    maximum_latency = max(latencies) * 1.1
    width, height = 1050, 625
    left, right, top, bottom = 125, width - 45, 95, 450
    parts = [
        _text(width / 2, 32, "Judge cost and latency", 22, "middle"),
        _text(width / 2, 56, "Cost per evaluator call (log scale); bubble size shows average total tokens", 14, "middle"),
    ]
    for fraction in range(5):
        exponent = log_min + (log_max - log_min) * fraction / 4
        x = left + (right - left) * fraction / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{BORDER}" stroke-width="1"/>')
        parts.append(_text(x, bottom + 24, f"${10 ** exponent:.4g}", 12, "middle"))
    for fraction in range(5):
        value = maximum_latency * fraction / 4
        y = bottom - (bottom - top) * fraction / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1"/>')
        parts.append(_text(left - 10, y + 4, f"{value:.1f}s", 12, "end"))
    token_counts = [item["average_input_tokens"] + item["average_output_tokens"] for _, item in values]
    low_tokens, high_tokens = min(token_counts), max(token_counts)
    for index, (label, item) in enumerate(values):
        x = left + (right - left) * (math.log10(item["average_cost"]) - log_min) / (log_max - log_min)
        y = bottom - (bottom - top) * item["average_latency_seconds"] / maximum_latency
        tokens = item["average_input_tokens"] + item["average_output_tokens"]
        radius = 10 + 10 * (tokens - low_tokens) / max(high_tokens - low_tokens, 1)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{COLORS[index]}" fill-opacity="0.8">'
            f'<title>{html.escape(label)}: ${item["average_cost"]:.5f}/call, {item["average_latency_seconds"]:.2f}s, {tokens:.0f} tokens/call</title></circle>'
        )
    _legend(parts, [(label, label) for label, _ in values], 530, width)
    _guide(
        parts, width, height,
        "read cost left to right, latency bottom to top, and bubble size as total tokens per call.",
        "smaller lower-left bubbles are cheaper and faster; upper-right or larger bubbles consume more time or budget.",
    )
    return _svg(width, height, parts)


def _metric_by_repetition(data: dict, repetitions: int, metric: str) -> str:
    width, panel_height = 1000, 90 + 52 * len(data["judges"])
    height = 70 + panel_height + 65
    left, right = 220, width - 40
    parts = [
        _text(width / 2, 32, "Mean quality by repetition", 22, "middle"),
        _text(width / 2, 56, "Boxes show the middle 50%; whiskers show the full observed range.", 14, "middle"),
    ]
    top = 82
    bottom = top + 52 * len(data["judges"])
    values = {
        label: [statistics.fmean(case[repetition] for case in data["values"][metric][label]) for repetition in range(repetitions)]
        for _, label in data["judges"]
    }
    for fraction in range(6):
        x = left + (right - left) * fraction / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{BORDER}" stroke-width="1"/>')
        parts.append(_text(x, bottom + 20, f"{fraction / 5:.1f}", 12, "middle"))
    for judge_index, (_, label) in enumerate(data["judges"]):
        y = top + 24 + judge_index * 52
        scale = lambda value: left + (right - left) * value
        parts.append(_text(left - 12, y + 5, label, 12, "end"))
        low, q1, median, q3, high = (_percentile(values[label], probability) for probability in (0, 0.25, 0.5, 0.75, 1))
        parts.append(f'<line x1="{scale(low):.1f}" y1="{y}" x2="{scale(high):.1f}" y2="{y}" stroke="{MUTED_TEXT}" stroke-width="2"/>')
        parts.append(f'<rect x="{scale(q1):.1f}" y="{y - 12}" width="{max(scale(q3) - scale(q1), 1):.1f}" height="24" fill="{COLORS[judge_index]}" fill-opacity="0.75"/>')
        parts.append(f'<line x1="{scale(median):.1f}" y1="{y - 14}" x2="{scale(median):.1f}" y2="{y + 14}" stroke="{TEXT}" stroke-width="2"/>')
    _guide(
        parts, width, height,
        "boxes cover the middle half of repetition means; whiskers show the full range.",
        "a narrow box and short whiskers are steady; a wide range signals drift or occasional outlier judgments.",
    )
    return _svg(width, height, parts)


def _does_pass_variance(data: dict) -> str:
    cases = len(data["values"]["does_pass"]["Jev"])
    width, left, right, top = 1000, 190, 960, 105
    row_height = 76
    height = top + row_height * len(data["judges"]) + 105
    parts = [
        _text(width / 2, 32, "Binary pass/fail variance by frozen case", 22, "middle"),
        _text(width / 2, 56, "Each dot is one case; 0 is stable and 0.25 is a 50/50 pass/fail split", 14, "middle"),
    ]
    bottom = top + row_height * len(data["judges"])
    for tick in range(6):
        x = left + (right - left) * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{BORDER}" stroke-width="1"/>')
        parts.append(_text(x, bottom + 20, f"{tick / 20:.2f}", 12, "middle"))
    for judge_index, (_, label) in enumerate(data["judges"]):
        row_top = top + judge_index * row_height
        parts.append(_text(left - 12, row_top + row_height / 2 + 5, label, 12, "end"))
        for case, values in enumerate(data["values"]["does_pass"][label]):
            pass_rate = statistics.fmean(values)
            variance = pass_rate * (1 - pass_rate)
            x = left + (right - left) * variance / 0.25
            y = row_top + 14 + case * 12
            color = COLORS[case % len(COLORS)]
            parts.append(f'<line x1="{left}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="{color}"/>')
            parts.append(_text(x + 8 if x < right - 60 else x - 8, y + 4, f"C{case + 1} {pass_rate:.0%}", 10, "start" if x < right - 60 else "end"))
    _guide(
        parts, width, height,
        "read each dot's position as variance and its label as pass rate.",
        "dots near 0 are reliable outcomes; dots near 0.25 are unstable thresholds that need review before automation.",
    )
    return _svg(width, height, parts)


def _does_pass_accuracy(accuracy: dict) -> str:
    width, height = 1000, 150 + 58 * len(accuracy["judges"])
    left, right, top, bottom = 270, width - 45, 105, height - 75
    parts = [
        _text(width / 2, 32, "Human-oracle pass/fail accuracy", 22, "middle"),
        _text(width / 2, 56, "All repeated judge decisions compared with the fixed human label", 14, "middle"),
    ]
    _horizontal_axis(parts, left, right, top, bottom, 1)
    for index, (label, result) in enumerate(accuracy["judges"].items()):
        y = top + 25 + index * 58
        value = result["does_pass_accuracy"]
        x = left + (right - left) * value
        parts.append(_text(left - 12, y + 5, label, 12, "end"))
        parts.append(f'<rect x="{left}" y="{y - 12}" width="{x - left:.1f}" height="24" fill="{COLORS[index]}"/>')
        parts.append(_text(min(x + 8, right), y + 5, f"{value:.1%}", 12, "start" if x < right - 45 else "end"))
    _guide(
        parts, width, height,
        "bar length is the share of all repeated pass/fail judgments that match the oracle.",
        "longer bars mean the judge more often reaches the human pass/fail decision.",
    )
    return _svg(width, height, parts)


def _quality_agreement(accuracy: dict) -> str:
    width, height = 1100, 150 + 58 * len(accuracy["judges"])
    left, middle, right, top, bottom = 260, 610, width - 45, 105, height - 75
    maximum_mae = max(result["quality_mae"] for result in accuracy["judges"].values())
    maximum_mae = max(math.ceil(maximum_mae * 10) / 10, 0.1)
    parts = [
        _text(width / 2, 32, "Human-oracle quality agreement", 22, "middle"),
        _text(
            width / 2,
            56,
            f"MAE is lower; agreement is within ±{accuracy['quality_tolerance']:.2f} of the oracle quality score",
            14,
            "middle",
        ),
    ]
    _horizontal_axis(parts, left, middle - 35, top, bottom, maximum_mae)
    _horizontal_axis(parts, middle + 65, right, top, bottom, 1)
    for index, (label, result) in enumerate(accuracy["judges"].items()):
        y = top + 25 + index * 58
        mae_right = left + (middle - 35 - left) * result["quality_mae"] / maximum_mae
        agreement_left = middle + 65
        agreement_right = agreement_left + (right - agreement_left) * result["quality_within_tolerance"]
        parts.append(_text(left - 12, y + 5, label, 12, "end"))
        parts.append(f'<rect x="{left}" y="{y - 12}" width="{mae_right - left:.1f}" height="24" fill="{COLORS[index]}"/>')
        parts.append(_text(mae_right + 8, y + 5, f"{result['quality_mae']:.3f}", 12))
        parts.append(f'<rect x="{agreement_left}" y="{y - 12}" width="{agreement_right - agreement_left:.1f}" height="24" fill="{COLORS[index]}"/>')
        parts.append(_text(min(agreement_right + 8, right), y + 5, f"{result['quality_within_tolerance']:.1%}", 12))
    parts.append(_text((left + middle - 35) / 2, 87, "Mean absolute error", 13, "middle"))
    parts.append(_text((middle + 65 + right) / 2, 87, "Within tolerance", 13, "middle"))
    _guide(
        parts, width, height,
        "compare MAE on the left and the within-tolerance share on the right.",
        "shorter left bars and longer right bars indicate closer agreement with the human oracle.",
    )
    return _svg(width, height, parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate judge experiment visualizations.")
    parser.add_argument("experiment", help="Experiment name, project UUID, or comparison URL")
    parser.add_argument("--judges", nargs="+", choices=JUDGE_LABELS, help="Only include these judges")
    parser.add_argument("--metrics", nargs="+", choices=NUMERIC_METRICS, help="Only include these feedback metrics")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"), help="Parent directory for experiment assets")
    parser.add_argument("--cache", type=Path, help="Read a previously generated visualization cache")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached experiment data from LangSmith")
    parser.add_argument("--benchmark", type=Path, help="Archived benchmark JSON for oracle scoring")
    parser.add_argument("--oracle-labels", type=Path, help="Human oracle labels JSON")
    parser.add_argument("--quality-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    if bool(args.benchmark) != bool(args.oracle_labels):
        parser.error("--benchmark and --oracle-labels must be supplied together.")
    if args.quality_tolerance < 0:
        parser.error("--quality-tolerance must be non-negative.")
    cache_path = args.cache or _cache_path(Path("assets"), args.experiment, args.judges, args.metrics)
    if cache_path.exists() and not args.refresh:
        data, repetitions, experiment_id, trace_metrics = _read_cache(cache_path)
    else:
        load_dotenv(override=True)
        data, repetitions, experiment_id = _data(args.experiment, args.judges, args.metrics)
        trace_metrics = query_metrics(args.experiment, args.judges, args.metrics)
        _write_cache(cache_path, data, repetitions, experiment_id, trace_metrics)
    output_dir = _asset_dir(args.output_dir, experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = None
    if args.benchmark:
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
        if benchmark["experiment_id"] != experiment_id:
            parser.error("Benchmark and experiment IDs do not match.")
        oracle = json.loads(args.oracle_labels.read_text(encoding="utf-8"))
        accuracy = score_oracle(
            data,
            benchmark["frozen_cases"],
            oracle,
            args.quality_tolerance,
        )
        accuracy["experiment_id"] = experiment_id
        accuracy_output = output_dir / "accuracy.json"
        accuracy_output.write_text(json.dumps(accuracy, indent=2) + "\n", encoding="utf-8")
        print(accuracy_output)
    if _continuous_metrics(data):
        outputs = {"variance-ratios.svg": _relative_variance(data)}
    else:
        outputs = {}
    if "quality" in data["metrics"]:
        outputs["quality-by-repetition.svg"] = _metric_by_repetition(data, repetitions, "quality")
        outputs["quality-oscillation.svg"] = _metric_oscillation(data, repetitions, "quality")
    if trace_metrics["judges"]:
        outputs["cost-and-latency.svg"] = _cost_and_latency(trace_metrics)
    if "does_pass" in data["metrics"]:
        outputs["does-pass-oscillation.svg"] = _metric_oscillation(data, repetitions, "does_pass")
        outputs["does-pass-variance.svg"] = _does_pass_variance(data)
    if accuracy:
        outputs["does-pass-accuracy.svg"] = _does_pass_accuracy(accuracy)
        outputs["quality-agreement.svg"] = _quality_agreement(accuracy)
    for name, contents in outputs.items():
        output = output_dir / name
        output.write_text(contents, encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
