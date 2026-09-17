import statistics
from pathlib import Path

from dotenv import load_dotenv


EXPERIMENT_ID = "ff3d8aa4-28c4-482e-aa54-1ff1f2b604ba"
COLORS = {"Jev": "#2563eb", "GPT-5.6 Luna": "#f97316"}
WIDTH, HEIGHT = 800, 440
LEFT, RIGHT, TOP, BOTTOM = 70, 40, 75, 80
MIN_SCORE, MAX_SCORE = 0.55, 0.95


def _x(repetition: int, count: int) -> float:
    return LEFT + repetition * (WIDTH - LEFT - RIGHT) / (count - 1)


def _y(score: float) -> float:
    return TOP + (MAX_SCORE - score) * (HEIGHT - TOP - BOTTOM) / (
        MAX_SCORE - MIN_SCORE
    )


def _text(x: float, y: float, value: str, size: int = 16, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="#1f2937">{value}</text>'
    )


def _score_series() -> dict[str, list[float]]:
    load_dotenv(override=True)
    from langsmith import Client

    rows = list(
        Client().get_experiment_results(
            project_id=EXPERIMENT_ID,
            preview=True,
        )["examples_with_runs"]
    )
    runs_by_case = [sorted(row.runs, key=lambda run: run.start_time or "") for row in rows]
    repetitions = min(len(runs) for runs in runs_by_case)
    return {
        judge: [
            statistics.mean(
                runs_by_case[case][repetition]
                .feedback_stats[f"{prefix}_weather_score"]["avg"]
                for case in range(len(runs_by_case))
            )
            for repetition in range(repetitions)
        ]
        for judge, prefix in (("Jev", "jev"), ("GPT-5.6 Luna", "llm"))
    }


def main() -> None:
    series = _score_series()
    count = len(next(iter(series.values())))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(WIDTH / 2, 32, "Judge scores across 100 repetitions", 22, "middle"),
        _text(
            WIDTH / 2,
            57,
            "Mean score across five frozen cases; zoomed y-axis (score range is 0–1)",
            14,
            "middle",
        ),
    ]

    for tick in (0.6, 0.7, 0.8, 0.9):
        y = _y(tick)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" '
            'stroke="#d1d5db" stroke-width="1"/>'
        )
        parts.append(_text(LEFT - 10, y + 5, f"{tick:.1f}", 13, "end"))

    for repetition in (0, 24, 49, 74, 99):
        x = _x(repetition, count)
        parts.append(
            f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM}" '
            'stroke="#eef0f2" stroke-width="1"/>'
        )
        parts.append(_text(x, HEIGHT - BOTTOM + 28, str(repetition + 1), 13, "middle"))

    for judge, scores in series.items():
        points = " ".join(
            f"{_x(index, count):.1f},{_y(score):.1f}"
            for index, score in enumerate(scores)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[judge]}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    legend_y = HEIGHT - 25
    for index, judge in enumerate(series):
        x = LEFT + index * 170
        parts.append(
            f'<line x1="{x}" y1="{legend_y - 7}" x2="{x + 18}" y2="{legend_y - 7}" '
            f'stroke="{COLORS[judge]}" stroke-width="3"/>'
        )
        parts.append(_text(x + 26, legend_y - 2, judge, 13))

    parts.append("</svg>")
    output = Path(__file__).resolve().parents[2] / "score_repetitions.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
