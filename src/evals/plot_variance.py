import math
from pathlib import Path


VARIANCES = {
    "Quality": {"Jev": 0.0000147138, "GPT-5.6 Luna": 0.00328971},
    "Score": {"Jev": 0.0000720667, "GPT-5.6 Luna": 0.0201111},
}
COLORS = {"Jev": "#2563eb", "GPT-5.6 Luna": "#f97316"}
WIDTH, HEIGHT = 800, 420
LEFT, RIGHT, TOP, BOTTOM = 110, 40, 75, 80
MIN_RATIO, MAX_RATIO = 0.5, 400


def _x(value: float) -> float:
    scale = (WIDTH - LEFT - RIGHT) / (
        math.log10(MAX_RATIO) - math.log10(MIN_RATIO)
    )
    return LEFT + (math.log10(value) - math.log10(MIN_RATIO)) * scale


def _text(x: float, y: float, value: str, size: int = 16, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="#1f2937">{value}</text>'
    )


def main() -> None:
    ratios = {
        metric: {
            judge: value / values["Jev"]
            for judge, value in values.items()
        }
        for metric, values in VARIANCES.items()
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(WIDTH / 2, 32, "Relative variance of repeated judge ratings", 22, "middle"),
        _text(WIDTH / 2, 57, "Jev = 1× baseline; lower is more consistent", 14, "middle"),
    ]

    for tick in (1, 10, 100, 300):
        x = _x(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM}" '
            'stroke="#d1d5db" stroke-width="1"/>'
        )
        parts.append(_text(x, HEIGHT - BOTTOM + 28, f"{tick}×", 13, "middle"))

    bar_height = 28
    for row, (metric, values) in enumerate(ratios.items()):
        center_y = TOP + 65 + row * 125
        parts.append(_text(LEFT - 18, center_y + 5, metric, 16, "end"))
        for offset, judge in ((-20, "Jev"), (20, "GPT-5.6 Luna")):
            value = values[judge]
            x_start = _x(MIN_RATIO)
            x_end = _x(value)
            y = center_y + offset - bar_height / 2
            parts.append(
                f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{x_end - x_start:.1f}" '
                f'height="{bar_height}" rx="4" fill="{COLORS[judge]}"/>'
            )
            label_x = min(x_end + 8, WIDTH - RIGHT)
            anchor = "end" if label_x == WIDTH - RIGHT else "start"
            parts.append(_text(label_x, y + 20, f"{value:.0f}×", 14, anchor))

    legend_y = HEIGHT - 25
    for index, judge in enumerate(("Jev", "GPT-5.6 Luna")):
        x = LEFT + index * 170
        parts.append(
            f'<rect x="{x}" y="{legend_y - 13}" width="14" height="14" '
            f'fill="{COLORS[judge]}"/>'
        )
        parts.append(_text(x + 22, legend_y, judge, 13))

    parts.append("</svg>")
    output = Path(__file__).resolve().parents[2] / "variance_comparison.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
