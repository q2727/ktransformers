#!/usr/bin/env python3
"""Plot benchmark_micro.py summary JSON as dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


WIDTH = 960
HEIGHT = 600
LEFT = 92
RIGHT = 36
TOP = 64
BOTTOM = 78


def svg_text(x: float, y: float, text: str, **attrs: Any) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{html.escape(text)}</text>'


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return step * magnitude


def plot(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    benchmark = summary["benchmark"]
    points = summary["points"]
    if benchmark == "prefill":
        x_values = [float(point["actual_prompt_tokens_median"]) for point in points]
        median_values = [float(point["ttft_median_seconds"]) for point in points]
        p95_values = [float(point["ttft_p95_seconds"]) for point in points]
        x_labels = [
            f"{int(point['nominal_prompt_tokens']) // 1024}K" for point in points
        ]
        title = f"{summary['model']} — Prefill TTFT"
        x_title = "Prompt length (tokens, log2 spacing)"
        y_title = "TTFT (seconds)"
    else:
        x_values = [float(point["batch_size"]) for point in points]
        median_values = [1000 * float(point["tpot_median_seconds"]) for point in points]
        p95_values = (
            [1000 * float(point["tpot_p95_seconds"]) for point in points]
            if all("tpot_p95_seconds" in point for point in points)
            else []
        )
        x_labels = [str(int(point["batch_size"])) for point in points]
        title = f"{summary['model']} — Decode TPOT at 8K context"
        x_title = "Synchronized batch size"
        y_title = "TPOT (milliseconds/token)"

    if not points:
        raise ValueError("Summary contains no points")
    y_max = nice_upper(max(p95_values or median_values) * 1.08)
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM

    def x_position(index: int) -> float:
        if len(points) == 1:
            return LEFT + plot_width / 2
        return LEFT + index * plot_width / (len(points) - 1)

    def y_position(value: float) -> float:
        return TOP + plot_height * (1 - value / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(WIDTH / 2, 34, title, text_anchor="middle", font_family="sans-serif", font_size="22", font_weight="600", fill="#172033"),
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="#dce2ea" stroke-width="1"/>')
        label = f"{value:.1f}" if y_max < 10 else f"{value:.0f}"
        parts.append(svg_text(LEFT - 12, y + 5, label, text_anchor="end", font_family="sans-serif", font_size="13", fill="#4c566a"))

    for index, label in enumerate(x_labels):
        x = x_position(index)
        parts.append(f'<line x1="{x:.1f}" y1="{TOP + plot_height}" x2="{x:.1f}" y2="{TOP + plot_height + 6}" stroke="#273246"/>')
        parts.append(svg_text(x, TOP + plot_height + 25, label, text_anchor="middle", font_family="sans-serif", font_size="13", fill="#4c566a"))

    parts.extend(
        [
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_height}" stroke="#273246" stroke-width="1.5"/>',
            f'<line x1="{LEFT}" y1="{TOP + plot_height}" x2="{WIDTH - RIGHT}" y2="{TOP + plot_height}" stroke="#273246" stroke-width="1.5"/>',
            svg_text(LEFT + plot_width / 2, HEIGHT - 22, x_title, text_anchor="middle", font_family="sans-serif", font_size="15", fill="#273246"),
            f'<text x="22" y="{TOP + plot_height / 2:.1f}" transform="rotate(-90 22 {TOP + plot_height / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#273246">{html.escape(y_title)}</text>',
        ]
    )

    colors = [("Median", median_values, "#146c94")]
    if p95_values:
        colors.append(("P95", p95_values, "#d1495b"))
    for name, values, color in colors:
        coordinates = " ".join(
            f"{x_position(index):.1f},{y_position(value):.1f}"
            for index, value in enumerate(values)
        )
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for index, value in enumerate(values):
            x, y = x_position(index), y_position(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            parts.append(svg_text(x, y - 10, f"{value:.1f}", text_anchor="middle", font_family="sans-serif", font_size="11", fill=color))

    legend_x = WIDTH - RIGHT - 170
    for offset, (name, _values, color) in enumerate(colors):
        x = legend_x + offset * 88
        parts.append(f'<line x1="{x}" y1="50" x2="{x + 22}" y2="50" stroke="{color}" stroke-width="3"/>')
        parts.append(svg_text(x + 28, 55, name, font_family="sans-serif", font_size="13", fill="#273246"))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.summary.with_suffix(".svg")
    plot(args.summary, output)


if __name__ == "__main__":
    main()
