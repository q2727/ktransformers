#!/usr/bin/env python3
"""Summarize an SSD SM-cap/fanout sweep and check output equivalence."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


LABEL_RE = re.compile(r"target(?P<target>\d+)_draft(?P<draft>\d+)_f(?P<fanout>\d+)")


def load_point(path: Path) -> dict:
    analysis = json.loads(path.read_text())
    label = analysis["label"]
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Cannot parse resource configuration from {label!r}")

    config_dir = path.parent
    summary_path = config_dir / "records.jsonl.summary.json"
    summary = json.loads(summary_path.read_text())
    overall = analysis["overall"]
    build_ms = overall["outcome_cache"]["median_ms"]
    verify_ms = overall["target_verify_ms_per_round"]
    rows = analysis["rows"]
    fan_outs = rows[0].get("fan_outs", []) if rows else []
    return {
        "label": label,
        "target_mps_pct": int(match.group("target")),
        "draft_mps_pct": int(match.group("draft")),
        "fanout": int(match.group("fanout")),
        "branches": sum(fan_outs) or int(match.group("fanout")) * 6,
        "requests": overall["requests"],
        "rounds": overall["rounds"],
        "throughput_tok_s": summary["throughput_tok_s"],
        "latency_mean_s": overall["client_latency_mean_s"],
        "cache_miss_pct": overall["cache_miss_rate"] * 100,
        "accepted_draft_per_round": overall["accepted_draft_per_round"],
        "verify_ms_per_round": verify_ms,
        "draft_wait_ms_per_round": overall["draft_wait_ms_per_round"],
        "critical_ms_per_round": verify_ms + overall["draft_wait_ms_per_round"],
        "build_p50_ms": build_ms,
        "build_slack_p50_ms": verify_ms - build_ms,
        "select_hit_p50_ms": overall.get("select_hit", {}).get("median_ms"),
        "select_miss_jit_p50_ms": overall.get("select_miss_jit", {}).get(
            "median_ms"
        ),
        "records": rows,
    }


def output_mismatches(points: list[dict]) -> dict[str, int]:
    reference: dict[tuple[str, int], str] = {}
    mismatches: dict[str, int] = {}
    for point in points:
        count = 0
        for row in point.pop("records"):
            key = (row["dataset"], row["dataset_index"])
            digest = row["token_sha256"]
            if key not in reference:
                reference[key] = digest
            elif reference[key] != digest:
                count += 1
        mismatches[point["label"]] = count
    return mismatches


def miss_power_law(points: list[dict]) -> dict:
    by_fanout: dict[int, list[float]] = defaultdict(list)
    for point in points:
        by_fanout[point["fanout"]].append(point["cache_miss_pct"] / 100)
    samples = [
        (math.log(fanout), math.log(sum(values) / len(values)))
        for fanout, values in sorted(by_fanout.items())
        if fanout > 0 and all(value > 0 for value in values)
    ]
    if len(samples) < 2:
        return {"exponent": None, "fanout_mean_miss_rate": {}}
    x_mean = sum(x for x, _ in samples) / len(samples)
    y_mean = sum(y for _, y in samples) / len(samples)
    denominator = sum((x - x_mean) ** 2 for x, _ in samples)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in samples) / denominator
    return {
        "exponent": slope,
        "fanout_mean_miss_rate": {
            str(fanout): sum(values) / len(values)
            for fanout, values in sorted(by_fanout.items())
        },
    }


def render_table(points: list[dict]) -> str:
    header = (
        "label\tthroughput\tmiss_pct\tverify_ms\tbuild_p50_ms\t"
        "slack_p50_ms\tjit_p50_ms\twait_ms\tmismatches"
    )
    rows = [header]
    for point in points:
        rows.append(
            "\t".join(
                (
                    point["label"],
                    f'{point["throughput_tok_s"]:.3f}',
                    f'{point["cache_miss_pct"]:.2f}',
                    f'{point["verify_ms_per_round"]:.3f}',
                    f'{point["build_p50_ms"]:.3f}',
                    f'{point["build_slack_p50_ms"]:.3f}',
                    (
                        f'{point["select_miss_jit_p50_ms"]:.3f}'
                        if point["select_miss_jit_p50_ms"] is not None
                        else "NA"
                    ),
                    f'{point["draft_wait_ms_per_round"]:.3f}',
                    str(point["output_mismatches"]),
                )
            )
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    analysis_paths = sorted(
        path
        for root in args.root
        for path in root.glob("*/analysis.json")
        if (path.parent / "complete").exists()
    )
    if not analysis_paths:
        raise RuntimeError("No completed analysis.json files found.")
    points = [load_point(path) for path in analysis_paths]
    points.sort(key=lambda row: (row["target_mps_pct"], row["draft_mps_pct"], row["fanout"]))
    mismatches = output_mismatches(points)
    for point in points:
        point["output_mismatches"] = mismatches[point["label"]]

    result = {
        "points": points,
        "best_throughput": max(points, key=lambda row: row["throughput_tok_s"]),
        "miss_power_law": miss_power_law(points),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(render_table(points))
    print(
        f'best={result["best_throughput"]["label"]} '
        f'throughput={result["best_throughput"]["throughput_tok_s"]:.3f} tok/s '
        f'miss_exponent={result["miss_power_law"]["exponent"]:.3f}'
    )


if __name__ == "__main__":
    main()
