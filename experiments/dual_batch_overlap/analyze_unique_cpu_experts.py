#!/usr/bin/env python3
"""Aggregate repeated KT unique-active-expert scaling profiles."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPERT_WEIGHT_BYTES = 18_874_368


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "experiments/artifacts/dual_batch_overlap/cpu_kernel_unique_experts"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def aggregate(root: Path, pattern: str, expected_top_k: int) -> dict:
    paths = sorted(root.glob(pattern))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three {pattern} runs, found {len(paths)}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, run in zip(paths, runs, strict=True):
        config = run["configuration"]
        if config["top_k_override"] != expected_top_k:
            raise RuntimeError(f"Unexpected top-k in {path}")
        if config["expert_weight_bytes_node_total"] != EXPERT_WEIGHT_BYTES:
            raise RuntimeError(f"Unexpected expert size in {path}")

    rows = []
    for group in zip(*(run["rows"] for run in runs), strict=True):
        expert_counts = {row["unique_cpu_experts"] for row in group}
        if len(expert_counts) != 1:
            raise RuntimeError("Expert counts differ across repeated runs")
        count = next(iter(expert_counts))
        medians = [row["timing"]["median_ms"] for row in group]
        bandwidths = [
            row["effective_weight_bandwidth_gb_per_second_median"]
            for row in group
        ]
        rows.append(
            {
                "unique_cpu_experts": count,
                "tokens": group[0]["tokens"],
                "median_time_ms_across_runs_mean": statistics.fmean(medians),
                "median_time_ms_across_runs_min": min(medians),
                "median_time_ms_across_runs_max": max(medians),
                "median_bandwidth_gb_per_second_across_runs_mean": (
                    statistics.fmean(bandwidths)
                ),
                "median_bandwidth_gb_per_second_across_runs_min": min(bandwidths),
                "median_bandwidth_gb_per_second_across_runs_max": max(bandwidths),
            }
        )

    high = [row for row in rows if row["unique_cpu_experts"] >= 64]
    x_mean = statistics.fmean(row["unique_cpu_experts"] for row in high)
    y_mean = statistics.fmean(
        row["median_time_ms_across_runs_mean"] for row in high
    )
    slope = sum(
        (row["unique_cpu_experts"] - x_mean)
        * (row["median_time_ms_across_runs_mean"] - y_mean)
        for row in high
    ) / sum((row["unique_cpu_experts"] - x_mean) ** 2 for row in high)
    intercept = y_mean - slope * x_mean
    max_bandwidth = max(
        row["median_bandwidth_gb_per_second_across_runs_mean"] for row in rows
    )

    def first_at_fraction(fraction: float) -> dict:
        return next(
            row
            for row in rows
            if row["median_bandwidth_gb_per_second_across_runs_mean"]
            >= fraction * max_bandwidth
        )

    return {
        "top_k": expected_top_k,
        "sources": [str(path) for path in paths],
        "rows": rows,
        "summary": {
            "maximum_observed_mean_median_bandwidth_gb_per_second": max_bandwidth,
            "first_count_at_90_percent_of_observed_max": first_at_fraction(0.90)[
                "unique_cpu_experts"
            ],
            "first_count_at_95_percent_of_observed_max": first_at_fraction(0.95)[
                "unique_cpu_experts"
            ],
            "high_count_linear_fit_min_experts": 64,
            "high_count_time_intercept_ms": intercept,
            "high_count_time_slope_ms_per_expert": slope,
            "high_count_asymptotic_bandwidth_gb_per_second": (
                EXPERT_WEIGHT_BYTES / (slope / 1000.0) / 1e9
            ),
            "experts_implied_by_2_513972_ms_linear_fit": (
                (2.5139719918370247 - intercept) / slope
            ),
        },
    }


def main() -> None:
    args = parse_args()
    top1 = aggregate(args.root, "run[123].json", 1)
    top8 = aggregate(args.root, "topk8_run[123].json", 8)
    top1_by_count = {row["unique_cpu_experts"]: row for row in top1["rows"]}
    route_shape_differences = []
    for row in top8["rows"]:
        count = row["unique_cpu_experts"]
        top1_row = top1_by_count[count]
        route_shape_differences.append(
            100.0
            * (
                row["median_time_ms_across_runs_mean"]
                / top1_row["median_time_ms_across_runs_mean"]
                - 1.0
            )
        )
    payload = {
        "schema_version": 1,
        "experiment": "kt_bf16_unique_cpu_expert_scaling_aggregate",
        "top1": top1,
        "top8_validation": top8,
        "cross_route_validation": {
            "definition": "100 * (top8_time / top1_time - 1) at equal unique-expert count",
            "mean_percent": statistics.fmean(route_shape_differences),
            "mean_absolute_percent": statistics.fmean(
                abs(value) for value in route_shape_differences
            ),
            "max_absolute_percent": max(
                abs(value) for value in route_shape_differences
            ),
        },
        "hardware_reference": {
            "dual_socket_epyc_9554_theoretical_gb_per_second": 921.6,
            "note": (
                "Two sockets * 460.8 GB/s AMD specification. Actual DIMM speed "
                "was not exposed by unprivileged dmidecode, so this is a CPU-limit reference."
            ),
        },
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
