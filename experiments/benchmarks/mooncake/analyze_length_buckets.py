#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Callable


INPUT_BUCKETS = (
    ("<4K", lambda value: value < 4096),
    ("4K-8K", lambda value: 4096 <= value < 8192),
    ("8K-16K", lambda value: 8192 <= value < 16384),
    ("16K-32K", lambda value: 16384 <= value < 32768),
    (">=32K", lambda value: value >= 32768),
)
OUTPUT_BUCKETS = (
    ("<64", lambda value: value < 64),
    ("64-256", lambda value: 64 <= value < 256),
    ("256-1024", lambda value: 256 <= value < 1024),
    (">=1024", lambda value: value >= 1024),
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metric(record: dict, name: str) -> float:
    return float(record["metrics"][name]["value"])


def summarize(records: list[dict]) -> dict:
    ttft = [metric(record, "time_to_first_token") for record in records]
    itl = [metric(record, "inter_token_latency") for record in records]
    e2e = [metric(record, "request_latency") for record in records]
    good = sum(
        ttft_value <= 20_000 and itl_value <= 250 and e2e_value <= 120_000
        for ttft_value, itl_value, e2e_value in zip(ttft, itl, e2e)
    )
    return {
        "count": len(records),
        "ttft_ms": {
            "mean": statistics.fmean(ttft),
            "p50": percentile(ttft, 0.50),
            "p95": percentile(ttft, 0.95),
        },
        "itl_ms": {
            "mean": statistics.fmean(itl),
            "p50": percentile(itl, 0.50),
            "p95": percentile(itl, 0.95),
        },
        "e2e_ms": {
            "mean": statistics.fmean(e2e),
            "p50": percentile(e2e, 0.50),
            "p95": percentile(e2e, 0.95),
        },
        "joint_slo_pass": good,
        "joint_slo_pct": good / len(records) * 100,
    }


def bucketize(
    records: list[dict],
    metric_name: str,
    buckets: tuple[tuple[str, Callable[[float], bool]], ...],
) -> dict:
    result = {}
    for label, predicate in buckets:
        selected = [
            record for record in records if predicate(metric(record, metric_name))
        ]
        if selected:
            result[label] = summarize(selected)
    return result


def rate_from_name(name: str) -> float:
    match = re.search(r"_r([0-9]+p[0-9]+)$", name)
    if not match:
        raise ValueError(f"Cannot parse request rate from {name}")
    return float(match.group(1).replace("p", "."))


def load_runs(
    results_root: Path, run_glob: str
) -> list[tuple[float, Path, list[dict]]]:
    runs = []
    for path in results_root.glob(run_glob):
        records_path = path / "profile_export.jsonl"
        if not path.is_dir() or not records_path.is_file():
            continue
        records = [json.loads(line) for line in records_path.open(encoding="utf-8")]
        if len(records) != 256:
            continue
        runs.append((rate_from_name(path.name), path, records))
    return sorted(runs)


def format_bucket_rows(rate: float, buckets: dict) -> list[str]:
    rows = []
    for label, values in buckets.items():
        rows.append(
            "| "
            + " | ".join(
                (
                    f"{rate:.3f}",
                    label,
                    str(values["count"]),
                    f'{values["ttft_ms"]["p50"] / 1000:.1f}',
                    f'{values["ttft_ms"]["p95"] / 1000:.1f}',
                    f'{values["itl_ms"]["p95"]:.1f}',
                    f'{values["e2e_ms"]["p95"] / 1000:.1f}',
                    f'{values["joint_slo_pct"]:.1f}%',
                )
            )
            + " |"
        )
    return rows


def render_markdown(report: dict) -> str:
    lines = [
        "# Mooncake latency by request length",
        "",
        "All tables use the fixed 256-request Mooncake ToolAgent window.",
        "The joint SLO is TTFT <= 20 s, ITL <= 250 ms, and E2E <= 120 s.",
    ]
    for bucket_key, title in (
        ("input_buckets", "Input-length buckets"),
        ("output_buckets", "Output-length buckets"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Target rate | Bucket | N | TTFT p50 s | TTFT p95 s | ITL p95 ms | E2E p95 s | Joint SLO |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for rate_key, values in report["rates"].items():
            lines.extend(format_bucket_rows(float(rate_key), values[bucket_key]))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--run-glob", default="poisson_n256_qcn_fp8_*_r*"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    report = {"results_root": str(args.results_root), "rates": {}}
    for rate, path, records in load_runs(args.results_root, args.run_glob):
        report["rates"][f"{rate:.3f}"] = {
            "path": str(path),
            "overall": summarize(records),
            "input_buckets": bucketize(
                records, "input_sequence_length", INPUT_BUCKETS
            ),
            "output_buckets": bucketize(
                records, "output_sequence_length", OUTPUT_BUCKETS
            ),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Analyzed {len(report['rates'])} completed rate(s)")


if __name__ == "__main__":
    main()
