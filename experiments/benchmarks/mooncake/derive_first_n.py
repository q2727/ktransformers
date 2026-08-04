#!/usr/bin/env python3
"""Derive a reproducible first-N view from a completed AIPerf run.

The JSONL exported by AIPerf is not guaranteed to be in submission order, so
records are selected and ordered by ``metadata.session_num``.  The source run
is never modified.  This utility writes the selected records plus a compact
summary whose throughput window ends when the last selected request finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
SCALAR_METRICS = (
    "request_latency",
    "time_to_first_token",
    "time_to_second_token",
    "inter_token_latency",
    "input_sequence_length",
    "output_sequence_length",
    "output_token_count",
    "output_token_throughput_per_user",
    "e2e_output_token_throughput",
    "http_req_waiting",
)


def percentile(sorted_values: list[float], q: int) -> float:
    """Match NumPy's default linear percentile interpolation."""
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def summarize(values: Iterable[float], unit: str) -> dict[str, Any]:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return {"unit": unit, "count": 0}
    avg = sum(data) / len(data)
    result: dict[str, Any] = {
        "unit": unit,
        "avg": avg,
        "min": data[0],
        "max": data[-1],
        "std": math.sqrt(sum((value - avg) ** 2 for value in data) / len(data)),
        "count": len(data),
        "sum": sum(data),
    }
    for q in PERCENTILES:
        result[f"p{q}"] = percentile(data, q)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(rows: list[dict[str, Any]], name: str) -> tuple[list[float], str]:
    values: list[float] = []
    unit = ""
    for row in rows:
        metric = row.get("metrics", {}).get(name)
        if not metric:
            continue
        unit = metric.get("unit", unit)
        value = metric.get("value")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values, unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--target-rate", type=float)
    parser.add_argument("--capacity", type=float)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    source = args.source_dir.resolve() / "profile_export.jsonl"
    if not source.is_file():
        raise SystemExit(f"missing AIPerf records: {source}")

    all_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if len(all_rows) < args.count:
        raise SystemExit(f"source has {len(all_rows)} records; need {args.count}")

    session_rows: dict[int, dict[str, Any]] = {}
    for row in all_rows:
        session_num = int(row["metadata"]["session_num"])
        if session_num in session_rows:
            raise SystemExit(f"duplicate session_num: {session_num}")
        session_rows[session_num] = row
    selected_ids = sorted(session_rows)[: args.count]
    if selected_ids != list(range(selected_ids[0], selected_ids[0] + args.count)):
        raise SystemExit("the first-N session numbers are not contiguous")
    rows = [session_rows[session_num] for session_num in selected_ids]

    request_start_ns = min(int(row["metadata"]["request_start_ns"]) for row in rows)
    request_end_ns = max(int(row["metadata"]["request_end_ns"]) for row in rows)
    first_credit_ns = min(int(row["metadata"]["credit_issued_ns"]) for row in rows)
    last_credit_ns = max(int(row["metadata"]["credit_issued_ns"]) for row in rows)
    duration_sec = (request_end_ns - request_start_ns) / 1e9
    arrival_span_sec = (last_credit_ns - first_credit_ns) / 1e9
    if duration_sec <= 0:
        raise SystemExit("invalid selected request time window")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / f"profile_export.first{args.count}.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics: dict[str, Any] = {}
    for name in SCALAR_METRICS:
        values, unit = metric_values(rows, name)
        if values:
            metrics[name] = summarize(values, unit)

    inter_chunk_values: list[float] = []
    inter_chunk_unit = ""
    for row in rows:
        metric = row.get("metrics", {}).get("inter_chunk_latency")
        if not metric:
            continue
        inter_chunk_unit = metric.get("unit", inter_chunk_unit)
        value = metric.get("value")
        if isinstance(value, list):
            inter_chunk_values.extend(float(item) for item in value)
    if inter_chunk_values:
        metrics["inter_chunk_latency"] = summarize(inter_chunk_values, inter_chunk_unit)

    total_output_tokens = sum(
        row.get("metrics", {}).get("output_token_count", {}).get("value", 0) for row in rows
    )
    total_input_tokens = sum(
        row.get("metrics", {}).get("input_sequence_length", {}).get("value", 0) for row in rows
    )
    good_requests = sum(
        row.get("metrics", {}).get("good_request_count", {}).get("value", 0) for row in rows
    )
    cancelled = sum(bool(row.get("metadata", {}).get("was_cancelled")) for row in rows)

    summary: dict[str, Any] = {
        "schema_version": "first-n-derived-v1",
        "selection": {
            "rule": "lowest contiguous metadata.session_num values",
            "count": args.count,
            "session_num_first": selected_ids[0],
            "session_num_last": selected_ids[-1],
            "source_record_count": len(all_rows),
            "source_directory": str(args.source_dir.resolve()),
            "source_profile_sha256": sha256(source),
            "selected_profile": selected_path.name,
            "selected_profile_sha256": sha256(selected_path),
            "label": args.label,
        },
        "workload": {
            "target_request_rate_requests_per_second": args.target_rate,
            "closed_loop_capacity_requests_per_second": args.capacity,
            "capacity_fraction": (
                args.target_rate / args.capacity
                if args.target_rate is not None and args.capacity is not None
                else None
            ),
        },
        "window": {
            "request_start_ns": request_start_ns,
            "request_end_ns": request_end_ns,
            "duration_seconds": duration_sec,
            "first_credit_issued_ns": first_credit_ns,
            "last_credit_issued_ns": last_credit_ns,
            "arrival_span_seconds": arrival_span_sec,
        },
        "aggregate": {
            "request_count": args.count,
            "request_throughput_requests_per_second": args.count / duration_sec,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "output_token_throughput_tokens_per_second": total_output_tokens / duration_sec,
            "total_token_throughput_tokens_per_second": (
                total_input_tokens + total_output_tokens
            )
            / duration_sec,
            "good_request_count": good_requests,
            "goodput_requests_per_second": good_requests / duration_sec,
            "cancelled_request_count": cancelled,
        },
        "metrics": metrics,
        "notes": [
            "No inference requests were replayed.",
            "Later requests from the original run may have overlapped the selected requests; this preserves the observed serving contention.",
            "GPU telemetry is not recomputed because the original samples cover the full source run.",
        ],
    }
    summary_path = args.output_dir / f"derived_first{args.count}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
