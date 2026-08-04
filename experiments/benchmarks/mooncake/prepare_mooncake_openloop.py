#!/usr/bin/env python3
"""Build time-scaled, contiguous Mooncake trace windows for open-loop replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=9236)
    parser.add_argument("--request-count", type=int, default=1024)
    parser.add_argument("--warmup-count", type=int, default=256)
    parser.add_argument("--warmup-output-length", type=int)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.08, 0.10, 0.12])
    return parser.parse_args()


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def describe(rows: list[dict]) -> dict:
    inputs = [int(row["input_length"]) for row in rows]
    outputs = [int(row["output_length"]) for row in rows]
    return {
        "count": len(rows),
        "source_timestamp_first_ms": rows[0]["timestamp"],
        "source_timestamp_last_ms": rows[-1]["timestamp"],
        "source_span_ms": rows[-1]["timestamp"] - rows[0]["timestamp"],
        "input_length": {
            "mean": statistics.fmean(inputs),
            "p50": percentile(inputs, 0.50),
            "p90": percentile(inputs, 0.90),
            "p95": percentile(inputs, 0.95),
            "p99": percentile(inputs, 0.99),
            "max": max(inputs),
        },
        "output_length": {
            "mean": statistics.fmean(outputs),
            "p50": percentile(outputs, 0.50),
            "p90": percentile(outputs, 0.90),
            "p95": percentile(outputs, 0.95),
            "p99": percentile(outputs, 0.99),
            "max": max(outputs),
        },
        "max_total_length": max(
            int(row["input_length"]) + int(row["output_length"]) for row in rows
        ),
    }


def rate_tag(rate: float) -> str:
    return f"{rate:.3f}".replace(".", "p").rstrip("0")


def scale_timestamps(rows: list[dict], target_rate: float) -> tuple[list[dict], dict]:
    if len(rows) < 2:
        raise ValueError("At least two trace records are required")
    first = float(rows[0]["timestamp"])
    original_span_ms = float(rows[-1]["timestamp"]) - first
    if original_span_ms <= 0:
        target_span_ms = (len(rows) - 1) / target_rate * 1000.0
        result = [
            {**row, "timestamp": round(index / target_rate * 1000.0)}
            for index, row in enumerate(rows)
        ]
        actual_span_ms = result[-1]["timestamp"] - result[0]["timestamp"]
        return result, {
            "target_rate_requests_per_second": target_rate,
            "original_span_ms": original_span_ms,
            "target_span_ms": target_span_ms,
            "actual_span_ms": actual_span_ms,
            "timestamp_scale": None,
            "synthetic_even_spacing": True,
            "actual_average_rate_requests_per_second": (len(result) - 1)
            / (actual_span_ms / 1000.0),
        }

    target_span_ms = (len(rows) - 1) / target_rate * 1000.0
    scale = target_span_ms / original_span_ms
    result: list[dict] = []
    previous = -1
    for row in rows:
        timestamp = round((float(row["timestamp"]) - first) * scale)
        if timestamp < previous:
            raise ValueError("Scaled timestamps are not monotonic")
        previous = timestamp
        result.append({**row, "timestamp": timestamp})

    actual_span_ms = result[-1]["timestamp"] - result[0]["timestamp"]
    metadata = {
        "target_rate_requests_per_second": target_rate,
        "original_span_ms": original_span_ms,
        "target_span_ms": target_span_ms,
        "actual_span_ms": actual_span_ms,
        "timestamp_scale": scale,
        "actual_average_rate_requests_per_second": (len(result) - 1)
        / (actual_span_ms / 1000.0),
    }
    return result, metadata


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_rows = [json.loads(line) for line in args.source.open(encoding="utf-8")]
    start = args.start_index
    end = start + args.request_count
    warm_start = start - args.warmup_count
    if warm_start < 0 or end > len(source_rows):
        raise ValueError("Requested trace window is outside the source dataset")

    warmup = source_rows[warm_start:start]
    if args.warmup_output_length is not None:
        if args.warmup_output_length < 1:
            raise ValueError("Warmup output length must be positive")
        warmup = [
            {**row, "output_length": args.warmup_output_length} for row in warmup
        ]
    measurement = source_rows[start:end]
    for name, rows in (("warmup", warmup), ("measurement", measurement)):
        timestamps = [row["timestamp"] for row in rows]
        if timestamps != sorted(timestamps):
            raise ValueError(f"{name} timestamps are not monotonic")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.source),
        "source_total_records": len(source_rows),
        "source_indexing": "zero-based, end-exclusive",
        "warmup_source_indices": [warm_start, start],
        "measurement_source_indices": [start, end],
        "warmup_source_lines_one_based": [warm_start + 1, start],
        "measurement_source_lines_one_based": [start + 1, end],
        "warmup_output_length_override": args.warmup_output_length,
        "warmup": describe(warmup),
        "measurement": describe(measurement),
        "rates": {},
    }

    for rate in args.rates:
        tag = rate_tag(rate)
        warm_scaled, warm_meta = scale_timestamps(warmup, rate)
        measure_scaled, measure_meta = scale_timestamps(measurement, rate)
        warm_path = args.output_dir / f"toolagent_openloop_warm{len(warmup)}_r{tag}.jsonl"
        measure_path = (
            args.output_dir
            / f"toolagent_openloop_measure{len(measurement)}_r{tag}.jsonl"
        )
        manifest["rates"][str(rate)] = {
            "warmup": {
                "path": str(warm_path),
                "sha256": write_jsonl(warm_path, warm_scaled),
                **warm_meta,
            },
            "measurement": {
                "path": str(measure_path),
                "sha256": write_jsonl(measure_path, measure_scaled),
                **measure_meta,
            },
        }

    canary = [
        {
            "timestamp": index * 250,
            "input_length": 32 + index,
            "output_length": 4,
            "hash_ids": [900000 + index],
        }
        for index in range(8)
    ]
    canary_path = args.output_dir / "toolagent_openloop_canary8.jsonl"
    manifest["canary"] = {
        "path": str(canary_path),
        "sha256": write_jsonl(canary_path, canary),
    }

    manifest_path = args.output_dir / "toolagent_openloop_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
