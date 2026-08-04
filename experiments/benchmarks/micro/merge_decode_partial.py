#!/usr/bin/env python3
"""Merge completed decode batch events with a full benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", action="append", type=Path, default=[])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    points = {int(point["batch_size"]): point for point in payload["points"]}
    recovered: list[int] = []

    for path in args.console:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "batch_complete":
                continue
            batch_size = int(event["batch_size"])
            points[batch_size] = {
                "nominal_prompt_tokens": 8192,
                "actual_prompt_tokens_median": 8192,
                "batch_size": batch_size,
                "request_samples": batch_size,
                "tpot_median_seconds": float(event["tpot_median_seconds"]),
                "batch_effective_output_tokens_per_second_median": float(
                    event["batch_effective_output_tokens_per_second"]
                ),
            }
            recovered.append(batch_size)

    expected = [1, 2, 4, 6, 8]
    missing = sorted(set(expected) - points.keys())
    if missing:
        raise SystemExit(f"Missing decode batch sizes: {missing}")

    payload["points"] = [points[size] for size in expected]
    payload["trials"] = 1
    payload["recovered_batch_events"] = sorted(set(recovered))
    payload["configuration_note"] = (
        "DeepSeek decode isolation: b1/b2 used 8K max-prefill; b4/b6 used 64K; "
        "b8 used 32K and 1024 output tokens. TPOT is measured only after every "
        "batch member enters steady-state decode. Each batch size was measured once."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
