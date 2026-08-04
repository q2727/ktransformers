#!/usr/bin/env python3
"""Summarize the Qwen3.5 KT dual-batch overlap reproduction."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED = {
    "single_request": 1,
    "baseline": 2,
    "dual_batch": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/artifacts/dual_batch_overlap"),
    )
    return parser.parse_args()


def summarize_mode(root: Path, mode: str, expected_requests: int) -> dict:
    paths = sorted((root / mode).glob("run*.json"))
    if not paths:
        raise RuntimeError(f"No repeated runs found for {mode!r} under {root}")

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, row in zip(paths, rows, strict=True):
        config = row["configuration"]
        if config["num_requests"] != expected_requests:
            raise RuntimeError(f"Unexpected request count in {path}")
        if config["prompt_tokens"] != 32 or config["output_tokens"] != 128:
            raise RuntimeError(f"Unexpected token lengths in {path}")
        if not config["ignore_eos"]:
            raise RuntimeError(f"EOS was not disabled in throughput run {path}")
        if len(row["requests"]) != expected_requests:
            raise RuntimeError(f"Incomplete request list in {path}")
        if not all(
            request["completion_tokens"] == 128
            and request["finished_by_length"]
            for request in row["requests"]
        ):
            raise RuntimeError(f"Incomplete generation in {path}")

    throughputs = [row["output_throughput_tokens_per_second"] for row in rows]
    cpu_busy = [row["system_observation"]["cpu_busy_percent"] for row in rows]
    return {
        "runs": len(rows),
        "throughput_tokens_per_second": {
            "values": throughputs,
            "mean": statistics.mean(throughputs),
            "median": statistics.median(throughputs),
            "stdev": statistics.stdev(throughputs) if len(rows) > 1 else 0.0,
            "min": min(throughputs),
            "max": max(throughputs),
        },
        "mean_cpu_busy_percent": statistics.mean(cpu_busy),
        "all_requests_finished_by_length": True,
    }


def correctness_summary(root: Path) -> dict:
    baseline_path = root / "correctness" / "baseline.json"
    dual_path = root / "correctness" / "dual_batch.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    dual = json.loads(dual_path.read_text(encoding="utf-8"))
    pairs = list(zip(baseline["requests"], dual["requests"], strict=True))
    return {
        "requests": len(pairs),
        "output_tokens_per_request": baseline["configuration"]["output_tokens"],
        "hashes_equal": [a["output_sha256"] == b["output_sha256"] for a, b in pairs],
        "texts_equal": [a["output_text"] == b["output_text"] for a, b in pairs],
    }


def main() -> None:
    args = parse_args()
    modes = {
        mode: summarize_mode(args.root, mode, request_count)
        for mode, request_count in EXPECTED.items()
    }
    single = modes["single_request"]["throughput_tokens_per_second"]["mean"]
    baseline = modes["baseline"]["throughput_tokens_per_second"]["mean"]
    dual = modes["dual_batch"]["throughput_tokens_per_second"]["mean"]
    result = {
        "modes": modes,
        "comparisons": {
            "ordinary_batch2_vs_single_system_throughput": baseline / single,
            "dual_batch_vs_single_system_throughput": dual / single,
            "dual_batch_vs_ordinary_batch2_system_throughput": dual / baseline,
            "ordinary_batch2_per_request_vs_single": (baseline / 2) / single,
            "dual_batch_per_request_vs_single": (dual / 2) / single,
        },
        "correctness": correctness_summary(args.root),
        "paper_reference": {
            "single_stream_tokens_per_second": 21.5,
            "dual_stream_system_tokens_per_second": 33.6,
            "dual_vs_single_system_throughput": 33.6 / 21.5,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
