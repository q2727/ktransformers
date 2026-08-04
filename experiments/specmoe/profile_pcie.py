#!/usr/bin/env python3
"""Measure pinned host-to-device bandwidth for expert-sized transfers."""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-bytes", type=int, default=9_437_184)
    parser.add_argument("--expert-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def benchmark_copy(
    expert_bytes: int, expert_count: int, warmup: int, iterations: int, device: str
) -> dict:
    total_bytes = expert_bytes * expert_count
    host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    host.random_(0, 256)
    stream = torch.cuda.Stream(device=device)

    for _ in range(warmup):
        with torch.cuda.stream(stream):
            gpu.copy_(host, non_blocking=True)
    stream.synchronize()

    elapsed_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            gpu.copy_(host, non_blocking=True)
            end.record(stream)
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    mean_ms = statistics.fmean(elapsed_ms)
    return {
        "expert_count": expert_count,
        "bytes": total_bytes,
        "mean_ms": mean_ms,
        "p50_ms": percentile(elapsed_ms, 0.50),
        "p90_ms": percentile(elapsed_ms, 0.90),
        "p99_ms": percentile(elapsed_ms, 0.99),
        "gib_per_second": total_bytes / (mean_ms / 1000.0) / 1024**3,
        "microseconds_per_expert": mean_ms * 1000.0 / expert_count,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    result = {
        "device": torch.cuda.get_device_name(args.device),
        "expert_bytes": args.expert_bytes,
        "transfers": [
            benchmark_copy(
                args.expert_bytes,
                count,
                args.warmup,
                args.iterations,
                args.device,
            )
            for count in args.expert_counts
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
