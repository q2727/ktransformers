#!/usr/bin/env python3
"""Measure standalone SSD draft JIT and outcome-cache construction latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

from sglang.srt.speculative.ssd_draft_client import SSDDraftClient


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def run_one(
    client: SSDDraftClient,
    base_prefix: list[int],
    draft_length: int,
    fan_out: int,
    warmups: int,
    repetitions: int,
) -> dict:
    prefix = list(base_prefix)
    jit_ms = []
    cache_ms = []
    for iteration in range(warmups + repetitions):
        begin = time.perf_counter()
        candidate = client.jit_draft(prefix, draft_length, fan_out)
        current_jit_ms = (time.perf_counter() - begin) * 1e3

        begin = time.perf_counter()
        client.build_outcome_cache(prefix, candidate, draft_length, fan_out)
        current_cache_ms = (time.perf_counter() - begin) * 1e3

        endpoint = iteration % (draft_length + 1)
        recovery = candidate.recovery_tokens[endpoint][0]
        prefix.extend(candidate.tokens[:endpoint])
        prefix.append(recovery)
        if iteration >= warmups:
            jit_ms.append(current_jit_ms)
            cache_ms.append(current_cache_ms)

    return {
        "fan_out": fan_out,
        "branch_batch": (draft_length + 1) * fan_out,
        "jit": summary(jit_ms),
        "outcome_cache": summary(cache_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31021")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--draft-length", type=int, default=5)
    parser.add_argument("--fan-outs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix = tokenizer.encode(
        "Write a Python function that returns the first n Fibonacci numbers.",
        add_special_tokens=False,
    )
    client = SSDDraftClient(args.url)
    results = [
        run_one(
            client,
            prefix,
            args.draft_length,
            fan_out,
            args.warmups,
            args.repetitions,
        )
        for fan_out in args.fan_outs
    ]
    output = {
        "label": args.label,
        "draft_length": args.draft_length,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
