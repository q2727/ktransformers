#!/usr/bin/env python3
"""Run a fixed prompt mix with bounded HTTP request concurrency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from benchmark_paper_workload import generate, load_prompts


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31020")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--num-per-dataset", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = load_prompts(args.dataset_dir, args.num_per_dataset)
    jobs = []
    for ordinal, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
        if len(input_ids) > args.max_input_tokens:
            raise ValueError(
                f"{prompt['dataset']}[{prompt['index']}] has {len(input_ids)} "
                f"tokens, exceeding --max-input-tokens={args.max_input_tokens}"
            )
        jobs.append((ordinal, prompt, input_ids))

    energy_handle = None
    energy_start_mj = None
    if args.gpu_index is not None:
        import pynvml

        pynvml.nvmlInit()
        energy_handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        energy_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(
            energy_handle
        )

    def run(job: tuple[int, dict, list[int]]) -> dict:
        ordinal, prompt, input_ids = job
        result = generate(args.url, input_ids, args.max_new_tokens, args.timeout)
        return {
            "label": args.label,
            "ordinal": ordinal,
            "dataset": prompt["dataset"],
            "dataset_index": prompt["index"],
            "prompt_tokens": len(input_ids),
            **result,
        }

    wall_begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        records = list(executor.map(run, jobs))
    wall_s = time.perf_counter() - wall_begin

    energy_summary = {}
    if energy_handle is not None:
        energy_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(energy_handle)
        energy_j = (energy_end_mj - energy_start_mj) / 1e3
        total_tokens = sum(row["completion_tokens"] for row in records)
        energy_summary = {
            "gpu_energy_j": energy_j,
            "gpu_average_power_w": energy_j / wall_s,
            "gpu_j_per_output_token": energy_j / total_tokens,
        }
        pynvml.nvmlShutdown()

    records.sort(key=lambda row: row["ordinal"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    latencies = [row["latency_s"] for row in records]
    total_tokens = sum(row["completion_tokens"] for row in records)
    summary = {
        "label": args.label,
        "requests": len(records),
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "total_tokens": total_tokens,
        "wall_s": wall_s,
        "throughput_tok_s": total_tokens / wall_s,
        "request_latency_mean_s": statistics.fmean(latencies),
        "request_latency_p50_s": percentile(latencies, 0.50),
        "request_latency_p95_s": percentile(latencies, 0.95),
        "request_latency_max_s": max(latencies),
        **energy_summary,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
