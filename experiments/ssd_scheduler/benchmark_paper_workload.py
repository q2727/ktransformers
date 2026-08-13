#!/usr/bin/env python3
"""Run sequential B=1 greedy requests on an SSD-paper-aligned prompt mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


DATASET_ORDER = ("humaneval", "alpaca", "gsm8k", "ultrafeedback")


def load_prompts(
    dataset_dir: Path,
    num_per_dataset: int,
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> list[dict]:
    prompts = []
    for name in datasets:
        path = dataset_dir / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) < num_per_dataset:
            raise ValueError(
                f"{path} contains {len(rows)} prompts, need {num_per_dataset}"
            )
        prompts.extend(rows[:num_per_dataset])
    return prompts


def generate(
    url: str,
    input_ids: list[int],
    max_new_tokens: int,
    timeout: float,
    response_mode: str,
) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
    }
    if response_mode == "token-ids":
        payload["return_logprob"] = True
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    latency_s = time.perf_counter() - begin
    meta_info = result["meta_info"]
    spec_stats = {
        key: meta_info[key]
        for key in (
            "spec_accept_rate",
            "spec_accept_length",
            "spec_accept_token_num",
            "spec_draft_token_num",
            "spec_verify_ct",
        )
        if key in meta_info
    }
    if response_mode == "text":
        output_text = result["text"]
        if not isinstance(output_text, str):
            raise TypeError(f"Expected a text response, got {type(output_text)!r}")
        return {
            "latency_s": latency_s,
            "completion_tokens": int(meta_info["completion_tokens"]),
            "output_text_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            **spec_stats,
        }

    token_ids = [
        int(entry[1]) for entry in meta_info["output_token_logprobs"]
    ]
    return {
        "latency_s": latency_s,
        "completion_tokens": len(token_ids),
        "token_ids": token_ids,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, token_ids)).encode()
        ).hexdigest(),
        **spec_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30020")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--num-per-dataset", type=int, default=8)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="Dataset subsets to run, in the requested order.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--response-mode",
        choices=("token-ids", "text"),
        default="token-ids",
        help=(
            "Use token logprobs for exact token IDs, or hash the complete raw "
            "response text for runtimes that do not support return_logprob."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        help="Physical NVML GPU index for total-energy measurement",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = load_prompts(
        args.dataset_dir, args.num_per_dataset, tuple(args.datasets)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")

    latencies = []
    total_tokens = 0
    energy_handle = None
    energy_start_mj = None
    energy_begin = None
    if args.gpu_index is not None:
        import pynvml

        pynvml.nvmlInit()
        energy_handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        energy_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(
            energy_handle
        )
        energy_begin = time.perf_counter()
    with args.output.open("a") as output_file:
        for ordinal, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
            if len(input_ids) > args.max_input_tokens:
                raise ValueError(
                    f"{prompt['dataset']}[{prompt['index']}] has {len(input_ids)} "
                    f"tokens, exceeding --max-input-tokens={args.max_input_tokens}"
                )
            result = generate(
                args.url,
                input_ids,
                args.max_new_tokens,
                args.timeout,
                args.response_mode,
            )
            record = {
                "label": args.label,
                "ordinal": ordinal,
                "dataset": prompt["dataset"],
                "dataset_index": prompt["index"],
                "prompt_tokens": len(input_ids),
                **result,
            }
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()
            latencies.append(result["latency_s"])
            total_tokens += result["completion_tokens"]
            print(
                f"[{ordinal + 1}/{len(prompts)}] {prompt['dataset']}"
                f"[{prompt['index']}] input={len(input_ids)} "
                f"output={result['completion_tokens']} "
                f"latency={result['latency_s']:.3f}s",
                flush=True,
            )

    energy_summary = {}
    if energy_handle is not None:
        energy_elapsed_s = time.perf_counter() - energy_begin
        energy_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(energy_handle)
        energy_j = (energy_end_mj - energy_start_mj) / 1e3
        energy_summary = {
            "gpu_energy_j": energy_j,
            "gpu_energy_elapsed_s": energy_elapsed_s,
            "gpu_average_power_w": energy_j / energy_elapsed_s,
            "gpu_j_per_output_token": energy_j / total_tokens,
        }
        pynvml.nvmlShutdown()

    summary = {
        "label": args.label,
        "requests": len(prompts),
        "datasets": args.datasets,
        "num_per_dataset": args.num_per_dataset,
        "max_new_tokens": args.max_new_tokens,
        "response_mode": args.response_mode,
        "total_tokens": total_tokens,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_median_s": statistics.median(latencies),
        "throughput_tok_s": total_tokens / sum(latencies),
        **energy_summary,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
