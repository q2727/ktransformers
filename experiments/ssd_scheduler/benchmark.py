#!/usr/bin/env python3
"""Run deterministic requests and persist token IDs for SSD comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


DEFAULT_PROMPTS = [
    "Explain in one concise paragraph why speculative decoding can be lossless.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h?",
]


def generate(url: str, prompt: str, max_new_tokens: int) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    latency_s = time.perf_counter() - begin
    token_ids = [
        int(entry[1]) for entry in result["meta_info"]["output_token_logprobs"]
    ]
    digest = hashlib.sha256(
        ",".join(map(str, token_ids)).encode("ascii")
    ).hexdigest()
    return {
        "latency_s": latency_s,
        "completion_tokens": len(token_ids),
        "token_ids": token_ids,
        "token_sha256": digest,
        "text": result["text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30020")
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    records = []
    for prompt_index, prompt in enumerate(prompts):
        for repetition in range(args.repetitions):
            result = generate(args.url, prompt, args.max_new_tokens)
            result.update(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "repetition": repetition,
                }
            )
            records.append(result)
            print(
                f"label={args.label} prompt={prompt_index} repetition={repetition} "
                f"latency={result['latency_s']:.3f}s "
                f"tokens={result['completion_tokens']} "
                f"sha256={result['token_sha256'][:12]}",
                flush=True,
            )

    latencies = [record["latency_s"] for record in records]
    output = {
        "label": args.label,
        "url": args.url,
        "max_new_tokens": args.max_new_tokens,
        "repetitions": args.repetitions,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_median_s": statistics.median(latencies),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
