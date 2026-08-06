#!/usr/bin/env python3
"""Probe SSD at controlled prompt lengths using a shared natural-text suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


DATASET_FILES = ("humaneval.jsonl", "alpaca.jsonl", "gsm8k.jsonl", "ultrafeedback.jsonl")


def generate(
    url: str, input_ids: list[int], max_new_tokens: int, timeout: float
) -> dict:
    payload = {
        "input_ids": input_ids,
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
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    latency_s = time.perf_counter() - begin
    token_ids = [
        int(entry[1]) for entry in result["meta_info"]["output_token_logprobs"]
    ]
    return {
        "latency_s": latency_s,
        "completion_tokens": len(token_ids),
        "token_ids": token_ids,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, token_ids)).encode()
        ).hexdigest(),
    }


def load_natural_token_stream(
    dataset_dir: Path, tokenizer: AutoTokenizer
) -> tuple[list[int], list[int]]:
    texts = []
    suffix_text = None
    for filename in DATASET_FILES:
        rows = [
            json.loads(line)
            for line in (dataset_dir / filename).read_text().splitlines()
        ]
        texts.extend(row["text"] for row in rows)
        if filename == "humaneval.jsonl":
            suffix_text = rows[0]["text"]
    if suffix_text is None:
        raise RuntimeError("No HumanEval suffix prompt found.")
    filler = tokenizer.encode("\n\n".join(texts), add_special_tokens=False)
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    return filler, suffix


def make_context(filler: list[int], suffix: list[int], length: int) -> list[int]:
    if length < len(suffix):
        return suffix[-length:]
    filler_needed = length - len(suffix)
    if filler_needed > len(filler):
        repeats = (filler_needed + len(filler) - 1) // len(filler)
        filler = (filler * repeats)[:filler_needed]
    else:
        filler = filler[:filler_needed]
    result = [*filler, *suffix]
    if len(result) != length:
        raise AssertionError(f"Expected {length} tokens, constructed {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31020")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--lengths", default="128,512,1024,1536")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("--lengths must contain positive comma-separated integers")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    filler, suffix = load_natural_token_stream(args.dataset_dir, tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")
    latencies = []
    total_tokens = 0
    with args.output.open("a") as output_file:
        for ordinal, length in enumerate(lengths):
            input_ids = make_context(filler, suffix, length)
            result = generate(args.url, input_ids, args.max_new_tokens, args.timeout)
            record = {
                "label": args.label,
                "ordinal": ordinal,
                "dataset": "context_length",
                "dataset_index": length,
                "prompt_tokens": length,
                **result,
            }
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()
            latencies.append(result["latency_s"])
            total_tokens += result["completion_tokens"]
            print(
                f"[{ordinal + 1}/{len(lengths)}] context={length} "
                f"output={result['completion_tokens']} "
                f"latency={result['latency_s']:.3f}s",
                flush=True,
            )

    summary = {
        "label": args.label,
        "requests": len(lengths),
        "context_lengths": lengths,
        "max_new_tokens": args.max_new_tokens,
        "total_tokens": total_tokens,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_median_s": statistics.median(latencies),
        "throughput_tok_s": total_tokens / sum(latencies),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
