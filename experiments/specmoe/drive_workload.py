#!/usr/bin/env python3
"""Drive synchronized decode batches and control SGLang expert recording."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


PROMPT_SEEDS = [
    "Explain how a database query planner chooses a join order, with a concrete example.",
    "Write a careful comparison of renewable energy storage technologies and their tradeoffs.",
    "Analyze the causes of a fictional network outage and propose a methodical incident response.",
    "Describe the mathematical intuition behind Fourier transforms for a software engineer.",
    "Draft a technical design for a distributed task queue that tolerates worker failures.",
    "Discuss how compilers optimize loops while preserving observable program behavior.",
    "Explain the history and engineering constraints of high speed railway signaling systems.",
    "Provide a rigorous tutorial on Bayesian inference using a simple diagnostic test example.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30006")
    parser.add_argument(
        "--tokenizer", default="/home/qinchong/models/Qwen3.5-122B-A10B"
    )
    parser.add_argument(
        "--dataset-rows",
        type=Path,
        help="dataset_rows.json produced by prepare_speed_subset.py",
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Disable normal EOS handling (never use for the formal SPEED run)",
    )
    parser.add_argument(
        "--include-output-text",
        action="store_true",
        help="Store generated text for explicit correctness/debug runs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disk-device", default="nvme0n1")
    args = parser.parse_args()
    if args.concurrency < 1 or args.num_requests < 1:
        parser.error("concurrency and num-requests must be positive")
    if args.prompt_tokens < 8 or args.output_tokens < 1:
        parser.error("prompt-tokens must be >= 8 and output-tokens must be positive")
    if args.dataset_rows is not None and not args.dataset_rows.is_file():
        parser.error(f"dataset rows file does not exist: {args.dataset_rows}")
    return args


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def read_system_counters(disk_device: str) -> dict:
    cpu_fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    cpu_ticks = [int(value) for value in cpu_fields]
    disk_fields = None
    for line in Path("/proc/diskstats").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 14 and fields[2] == disk_device:
            disk_fields = fields
            break
    load_1m, load_5m, load_15m = os.getloadavg()
    return {
        "timestamp": time.monotonic(),
        "cpu_total_ticks": sum(cpu_ticks),
        "cpu_idle_ticks": cpu_ticks[3] + cpu_ticks[4],
        "load_average": [load_1m, load_5m, load_15m],
        "disk_device": disk_device,
        "disk_sectors_read": int(disk_fields[5]) if disk_fields else None,
        "disk_sectors_written": int(disk_fields[9]) if disk_fields else None,
        "disk_io_ms": int(disk_fields[12]) if disk_fields else None,
    }


def summarize_system_counters(before: dict, after: dict) -> dict:
    elapsed = after["timestamp"] - before["timestamp"]
    total_ticks = after["cpu_total_ticks"] - before["cpu_total_ticks"]
    idle_ticks = after["cpu_idle_ticks"] - before["cpu_idle_ticks"]
    result = {
        "elapsed_seconds": elapsed,
        "cpu_busy_percent": (
            100.0 * (total_ticks - idle_ticks) / total_ticks if total_ticks else 0.0
        ),
        "load_average_before": before["load_average"],
        "load_average_after": after["load_average"],
        "disk_device": before["disk_device"],
    }
    if before["disk_sectors_read"] is not None and elapsed > 0:
        result.update(
            {
                "disk_read_mib_per_second": (
                    after["disk_sectors_read"] - before["disk_sectors_read"]
                )
                * 512
                / 1024**2
                / elapsed,
                "disk_write_mib_per_second": (
                    after["disk_sectors_written"] - before["disk_sectors_written"]
                )
                * 512
                / 1024**2
                / elapsed,
                "disk_util_percent": (
                    after["disk_io_ms"] - before["disk_io_ms"]
                )
                / elapsed
                / 10.0,
            }
        )
    return result


def build_prompts(tokenizer, count: int, target_tokens: int) -> list[str]:
    prompts = []
    for index in range(count):
        seed = PROMPT_SEEDS[index % len(PROMPT_SEEDS)]
        suffix = (
            f" Request identifier {index}. Give a self-contained answer, retain technical "
            "precision, and reason through relevant constraints before reaching conclusions."
        )
        text = (seed + suffix + " ") * (target_tokens // 20 + 2)
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
        if len(token_ids) != target_tokens:
            raise RuntimeError(
                f"Could only construct {len(token_ids)} tokens for request {index}"
            )
        prompts.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    return prompts


def load_dataset_rows(path: Path) -> tuple[list[str], list[dict[str, Any]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No rows found in {path}")
    required = {
        "submission_index",
        "dataset_row_index",
        "question_id",
        "category",
        "anonymized_request_id",
        "prompt",
        "prompt_sha256",
    }
    for expected_index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Dataset row {expected_index} is missing {sorted(missing)}")
        if int(row["submission_index"]) != expected_index:
            raise RuntimeError("Dataset rows are not in frozen submission order")
        digest = hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest()
        if digest != row["prompt_sha256"]:
            raise RuntimeError(f"Prompt checksum mismatch for row {expected_index}")
    request_ids = [str(row["anonymized_request_id"]) for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("Dataset request IDs are not unique")
    return [str(row["prompt"]) for row in rows], rows, payload


async def recorder_request(session: aiohttp.ClientSession, base_url: str, action: str):
    async with session.post(f"{base_url}/{action}") as response:
        response.raise_for_status()
        return (await response.text()).strip()


async def send_one(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    output_tokens: int,
    request_id: int,
    gate: asyncio.Event | None,
    request_metadata: dict[str, Any] | None,
    ignore_eos: bool,
    include_output_text: bool,
) -> dict:
    if gate is not None:
        await gate.wait()
    payload = {
        "text": prompt,
        "rid": (
            str(request_metadata["anonymized_request_id"])
            if request_metadata is not None
            else f"specmoe-template-{request_id}"
        ),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": ignore_eos,
        },
    }
    joined_at = time.time()
    started = time.perf_counter()
    async with session.post(f"{base_url}/generate", json=payload) as response:
        response.raise_for_status()
        body = await response.json()
    elapsed = time.perf_counter() - started
    completed_at = time.time()
    meta = body.get("meta_info") or {}
    output_text = body.get("text", "")
    if not isinstance(output_text, str):
        output_text = json.dumps(output_text, sort_keys=True, ensure_ascii=False)
    finish_reason = meta.get("finish_reason")
    if isinstance(finish_reason, str):
        finish_type = finish_reason
    elif isinstance(finish_reason, dict):
        finish_type = str(finish_reason.get("type") or "")
    else:
        finish_type = ""
    result = {
        "request_id": request_id,
        "anonymized_request_id": payload["rid"],
        "joined_at_unix_seconds": joined_at,
        "completed_at_unix_seconds": completed_at,
        "elapsed_seconds": elapsed,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens", output_tokens),
        "finish_reason": finish_reason,
        "finished_by_eos_or_stop": finish_type == "stop",
        "finished_by_length": finish_type == "length",
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }
    if request_metadata is not None:
        result.update(
            {
                "dataset_row_index": int(request_metadata["dataset_row_index"]),
                "question_id": str(request_metadata["question_id"]),
                "category": str(request_metadata["category"]),
                "prompt_sha256": str(request_metadata["prompt_sha256"]),
            }
        )
    if include_output_text:
        result["output_text"] = output_text
    return result


async def run(args: argparse.Namespace) -> dict:
    dataset_payload = None
    if args.dataset_rows is not None:
        prompts, request_metadata, dataset_payload = load_dataset_rows(
            args.dataset_rows
        )
        if len(prompts) != args.num_requests:
            raise RuntimeError(
                f"dataset contains {len(prompts)} prompts, but --num-requests="
                f"{args.num_requests}"
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True, local_files_only=True
        )
        prompts = build_prompts(
            tokenizer,
            max(args.num_requests, args.warmup_requests),
            args.prompt_tokens,
        )
        request_metadata = [None] * args.num_requests
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency, 4))

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(f"{args.base_url}/health") as response:
            response.raise_for_status()

        for index in range(args.warmup_requests):
            await send_one(
                session,
                args.base_url,
                prompts[index],
                min(args.output_tokens, 16),
                -index - 1,
                None,
                None,
                args.ignore_eos,
                args.include_output_text,
            )

        if args.record:
            await recorder_request(
                session, args.base_url, "start_expert_distribution_record"
            )

        semaphore = asyncio.Semaphore(args.concurrency)
        gate = asyncio.Event()

        async def bounded(index: int) -> dict:
            async with semaphore:
                return await send_one(
                    session,
                    args.base_url,
                    prompts[index],
                    args.output_tokens,
                    index,
                    gate,
                    request_metadata[index],
                    args.ignore_eos,
                    args.include_output_text,
                )

        tasks = [asyncio.create_task(bounded(i)) for i in range(args.num_requests)]
        await asyncio.sleep(0.1)
        system_before = read_system_counters(args.disk_device)
        batch_started = time.perf_counter()
        gate.set()
        results = await asyncio.gather(*tasks)
        batch_elapsed = time.perf_counter() - batch_started
        system_after = read_system_counters(args.disk_device)

        if args.record:
            await recorder_request(
                session, args.base_url, "stop_expert_distribution_record"
            )
            await recorder_request(
                session, args.base_url, "dump_expert_distribution_record"
            )

    latencies = [item["elapsed_seconds"] for item in results]
    total_completion_tokens = sum(
        int(item["completion_tokens"] or args.output_tokens) for item in results
    )
    return {
        "configuration": {
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "num_requests": args.num_requests,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "record": args.record,
            "ignore_eos": args.ignore_eos,
            "eos_policy": "ignored" if args.ignore_eos else "normal",
            "include_output_text": args.include_output_text,
            "dataset_rows": str(args.dataset_rows) if args.dataset_rows else None,
            "dataset_revision": (
                dataset_payload["dataset"]["revision"]
                if dataset_payload is not None
                else None
            ),
        },
        "batch_elapsed_seconds": batch_elapsed,
        "output_throughput_tokens_per_second": total_completion_tokens
        / batch_elapsed,
        "request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "system_observation": summarize_system_counters(
            system_before, system_after
        ),
        "requests": results,
    }


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
