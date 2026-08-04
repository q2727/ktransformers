#!/usr/bin/env python3
"""Drive one fixed batch-8 timing sample in natural and CPU-isolated modes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30008")
    parser.add_argument("--dataset-rows", type=Path, required=True)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="/home/qinchong/models/Qwen3.5-122B-A10B"
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        help="Use exact token-id inputs of this length for every request",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup-output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--record-experts", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("natural", "cpu_isolate"),
        default=("natural", "cpu_isolate"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size != 8:
        parser.error("this focused experiment requires --batch-size=8")
    if args.output_tokens < 1 or args.warmup_output_tokens < 1:
        parser.error("output token counts must be positive")
    if args.context_tokens is not None and args.context_tokens < 1:
        parser.error("--context-tokens must be positive")
    if len(args.modes) != len(set(args.modes)):
        parser.error("--modes cannot contain duplicates")
    return args


def load_rows(path: Path, batch_size: int) -> tuple[list[dict[str, Any]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < batch_size:
        raise RuntimeError(f"{path} does not contain {batch_size} rows")
    selected = rows[:batch_size]
    for index, row in enumerate(rows):
        if int(row["submission_index"]) != index:
            raise RuntimeError("dataset rows are not in frozen submission order")
        digest = hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest()
        if digest != row["prompt_sha256"]:
            raise RuntimeError(f"prompt checksum mismatch at row {index}")
    return selected, payload


def build_fixed_contexts(
    *,
    all_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    tokenizer_path: str,
    context_tokens: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Build varied exact-length contexts from the frozen SPEED prompt corpus."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    prompt_ids = [
        tokenizer.encode(str(row["prompt"]), add_special_tokens=False)
        for row in all_rows
    ]
    separator_ids = tokenizer.encode(
        "\n\n--- Next SPEED-Bench context ---\n\n", add_special_tokens=False
    )
    contexts: list[list[int]] = []
    manifests: list[dict[str, Any]] = []
    for request_index, row in enumerate(selected_rows):
        query_ids = prompt_ids[int(row["submission_index"])]
        if len(query_ids) > context_tokens:
            raise RuntimeError(
                f"base prompt {request_index} has {len(query_ids)} tokens, "
                f"longer than requested context {context_tokens}"
            )
        prefix: list[int] = []
        source_indices: list[int] = []
        cursor = (request_index + 1) % len(all_rows)
        required_prefix = context_tokens - len(query_ids)
        while len(prefix) < required_prefix:
            source_indices.append(cursor)
            prefix.extend(prompt_ids[cursor])
            prefix.extend(separator_ids)
            cursor = (cursor + 1) % len(all_rows)
        input_ids = prefix[:required_prefix] + query_ids
        if len(input_ids) != context_tokens:
            raise AssertionError("fixed context construction produced the wrong length")
        digest = hashlib.sha256(
            json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        contexts.append(input_ids)
        manifests.append(
            {
                "submission_index": int(row["submission_index"]),
                "anonymized_request_id": str(row["anonymized_request_id"]),
                "base_prompt_tokens": len(query_ids),
                "input_tokens": len(input_ids),
                "input_ids_sha256": digest,
                "prefix_source_submission_indices": source_indices,
                "construction": (
                    "cyclic frozen SPEED prompts separated by a fixed tokenized "
                    "delimiter, truncated as a prefix; the original request is the suffix"
                ),
            }
        )
    return contexts, manifests


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


async def send_one(
    session: aiohttp.ClientSession,
    base_url: str,
    row: dict[str, Any],
    input_ids: list[int] | None,
    mode: str,
    output_tokens: int,
    ignore_eos: bool,
    gate: asyncio.Event,
) -> dict[str, Any]:
    await gate.wait()
    rid = f"component-{mode}-{row['anonymized_request_id']}"
    request = {
        "rid": rid,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": ignore_eos,
        },
    }
    if input_ids is None:
        request["text"] = row["prompt"]
    else:
        request["input_ids"] = input_ids
    started = time.perf_counter()
    async with session.post(f"{base_url}/generate", json=request) as response:
        response.raise_for_status()
        body = await response.json()
    elapsed = time.perf_counter() - started
    meta = body.get("meta_info") or {}
    output_text = body.get("text", "")
    if not isinstance(output_text, str):
        output_text = json.dumps(output_text, sort_keys=True, ensure_ascii=False)
    return {
        "rid": rid,
        "submission_index": int(row["submission_index"]),
        "anonymized_request_id": str(row["anonymized_request_id"]),
        "category": str(row["category"]),
        "prompt_sha256": str(row["prompt_sha256"]),
        "elapsed_seconds": elapsed,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }


async def run_batch(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    contexts: list[list[int]] | None,
    mode: str,
    output_tokens: int,
) -> dict[str, Any]:
    gate = asyncio.Event()
    tasks = [
        asyncio.create_task(
            send_one(
                session,
                args.base_url,
                row,
                contexts[index] if contexts is not None else None,
                mode,
                output_tokens,
                args.ignore_eos,
                gate,
            )
        )
        for index, row in enumerate(rows)
    ]
    await asyncio.sleep(0.1)
    started = time.perf_counter()
    gate.set()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    latencies = [float(row["elapsed_seconds"]) for row in results]
    return {
        "mode": mode,
        "batch_elapsed_seconds": elapsed,
        "request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "max": max(latencies),
        },
        "requests": results,
    }


async def recorder_request(
    session: aiohttp.ClientSession, base_url: str, action: str
) -> None:
    async with session.post(f"{base_url}/{action}") as response:
        response.raise_for_status()
        await response.text()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, dataset = load_rows(args.dataset_rows, args.batch_size)
    if args.context_tokens is not None:
        contexts, context_manifest = build_fixed_contexts(
            all_rows=dataset["rows"],
            selected_rows=rows,
            tokenizer_path=args.tokenizer,
            context_tokens=args.context_tokens,
        )
    else:
        contexts = None
        context_manifest = None
    args.control_file.parent.mkdir(parents=True, exist_ok=True)
    args.control_file.write_text("off\n", encoding="utf-8")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.batch_size)
    result: dict[str, Any] = {
        "schema_version": 1,
        "configuration": {
            "base_url": args.base_url,
            "batch_size": args.batch_size,
            "verification_width": 5,
            "output_tokens": args.output_tokens,
            "warmup_output_tokens": args.warmup_output_tokens,
            "temperature": 0,
            "ignore_eos": args.ignore_eos,
            "skip_warmup": args.skip_warmup,
            "record_experts": args.record_experts,
            "dataset_rows": str(args.dataset_rows),
            "dataset_revision": dataset["dataset"]["revision"],
            "selected_submission_indices": list(range(args.batch_size)),
            "context_tokens_per_request": args.context_tokens,
            "aggregate_context_tokens": (
                args.context_tokens * args.batch_size
                if args.context_tokens is not None
                else None
            ),
            "context_manifest": context_manifest,
            "measured_modes": list(args.modes),
        },
        "runs": [],
    }
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(f"{args.base_url}/health") as response:
            response.raise_for_status()

        if not args.skip_warmup:
            result["warmup"] = await run_batch(
                session, args, rows, contexts, "warmup", args.warmup_output_tokens
            )

        for mode in args.modes:
            session_id = f"{mode}-{time.time_ns()}"
            args.control_file.write_text(
                f"record:{session_id}:{mode}\n", encoding="utf-8"
            )
            if args.record_experts:
                await recorder_request(
                    session, args.base_url, "start_expert_distribution_record"
                )
            try:
                run_result = await run_batch(
                    session, args, rows, contexts, mode, args.output_tokens
                )
            finally:
                args.control_file.write_text("off\n", encoding="utf-8")
                if args.record_experts:
                    await recorder_request(
                        session, args.base_url, "stop_expert_distribution_record"
                    )
                    await recorder_request(
                        session, args.base_url, "dump_expert_distribution_record"
                    )
            run_result["component_timing_session"] = session_id
            result["runs"].append(run_result)

    return result


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
