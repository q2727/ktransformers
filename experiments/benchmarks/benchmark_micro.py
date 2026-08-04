#!/usr/bin/env python3
"""Reproducible prefill-TTFT and decode-TPOT benchmarks for KT/SGLang.

The benchmark uses the OpenAI-compatible chat-completions endpoint, but relies on
the server's final usage record for token counts.  Every measured trial can flush
the radix/Mamba cache so repeated prompts do not accidentally become cache hits.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


PREFILL_SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
DECODE_BATCH_SIZES = (1, 2, 4, 6, 8)
SEED_TEXT = (
    "This is deterministic synthetic context for a language-model serving "
    "benchmark. It contains prose, numbers 0123456789, punctuation, and a "
    "short instruction: inspect the context and continue consistently.\n"
)


@dataclass
class RequestResult:
    benchmark: str
    run_id: str
    timestamp_utc: str
    model: str
    nominal_prompt_tokens: int
    prompt_tokens: int
    completion_tokens: int
    batch_size: int
    trial: int
    request_index: int
    ttft_seconds: float
    e2e_seconds: float
    decode_seconds: float | None
    tpot_seconds: float | None
    prefill_tokens_per_second: float
    request_decode_tokens_per_second: float | None
    batch_wall_seconds: float | None = None
    batch_e2e_output_tokens_per_second: float | None = None
    batch_effective_output_tokens_per_second: float | None = None


def parse_json_object(value: str, flag: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{flag} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{flag} must be a JSON object")
    return parsed


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class PromptFactory:
    def __init__(
        self,
        tokenizer_path: str,
        chat_template_kwargs: dict[str, Any],
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        self.chat_template_kwargs = chat_template_kwargs
        self.seed_ids = self.tokenizer.encode(SEED_TEXT, add_special_tokens=False)
        if not self.seed_ids:
            raise RuntimeError("The tokenizer produced no tokens for the seed text")

    def _messages(self, content: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ]

    def _templated_count(self, content: str) -> int:
        encoded = self.tokenizer.apply_chat_template(
            self._messages(content),
            tokenize=True,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        # Transformers 5 returns BatchEncoding here, while Transformers 4
        # returns the input-ID list directly.
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise RuntimeError("Expected one templated conversation")
            token_ids = token_ids[0]
        return len(token_ids)

    def make(self, target_tokens: int) -> tuple[str, int]:
        """Build a deterministic prompt close to the requested templated length."""
        empty_overhead = self._templated_count("")
        content_target = max(target_tokens - empty_overhead, 1)
        repeats, remainder = divmod(content_target, len(self.seed_ids))
        content_ids = self.seed_ids * repeats + self.seed_ids[:remainder]

        # Decoding token IDs is much faster than repeatedly tokenizing large text.
        # Two correction passes compensate for merges at repeated-text boundaries.
        content = self.tokenizer.decode(
            content_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual = self._templated_count(content)
        for _ in range(2):
            delta = target_tokens - actual
            if delta == 0:
                break
            if delta > 0:
                extra_repeats, extra_remainder = divmod(delta, len(self.seed_ids))
                extra_ids = self.seed_ids * extra_repeats + self.seed_ids[:extra_remainder]
                content += self.tokenizer.decode(
                    extra_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                encoded = self.tokenizer.encode(content, add_special_tokens=False)
                keep = max(len(encoded) + delta, 1)
                content = self.tokenizer.decode(
                    encoded[:keep],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            actual = self._templated_count(content)
        return content, actual


def token_delta_present(event: dict[str, Any]) -> bool:
    choices = event.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    for key in ("content", "reasoning_content", "reasoning", "tool_calls"):
        if delta.get(key):
            return True
    return False


async def post_flush(session: aiohttp.ClientSession, flush_url: str) -> Any:
    async with session.post(flush_url) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(
                f"Cache flush failed with HTTP {response.status}: {body[:500]}"
            )
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body
        if isinstance(parsed, dict) and parsed.get("success") is False:
            raise RuntimeError(f"The server refused to flush its cache: {parsed}")
        return parsed


async def stream_one(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, Any],
    barrier: asyncio.Event,
) -> dict[str, Any]:
    await barrier.wait()
    started_at = time.perf_counter()
    first_token_at: float | None = None
    finished_at: float | None = None
    usage: dict[str, Any] = {}

    async with session.post(api_url, json=payload) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(
                f"Request failed with HTTP {response.status}: {body[:1000]}"
            )

        pending = b""
        async for chunk in response.content.iter_any():
            pending += chunk
            while b"\n" in pending:
                raw_line, pending = pending.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                if first_token_at is None and token_delta_present(event):
                    first_token_at = time.perf_counter()
                choices = event.get("choices") or []
                if choices and choices[0].get("finish_reason") is not None:
                    finished_at = time.perf_counter()

        # A final SSE record does not have to end in a newline.
        tail = pending.decode("utf-8", errors="replace").strip()
        if tail.startswith("data:") and tail[5:].strip() not in ("", "[DONE]"):
            event = json.loads(tail[5:].strip())
            if event.get("usage"):
                usage = event["usage"]
            if first_token_at is None and token_delta_present(event):
                first_token_at = time.perf_counter()
            choices = event.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                finished_at = time.perf_counter()

    ended_at = finished_at or time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("No streamed output token was received")
    if not usage:
        raise RuntimeError("The server did not return a final usage record")
    return {
        "started_at": started_at,
        "first_token_at": first_token_at,
        "ended_at": ended_at,
        "usage": usage,
    }


def build_payload(
    args: argparse.Namespace,
    content: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "max_tokens": max_tokens,
    }
    if args.chat_template_kwargs:
        payload["chat_template_kwargs"] = args.chat_template_kwargs
    payload.update(args.extra_body)
    return payload


async def run_group(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    run_id: str,
    benchmark: str,
    content: str,
    nominal_prompt_tokens: int,
    batch_size: int,
    trial: int,
    max_tokens: int,
) -> list[RequestResult]:
    if args.flush_between_trials:
        await post_flush(session, args.flush_url)

    payload = build_payload(args, content, max_tokens)
    barrier = asyncio.Event()
    tasks = [
        asyncio.create_task(stream_one(session, args.api_url, payload, barrier))
        for _ in range(batch_size)
    ]
    group_started_at = time.perf_counter()
    barrier.set()
    raw_results = await asyncio.gather(*tasks)
    group_ended_at = time.perf_counter()

    prompt_counts = {int(item["usage"]["prompt_tokens"]) for item in raw_results}
    if len(prompt_counts) != 1:
        raise RuntimeError(f"Requests in one batch reported different prompts: {prompt_counts}")

    batch_wall = group_ended_at - group_started_at
    total_output_tokens = sum(int(item["usage"]["completion_tokens"]) for item in raw_results)
    earliest_first = min(float(item["first_token_at"]) for item in raw_results)
    latest_end = max(float(item["ended_at"]) for item in raw_results)
    effective_decode_window = max(latest_end - earliest_first, 1e-9)

    records: list[RequestResult] = []
    for request_index, item in enumerate(raw_results):
        usage = item["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        ttft = float(item["first_token_at"] - item["started_at"])
        e2e = float(item["ended_at"] - item["started_at"])
        decode_token_count = max(completion_tokens - 1, 0)
        decode_seconds = max(e2e - ttft, 0.0) if decode_token_count else None
        tpot = decode_seconds / decode_token_count if decode_token_count else None
        decode_tps = decode_token_count / decode_seconds if decode_seconds else None
        records.append(
            RequestResult(
                benchmark=benchmark,
                run_id=run_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                model=args.model,
                nominal_prompt_tokens=nominal_prompt_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                batch_size=batch_size,
                trial=trial,
                request_index=request_index,
                ttft_seconds=ttft,
                e2e_seconds=e2e,
                decode_seconds=decode_seconds,
                tpot_seconds=tpot,
                prefill_tokens_per_second=prompt_tokens / ttft,
                request_decode_tokens_per_second=decode_tps,
                batch_wall_seconds=batch_wall,
                batch_e2e_output_tokens_per_second=total_output_tokens / batch_wall,
                batch_effective_output_tokens_per_second=(
                    total_output_tokens / effective_decode_window
                ),
            )
        )
    return records


def summarize(records: list[RequestResult], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[RequestResult]] = {}
    for record in records:
        key = (record.nominal_prompt_tokens, record.batch_size)
        grouped.setdefault(key, []).append(record)

    points: list[dict[str, Any]] = []
    for (nominal_tokens, batch_size), items in sorted(grouped.items()):
        ttfts = [item.ttft_seconds for item in items]
        tpots = [item.tpot_seconds for item in items if item.tpot_seconds is not None]
        point: dict[str, Any] = {
            "nominal_prompt_tokens": nominal_tokens,
            "actual_prompt_tokens_median": statistics.median(
                item.prompt_tokens for item in items
            ),
            "batch_size": batch_size,
            "request_samples": len(items),
            "ttft_median_seconds": statistics.median(ttfts),
            "ttft_p95_seconds": percentile(ttfts, 0.95),
            "prefill_tokens_per_second_median": statistics.median(
                item.prefill_tokens_per_second for item in items
            ),
            "batch_e2e_output_tokens_per_second_median": statistics.median(
                item.batch_e2e_output_tokens_per_second for item in items
                if item.batch_e2e_output_tokens_per_second is not None
            ),
            "batch_effective_output_tokens_per_second_median": statistics.median(
                item.batch_effective_output_tokens_per_second for item in items
                if item.batch_effective_output_tokens_per_second is not None
            ),
        }
        if tpots:
            point.update(
                {
                    "tpot_median_seconds": statistics.median(tpots),
                    "tpot_p95_seconds": percentile(tpots, 0.95),
                    "request_decode_tokens_per_second_median": statistics.median(
                        item.request_decode_tokens_per_second for item in items
                        if item.request_decode_tokens_per_second is not None
                    ),
                }
            )
        points.append(point)

    return {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "api_url": args.api_url,
        "trials": args.trials,
        "flush_between_trials": args.flush_between_trials,
        "chat_template_kwargs": args.chat_template_kwargs,
        "extra_body": args.extra_body,
        "points": points,
    }


def write_outputs(
    output_dir: Path,
    run_id: str,
    records: list[RequestResult],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{run_id}.jsonl"
    summary_path = output_dir / f"{run_id}.summary.json"
    csv_path = output_dir / f"{run_id}.summary.csv"

    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    points = summary["points"]
    fieldnames = sorted({key for point in points for key in point})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)

    print(json.dumps({"raw": str(raw_path), "summary": str(summary_path), "csv": str(csv_path)}))


async def async_main(args: argparse.Namespace) -> None:
    prompt_factory = PromptFactory(args.tokenizer, args.chat_template_kwargs)
    sizes = args.prefill_sizes if args.benchmark == "prefill" else [args.decode_context]
    prompts: dict[int, str] = {}
    for size in sizes:
        content, local_count = prompt_factory.make(size)
        prompts[size] = content
        print(
            json.dumps(
                {
                    "event": "prompt_prepared",
                    "nominal_prompt_tokens": size,
                    "local_templated_tokens": local_count,
                }
            ),
            flush=True,
        )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.decode_batch_sizes + [1]) + 2)
    records: list[RequestResult] = []
    run_id = f"{args.benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.warmup:
            warmup_content, _ = prompt_factory.make(min(1024, sizes[0]))
            await run_group(
                args,
                session,
                run_id,
                "warmup",
                warmup_content,
                min(1024, sizes[0]),
                1,
                0,
                args.warmup_output_tokens,
            )

        if args.benchmark == "prefill":
            for size in args.prefill_sizes:
                for trial in range(1, args.trials + 1):
                    measured = await run_group(
                        args,
                        session,
                        run_id,
                        "prefill",
                        prompts[size],
                        size,
                        1,
                        trial,
                        args.prefill_output_tokens,
                    )
                    records.extend(measured)
                    print(json.dumps(asdict(measured[0])), flush=True)
        else:
            for batch_size in args.decode_batch_sizes:
                for trial in range(1, args.trials + 1):
                    measured = await run_group(
                        args,
                        session,
                        run_id,
                        "decode",
                        prompts[args.decode_context],
                        args.decode_context,
                        batch_size,
                        trial,
                        args.decode_output_tokens,
                    )
                    records.extend(measured)
                    print(
                        json.dumps(
                            {
                                "event": "batch_complete",
                                "batch_size": batch_size,
                                "trial": trial,
                                "tpot_median_seconds": statistics.median(
                                    item.tpot_seconds for item in measured
                                    if item.tpot_seconds is not None
                                ),
                                "batch_effective_output_tokens_per_second": measured[
                                    0
                                ].batch_effective_output_tokens_per_second,
                            }
                        ),
                        flush=True,
                    )

    summary = summarize(records, args)
    write_outputs(args.output_dir, run_id, records, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("prefill", "decode"))
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--tokenizer", required=True, help="local tokenizer/model path")
    parser.add_argument(
        "--api-url", default="http://127.0.0.1:30005/v1/chat/completions"
    )
    parser.add_argument("--flush-url", default="http://127.0.0.1:30005/flush_cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--prefill-sizes", type=int, nargs="+", default=list(PREFILL_SIZES))
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--decode-context", type=int, default=8192)
    parser.add_argument(
        "--decode-batch-sizes", type=int, nargs="+", default=list(DECODE_BATCH_SIZES)
    )
    parser.add_argument("--decode-output-tokens", type=int, default=512)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument(
        "--no-flush-between-trials",
        dest="flush_between_trials",
        action="store_false",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        default="{}",
        help='for example: {"enable_thinking":false}',
    )
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="extra top-level OpenAI request fields",
    )
    args = parser.parse_args()
    args.chat_template_kwargs = parse_json_object(
        args.chat_template_kwargs_json, "--chat-template-kwargs-json"
    )
    args.extra_body = parse_json_object(args.extra_body_json, "--extra-body-json")

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.prefill_output_tokens < 1 or args.decode_output_tokens < 2:
        parser.error("prefill output must be >=1 and decode output must be >=2")
    if any(value < 1 for value in args.prefill_sizes + args.decode_batch_sizes):
        parser.error("prompt sizes and batch sizes must be positive")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
