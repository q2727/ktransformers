#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import runpy
import time
from pathlib import Path

import aiohttp


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCRIPT = REPO_ROOT / "archive/ktransformers/tests/test_speed.py"


async def benchmark(args: argparse.Namespace) -> None:
    legacy = runpy.run_path(str(LEGACY_SCRIPT))
    base_prompt = legacy["ktansformer_prompt1024"]
    prompt = base_prompt * (args.prompt_lens // 1024)

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "max_tokens": args.max_tokens,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    start = time.perf_counter()
    first_token_at: float | None = None
    finished_at: float | None = None
    usage: dict[str, int] = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.api_url, json=payload) as response:
            response.raise_for_status()
            pending = b""
            async for chunk in response.content.iter_any():
                pending += chunk
                while b"\n" in pending:
                    raw_line, pending = pending.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if not body or body == "[DONE]":
                        continue
                    event = json.loads(body)
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") and first_token_at is None:
                        first_token_at = time.perf_counter()
                    if choices[0].get("finish_reason") is not None:
                        finished_at = time.perf_counter()

    end = finished_at or time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("No streamed output token was received")
    if not usage:
        raise RuntimeError("Server did not return final token usage")

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    ttft = first_token_at - start
    e2e = end - start
    decode_time = max(end - first_token_at, 1e-9)
    decode_tokens = max(completion_tokens - 1, 0)

    print(
        json.dumps(
            {
                "nominal_prompt_tokens": args.prompt_lens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_seconds": ttft,
                "e2e_seconds": e2e,
                "prefill_tokens_per_second": prompt_tokens / ttft,
                "decode_tokens_per_second": decode_tokens / decode_time,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-Coder-Next")
    parser.add_argument(
        "--api-url", default="http://127.0.0.1:30005/v1/chat/completions"
    )
    parser.add_argument(
        "--prompt-lens", type=int, choices=(1024, 2048, 4096, 8192, 16384)
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
