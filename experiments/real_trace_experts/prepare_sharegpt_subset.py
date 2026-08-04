#!/usr/bin/env python3
"""Select eight native long ShareGPT conversation prefixes without padding."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import ijson
from transformers import AutoTokenizer


SOURCE_REPOSITORY = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SOURCE_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
SOURCE_FILE = "HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=7680)
    parser.add_argument("--max-tokens", type=int, default=8704)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--proxy-limit", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.count <= 64:
        parser.error("--count must be in [1, 64]")
    if not 1 <= args.min_tokens <= args.target_tokens <= args.max_tokens:
        parser.error("require min_tokens <= target_tokens <= max_tokens")
    if args.proxy_limit < args.count:
        parser.error("--proxy-limit must be at least --count")
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_messages(conversation: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    messages = []
    expected = "human"
    for item in conversation:
        source_role = str(item.get("from", ""))
        value = str(item.get("value", ""))
        if source_role not in ("human", "gpt") or source_role != expected or not value.strip():
            return None
        messages.append(
            {"role": "user" if source_role == "human" else "assistant", "content": value}
        )
        expected = "gpt" if expected == "human" else "human"
    return messages


def proxy_for_trace(
    *,
    ordinal: int,
    item: dict[str, Any],
    min_tokens: int,
    max_tokens: int,
    target_tokens: int,
    seed: int,
) -> dict[str, Any] | None:
    trace_id = str(item.get("id", ""))
    conversation = item.get("conversations")
    if not trace_id or not isinstance(conversation, list):
        return None
    messages = normalized_messages(conversation)
    if not messages:
        return None

    best = None
    for turn_index, message in enumerate(messages):
        if message["role"] != "user":
            continue
        prefix = messages[: turn_index + 1]
        character_count = sum(len(value["content"]) for value in prefix)
        # Retain a broad range around the usual English/code character-to-token
        # ratio. Exact token filtering happens only after the streaming scan.
        if character_count < int(min_tokens * 1.4) or character_count > max_tokens * 8:
            continue
        tie_break = sha256_bytes(f"{seed}:{trace_id}:{turn_index}".encode("utf-8"))
        candidate = {
            "source_ordinal": ordinal,
            "trace_id": trace_id,
            "final_turn_index": turn_index,
            "message_count": len(prefix),
            "character_count": character_count,
            "character_proxy_distance": abs(character_count - target_tokens * 4),
            "selection_tie_break": tie_break,
            "messages": prefix,
        }
        key = (candidate["character_proxy_distance"], tie_break)
        if best is None or key < best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


def materialize_candidate(
    proxy: dict[str, Any], tokenizer: Any, min_tokens: int, max_tokens: int, target_tokens: int
) -> dict[str, Any] | None:
    messages = proxy["messages"]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = (
        encoded["input_ids"]
        if hasattr(encoded, "keys") and "input_ids" in encoded
        else encoded
    )
    token_count = len(input_ids)
    if not min_tokens <= token_count <= max_tokens:
        return None
    messages_blob = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    ids_blob = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
    result = {key: value for key, value in proxy.items() if key != "messages"}
    result.update(
        {
            "input_tokens": token_count,
            "distance_from_target_tokens": abs(token_count - target_tokens),
            "messages_sha256": sha256_bytes(messages_blob),
            "input_ids_sha256": sha256_bytes(ids_blob),
            "input_ids": input_ids,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )

    proxy_heap: list[tuple[int, str, dict[str, Any]]] = []
    scanned = 0
    with args.source.open("rb") as handle:
        for ordinal, item in enumerate(ijson.items(handle, "item")):
            scanned += 1
            proxy = proxy_for_trace(
                ordinal=ordinal,
                item=item,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                target_tokens=args.target_tokens,
                seed=args.seed,
            )
            if proxy is not None:
                entry = (
                    -int(proxy["character_proxy_distance"]),
                    str(proxy["selection_tie_break"]),
                    proxy,
                )
                heapq.heappush(proxy_heap, entry)
                if len(proxy_heap) > args.proxy_limit:
                    heapq.heappop(proxy_heap)
            if scanned % 5000 == 0:
                print(
                    f"scanned={scanned} retained_proxies={len(proxy_heap)}",
                    file=sys.stderr,
                    flush=True,
                )

    proxies = [entry[2] for entry in proxy_heap]
    proxies.sort(
        key=lambda item: (item["character_proxy_distance"], item["selection_tie_break"])
    )
    candidates = []
    for index, proxy in enumerate(proxies):
        candidate = materialize_candidate(
            proxy, tokenizer, args.min_tokens, args.max_tokens, args.target_tokens
        )
        if candidate is not None:
            candidates.append(candidate)
        if (index + 1) % 100 == 0:
            print(
                f"tokenized_proxies={index + 1} exact_candidates={len(candidates)}",
                file=sys.stderr,
                flush=True,
            )

    if len(candidates) < args.count:
        raise RuntimeError(
            f"only found {len(candidates)} native prefixes in the requested token range"
        )
    candidates.sort(
        key=lambda item: (
            item["distance_from_target_tokens"], item["selection_tie_break"]
        )
    )
    unique_candidates = []
    seen_input_hashes = set()
    for candidate in candidates:
        digest = candidate["input_ids_sha256"]
        if digest in seen_input_hashes:
            continue
        seen_input_hashes.add(digest)
        unique_candidates.append(candidate)
    if len(unique_candidates) < args.count:
        raise RuntimeError(
            f"only found {len(unique_candidates)} unique native prefixes in range"
        )
    selected = unique_candidates[: args.count]
    random.Random(args.seed).shuffle(selected)

    rows = []
    for submission_index, item in enumerate(selected):
        trace_id = str(item["trace_id"])
        rows.append(
            {
                "submission_index": submission_index,
                "dataset_row_index": int(item["source_ordinal"]),
                "question_id": trace_id,
                "category": "sharegpt_real_conversation",
                "anonymized_request_id": "sharegpt-"
                + sha256_bytes(f"{args.seed}:{trace_id}".encode("utf-8"))[:20],
                "prompt_sha256": item["messages_sha256"],
                "input_ids_sha256": item["input_ids_sha256"],
                "input_tokens": int(item["input_tokens"]),
                "source_ordinal": int(item["source_ordinal"]),
                "source_trace_id": trace_id,
                "final_turn_index": int(item["final_turn_index"]),
                "message_count": int(item["message_count"]),
                "character_count": int(item["character_count"]),
                "input_ids": [int(value) for value in item["input_ids"]],
            }
        )

    input_hashes = [row["input_ids_sha256"] for row in rows]
    if len(input_hashes) != len(set(input_hashes)):
        raise RuntimeError("selected traces do not have unique token sequences")
    payload = {
        "schema_version": 1,
        "dataset": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "file": SOURCE_FILE,
            "source_path": str(args.source),
            "source_sha256": sha256_file(args.source),
            "source_records_scanned": scanned,
            "character_proxies_tokenized": len(proxies),
            "candidate_prefixes_in_range": len(candidates),
            "unique_candidate_token_sequences": len(unique_candidates),
            "license": "apache-2.0",
        },
        "selection": {
            "seed": args.seed,
            "count": args.count,
            "target_tokens": args.target_tokens,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "rule": (
                "For every strictly alternating ShareGPT conversation, consider each "
                "native prefix ending in a user turn. Keep unmodified Qwen chat-template "
                "inputs in the token window, choose the prefix nearest the target per "
                "conversation. Retain a fixed number of globally nearest character-count "
                "proxies, tokenize those exactly, then take the nearest traces with a "
                "seeded hash tie-break and seeded output "
                "order. No padding, concatenation, truncation, "
                "or generated filler is used."
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scanned": scanned,
                "candidates": len(candidates),
                "selected": [
                    {
                        "request_id": row["anonymized_request_id"],
                        "input_tokens": row["input_tokens"],
                        "messages": row["message_count"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
