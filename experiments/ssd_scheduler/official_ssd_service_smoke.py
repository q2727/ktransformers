#!/usr/bin/env python3
"""Exercise one persistent official-SSD draft session without a target model."""

from __future__ import annotations

import argparse
import json

from sglang.srt.speculative.ssd_official_client import OfficialSSDDraftClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--draft-length", type=int, default=5)
    parser.add_argument("--fan-out", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    args = parser.parse_args()

    fan_outs = (args.fan_out,) * (args.draft_length + 1)
    client = OfficialSSDDraftClient(f"unix://{args.socket}", timeout=120)
    client.ping()
    # All ids are in Qwen3's vocabulary.  The recovery token need not be the
    # draft argmax because in production it comes from the target model.
    prompt = [100 + (index % 1000) for index in range(args.prompt_tokens)]
    prefix = [*prompt, 2000]
    initial = client.init_draft(
        "smoke-request", prefix, args.draft_length, fan_outs
    )
    first_build = client.build_outcome_cache(
        "smoke-request",
        prefix,
        initial,
        args.draft_length,
        fan_outs,
    )

    # A deliberately unlikely recovery id validates the official cache-miss
    # path and its in-place JIT repair while retaining canonical KV.
    accepted_length = min(2, args.draft_length)
    next_recovery = 3000
    next_prefix_len = len(prefix) + accepted_length + 1
    selected = client.select_outcome(
        first_build,
        "smoke-request",
        [0] * next_prefix_len,
        (accepted_length, next_recovery),
        args.draft_length,
        fan_outs,
    )
    second_prefix = [0] * (next_prefix_len - 1) + [next_recovery]
    second_build = client.build_outcome_cache(
        "smoke-request",
        second_prefix,
        selected,
        args.draft_length,
        fan_outs,
    )
    print(
        json.dumps(
            {
                "initial_tokens": initial.tokens,
                "first_build_ms": first_build.total_ms,
                "selected_cache_hit": selected.cache_hit,
                "selected_tokens": selected.tokens,
                "second_build_ms": second_build.total_ms,
            },
            indent=2,
        )
    )
    client.close()


if __name__ == "__main__":
    main()
