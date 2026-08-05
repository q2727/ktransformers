#!/usr/bin/env python3
"""Compare deterministic token outputs from two benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> dict[tuple[int, int], list[int]]:
    data = json.loads(path.read_text())
    return {
        (record["prompt_index"], record["repetition"]): record["token_ids"]
        for record in data["records"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference = load_records(args.reference)
    candidate = load_records(args.candidate)
    if reference.keys() != candidate.keys():
        raise SystemExit("The runs do not contain the same prompt/repetition keys.")

    mismatches = []
    total = 0
    for key in sorted(reference):
        ref_ids = reference[key]
        candidate_ids = candidate[key]
        total += len(ref_ids)
        if ref_ids != candidate_ids:
            first = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(ref_ids, candidate_ids)
                    )
                    if left != right
                ),
                min(len(ref_ids), len(candidate_ids)),
            )
            mismatches.append((key, first, len(ref_ids), len(candidate_ids)))

    if mismatches:
        for key, first, ref_len, candidate_len in mismatches:
            print(
                f"mismatch key={key} first_token={first} "
                f"reference_len={ref_len} candidate_len={candidate_len}"
            )
        raise SystemExit(1)
    print(f"All {total} generated tokens match exactly.")


if __name__ == "__main__":
    main()
