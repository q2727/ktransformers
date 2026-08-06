#!/usr/bin/env python3
"""Compare token outputs from two paper-workload JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> dict[tuple[str, int], list[int]]:
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (str(row["dataset"]), int(row["dataset_index"]))
        if key in records:
            raise ValueError(f"Duplicate request key {key!r} in {path}")
        records[key] = [int(token) for token in row["token_ids"]]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference = load_records(args.reference)
    candidate = load_records(args.candidate)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise SystemExit(f"Request keys differ: missing={missing}, extra={extra}")

    mismatches = []
    total_tokens = 0
    for key in sorted(reference):
        expected = reference[key]
        observed = candidate[key]
        total_tokens += len(expected)
        if expected == observed:
            continue
        first = next(
            (
                index
                for index, (left, right) in enumerate(zip(expected, observed))
                if left != right
            ),
            min(len(expected), len(observed)),
        )
        mismatches.append((key, first, len(expected), len(observed)))

    if mismatches:
        for key, first, expected_len, observed_len in mismatches:
            print(
                f"mismatch key={key} first_token={first} "
                f"reference_len={expected_len} candidate_len={observed_len}"
            )
        raise SystemExit(1)
    print(
        f"All {total_tokens} generated tokens across "
        f"{len(reference)} requests match exactly."
    )


if __name__ == "__main__":
    main()
