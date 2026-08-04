#!/usr/bin/env python3
"""Merge non-overlapping microbenchmark record fragments into one formal run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark_micro import RequestResult, summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--summary-template", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    source_info: list[dict] = []
    for path in args.input:
        fragment = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.extend(fragment)
        source_info.append(
            {"path": str(path.resolve()), "sha256": sha256(path), "records": len(fragment)}
        )

    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["nominal_prompt_tokens"]), int(row["batch_size"]))
        grouped.setdefault(key, []).append(row)
    if len(grouped) != args.expected_points:
        raise SystemExit(f"expected {args.expected_points} points, found {len(grouped)}")
    for key, items in grouped.items():
        if len(items) != args.trials * key[1]:
            raise SystemExit(
                f"point {key} has {len(items)} request records; "
                f"expected {args.trials * key[1]}"
            )
        items.sort(key=lambda row: (row["timestamp_utc"], int(row["request_index"])))
        for index, row in enumerate(items):
            row["trial"] = index // key[1] + 1

    rows = [row for key in sorted(grouped) for row in grouped[key]]
    template = json.loads(args.summary_template.read_text(encoding="utf-8"))
    namespace = SimpleNamespace(
        benchmark=template["benchmark"],
        model=template["model"],
        tokenizer=template["tokenizer"],
        api_url=template["api_url"],
        trials=args.trials,
        flush_between_trials=template["flush_between_trials"],
        chat_template_kwargs=template.get("chat_template_kwargs", {}),
        extra_body=template.get("extra_body", {}),
    )
    rebuilt = summarize([RequestResult(**row) for row in rows], namespace)
    rebuilt["merged_fragments"] = {
        "reason": args.reason,
        "sources": source_info,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_suffix(".jsonl")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    csv_path = args.output_prefix.with_suffix(".summary.csv")
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for point in rebuilt["points"] for key in point})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rebuilt["points"])
    print(json.dumps({"raw": str(raw_path), "summary": str(summary_path), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
