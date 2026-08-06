#!/usr/bin/env python3
"""Materialize the first N prompts used by the official SSD benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from datasets import load_dataset


DATASETS = (
    ("humaneval", "openai/openai_humaneval", None, "test"),
    ("alpaca", "tatsu-lab/alpaca", None, "train"),
    ("gsm8k", "openai/gsm8k", "main", "train"),
    ("ultrafeedback", "openbmb/UltraFeedback", None, "train"),
)


def prompt_text(name: str, row: dict) -> str:
    if name == "humaneval":
        return row["prompt"]
    if name == "alpaca":
        text = row["instruction"]
        if row.get("input", "").strip():
            text = f"{text}\n\n{row['input']}"
        return text
    if name == "gsm8k":
        return row["question"]
    if name == "ultrafeedback":
        return row["instruction"]
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    args = parser.parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    manifest = {"num_samples_per_dataset": args.num_samples, "datasets": []}

    for name, repo_id, config, split in DATASETS:
        print(f"Loading {repo_id} ({split})", flush=True)
        dataset = load_dataset(repo_id, config, split=split, streaming=True)
        rows = []
        for index, row in enumerate(itertools.islice(dataset, args.num_samples)):
            rows.append(
                {
                    "dataset": name,
                    "index": index,
                    "text": prompt_text(name, row),
                }
            )
        if len(rows) != args.num_samples:
            raise RuntimeError(
                f"{name} yielded {len(rows)} rows, expected {args.num_samples}"
            )

        path = args.output_dir / f"{name}.jsonl"
        rendered = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        )
        path.write_text(rendered)
        digest = hashlib.sha256(rendered.encode()).hexdigest()
        manifest["datasets"].append(
            {
                "name": name,
                "repo_id": repo_id,
                "config": config,
                "split": split,
                "count": len(rows),
                "sha256": digest,
            }
        )
        combined.extend(rows)
        print(f"Wrote {len(rows)} prompts to {path}", flush=True)

    combined_path = args.output_dir / "all.jsonl"
    combined_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined)
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
