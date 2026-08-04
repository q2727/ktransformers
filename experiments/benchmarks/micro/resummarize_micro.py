#!/usr/bin/env python3
"""Replace documented contaminated samples and rebuild microbench summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark_micro import RequestResult, summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary-template", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="PROMPT:BATCH:TRIAL:REPLACEMENT_JSONL",
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.raw)
    repairs: list[dict] = []
    for spec in args.replace:
        prompt_text, batch_text, trial_text, replacement_text = spec.split(":", 3)
        prompt, batch, trial = int(prompt_text), int(batch_text), int(trial_text)
        matches = [
            row
            for row in read_jsonl(Path(replacement_text))
            if int(row["nominal_prompt_tokens"]) == prompt
            and int(row["batch_size"]) == batch
        ]
        if len(matches) != 1:
            raise SystemExit(f"Expected one replacement record for {spec}, got {len(matches)}")
        indices = [
            index
            for index, row in enumerate(rows)
            if int(row["nominal_prompt_tokens"]) == prompt
            and int(row["batch_size"]) == batch
            and int(row["trial"]) == trial
        ]
        if len(indices) != 1:
            raise SystemExit(f"Expected one base record for {spec}, got {len(indices)}")
        replacement = matches[0]
        replacement["trial"] = trial
        replacement["run_id"] = rows[indices[0]]["run_id"]
        repairs.append(
            {
                "nominal_prompt_tokens": prompt,
                "batch_size": batch,
                "trial": trial,
                "original_ttft_seconds": rows[indices[0]]["ttft_seconds"],
                "replacement_ttft_seconds": replacement["ttft_seconds"],
                "replacement_source": replacement_text,
                "reason": args.reason,
            }
        )
        rows[indices[0]] = replacement

    template = json.loads(args.summary_template.read_text(encoding="utf-8"))
    namespace = SimpleNamespace(
        benchmark=template["benchmark"],
        model=template["model"],
        tokenizer=template["tokenizer"],
        api_url=template["api_url"],
        trials=template["trials"],
        flush_between_trials=template["flush_between_trials"],
        chat_template_kwargs=template.get("chat_template_kwargs", {}),
        extra_body=template.get("extra_body", {}),
    )
    records = [RequestResult(**row) for row in rows]
    rebuilt = summarize(records, namespace)
    rebuilt["data_repairs"] = repairs

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output_prefix.with_suffix(".jsonl")
    summary_output = args.output_prefix.with_suffix(".summary.json")
    csv_output = args.output_prefix.with_suffix(".summary.csv")
    raw_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_output.write_text(
        json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for point in rebuilt["points"] for key in point})
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rebuilt["points"])
    print(json.dumps({"raw": str(raw_output), "summary": str(summary_output), "csv": str(csv_output)}))


if __name__ == "__main__":
    main()
