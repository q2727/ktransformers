#!/usr/bin/env python3
"""Create and validate a manifest for selected formal Mooncake results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_run(directory: Path) -> dict:
    directory = directory.resolve()
    derived_records = sorted(directory.glob("profile_export.first*.jsonl"))
    if derived_records:
        records = derived_records[0]
        summaries = sorted(directory.glob("derived_first*_summary.json"))
        kind = "derived_first_n"
    else:
        records = directory / "profile_export.jsonl"
        summaries = [directory / "profile_export_aiperf.json"]
        kind = "direct_aiperf"
    if not records.is_file() or not summaries or not summaries[0].is_file():
        raise SystemExit(f"incomplete Mooncake run: {directory}")
    count = sum(1 for line in records.open(encoding="utf-8") if line.strip())
    if count != 128:
        raise SystemExit(f"formal run must contain 128 records, found {count}: {records}")
    summary = summaries[0]
    return {
        "kind": kind,
        "directory": str(directory),
        "records": str(records),
        "records_sha256": sha256(records),
        "record_count": count,
        "summary": str(summary),
        "summary_sha256": sha256(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--closed-loop", type=Path, required=True)
    parser.add_argument(
        "--poisson",
        action="append",
        required=True,
        metavar="ACTUAL_RATE:DIRECTORY",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()

    poisson = []
    for spec in args.poisson:
        rate_text, directory_text = spec.split(":", 1)
        run = inspect_run(Path(directory_text))
        run["actual_request_rate_per_second"] = float(rate_text)
        poisson.append(run)
    if len(poisson) != 2:
        raise SystemExit("exactly two --poisson selections are required")

    manifest = {
        "schema_version": 1,
        "model": args.model,
        "request_budget": 128,
        "closed_loop_concurrency": 8,
        "closed_loop": inspect_run(args.closed_loop),
        "poisson": poisson,
        "notes": args.note,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
