#!/usr/bin/env python3
"""Recover completed benchmark records from a terminated console log."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("prefill", "decode"), required=True)
    parser.add_argument(
        "--completed-before",
        help="Keep only records whose timestamp_utc is before this ISO-8601 time",
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    cutoff = parse_time(args.completed_before) if args.completed_before else None
    rows: list[dict] = []
    for line in args.console.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("benchmark") != args.benchmark:
            continue
        if cutoff is not None and parse_time(row["timestamp_utc"]) >= cutoff:
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("no matching completed benchmark records found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_console": str(args.console.resolve()),
        "source_console_sha256": sha256(args.console),
        "benchmark": args.benchmark,
        "completed_before": args.completed_before,
        "record_count": len(rows),
        "reason": args.reason,
        "output": args.output.name,
        "output_sha256": sha256(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
