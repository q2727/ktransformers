#!/usr/bin/env python3
"""Audit every required artifact in the formal serving model matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = (
    ("qwen3-coder-next-fp8", "Qwen3-Coder-Next-FP8"),
    ("qwen3.5-122b-a10b", "Qwen3.5-122B-A10B BF16"),
    ("minimax-m2.5", "MiniMax-M2.5 FP8"),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash MXFP4"),
)
PREFILL_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
DECODE_BATCHES = [1, 2, 4, 6, 8]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_micro(repo: Path, slug: str, benchmark: str) -> dict[str, Any]:
    directory = repo / "experiments/artifacts/micro" / slug / benchmark
    selection_path = directory / "formal_selection.json"
    plot_path = directory / ("prefill_ttft.svg" if benchmark == "prefill" else "decode_tpot.svg")
    result: dict[str, Any] = {
        "selection": str(selection_path),
        "plot": str(plot_path),
        "complete": False,
        "errors": [],
    }
    if not selection_path.is_file():
        result["errors"].append("missing formal_selection.json")
        return result
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary_path = Path(selection["summary_path"])
    if not summary_path.is_file():
        result["errors"].append("selected summary is missing")
        return result
    if sha256(summary_path) != selection["summary_sha256"]:
        result["errors"].append("selected summary SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    points = summary.get("points", [])
    if benchmark == "prefill":
        actual = [int(point["nominal_prompt_tokens"]) for point in points]
        if actual != PREFILL_SIZES:
            result["errors"].append(f"prefill points differ: {actual}")
    else:
        actual = [int(point["batch_size"]) for point in points]
        if actual != DECODE_BATCHES:
            result["errors"].append(f"decode batch points differ: {actual}")
        if any(int(point["nominal_prompt_tokens"]) != 8192 for point in points):
            result["errors"].append("decode context is not uniformly 8192 tokens")
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        result["errors"].append("plot is missing or empty")
    result["summary"] = str(summary_path)
    result["trials"] = summary.get("trials")
    result["complete"] = not result["errors"]
    return result


def check_mooncake(repo: Path, slug: str) -> dict[str, Any]:
    path = repo / "experiments/artifacts/mooncake-toolagent/results" / slug / "formal_selection.json"
    result: dict[str, Any] = {"selection": str(path), "complete": False, "errors": []}
    if not path.is_file():
        result["errors"].append("missing formal_selection.json")
        return result
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("request_budget") != 128:
        result["errors"].append("request budget is not 128")
    if selection.get("closed_loop_concurrency") != 8:
        result["errors"].append("closed-loop concurrency is not 8")
    runs = [selection.get("closed_loop", {})] + list(selection.get("poisson", []))
    if len(selection.get("poisson", [])) != 2:
        result["errors"].append("expected two Poisson runs")
    for index, run in enumerate(runs):
        label = "closed_loop" if index == 0 else f"poisson_{index}"
        records = Path(run.get("records", ""))
        summary = Path(run.get("summary", ""))
        if not records.is_file() or not summary.is_file():
            result["errors"].append(f"{label} selected files are missing")
            continue
        count = sum(1 for line in records.open(encoding="utf-8") if line.strip())
        if count != 128 or run.get("record_count") != 128:
            result["errors"].append(f"{label} does not contain 128 records")
        if sha256(records) != run.get("records_sha256"):
            result["errors"].append(f"{label} records SHA-256 mismatch")
        if sha256(summary) != run.get("summary_sha256"):
            result["errors"].append(f"{label} summary SHA-256 mismatch")
    result["complete"] = not result["errors"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()

    models = []
    for slug, name in MODELS:
        checks = {
            "prefill_ttft": check_micro(repo, slug, "prefill"),
            "decode_tpot": check_micro(repo, slug, "decode"),
            "mooncake": check_mooncake(repo, slug),
        }
        complete = all(check["complete"] for check in checks.values())
        models.append({"slug": slug, "name": name, "complete": complete, "checks": checks})

    payload = {
        "schema_version": 1,
        "excluded_model": "Qwen3.5-35B-A3B",
        "models": models,
        "complete": all(model["complete"] for model in models),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for model in models:
        states = ", ".join(
            f"{name}={'ok' if check['complete'] else 'missing'}"
            for name, check in model["checks"].items()
        )
        print(f"{model['name']}: {states}")
    if not payload["complete"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
