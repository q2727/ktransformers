#!/usr/bin/env python3
"""Run same-process static/dynamic/static KT decode-hot comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    artifact_root = repo_root / "experiments/artifacts/specmoe/control-ab"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-path",
        type=Path,
        default=repo_root / "experiments/artifacts/specmoe/decode-hot/control",
    )
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--base-url", default="http://127.0.0.1:30006")
    parser.add_argument("--sequence", default="static,dynamic,static")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args()


def run_workload(
    args: argparse.Namespace,
    output_path: Path,
    *,
    concurrency: int | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).with_name("drive_workload.py")),
        "--base-url",
        args.base_url,
        "--concurrency",
        str(concurrency or args.concurrency),
        "--num-requests",
        str(concurrency or args.concurrency),
        "--prompt-tokens",
        str(prompt_tokens or args.prompt_tokens),
        "--output-tokens",
        str(output_tokens or args.output_tokens),
        "--warmup-requests",
        "0",
        "--timeout",
        str(args.timeout),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Workload failed: {' '.join(command)}")
    return json.loads(output_path.read_text())


def output_hashes(run: dict) -> list[str]:
    return [request["output_sha256"] for request in run["requests"]]


def hash_match_rate(reference: list[str], candidate: list[str]) -> float:
    if len(reference) != len(candidate):
        return 0.0
    return sum(left == right for left, right in zip(reference, candidate)) / len(
        reference
    )


def summarize_phase(runs: list[dict], reference_hashes: list[str]) -> dict:
    throughputs = [run["output_throughput_tokens_per_second"] for run in runs]
    cpu_busy = [
        run.get("system_observation", {}).get("cpu_busy_percent") for run in runs
    ]
    disk_read = [
        run.get("system_observation", {}).get("disk_read_mib_per_second")
        for run in runs
    ]
    return {
        "runs": len(runs),
        "throughput_tokens_per_second": throughputs,
        "throughput_mean": statistics.fmean(throughputs),
        "throughput_median": statistics.median(throughputs),
        "throughput_min": min(throughputs),
        "throughput_max": max(throughputs),
        "cpu_busy_percent": cpu_busy,
        "disk_read_mib_per_second": disk_read,
        "output_match_rate_to_first_phase": [
            hash_match_rate(reference_hashes, output_hashes(run)) for run in runs
        ],
    }


def main() -> None:
    args = parse_args()
    modes = [mode.strip().lower() for mode in args.sequence.split(",")]
    if not modes or any(mode not in ("static", "dynamic") for mode in modes):
        raise ValueError("sequence must contain only static and dynamic")
    if args.warmup_runs < 0 or args.measure_runs < 1:
        raise ValueError("warmup-runs must be non-negative and measure-runs positive")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.control_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "configuration": {
            "sequence": modes,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "concurrency": args.concurrency,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "control_path": str(args.control_path),
        },
        "phases": [],
    }
    reference_hashes = None

    for phase_index, mode in enumerate(modes):
        phase_name = f"{phase_index + 1:02d}_{mode}"
        phase_dir = args.artifact_root / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        args.control_path.write_text(mode + "\n")

        # The transition request is computed with the previous placement. Its
        # post-verify hook performs a complete reset before static measurements.
        if mode == "static":
            run_workload(
                args,
                phase_dir / "transition.json",
                concurrency=1,
                prompt_tokens=8,
                output_tokens=1,
            )

        for index in range(args.warmup_runs):
            run_workload(args, phase_dir / f"warmup_{index + 1:02d}.json")

        measured = []
        for index in range(args.measure_runs):
            run = run_workload(
                args, phase_dir / f"measure_{index + 1:02d}.json"
            )
            measured.append(run)
            print(
                f"{phase_name} run {index + 1}: "
                f"{run['output_throughput_tokens_per_second']:.3f} tok/s, "
                f"cpu={run['system_observation']['cpu_busy_percent']:.1f}%"
            )

        if reference_hashes is None:
            reference_hashes = output_hashes(measured[0])

        result["phases"].append(
            {
                "name": phase_name,
                "mode": mode,
                "summary": summarize_phase(measured, reference_hashes),
            }
        )
        (args.artifact_root / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
