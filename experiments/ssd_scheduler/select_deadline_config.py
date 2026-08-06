#!/usr/bin/env python3
"""Select an SSD resource/quality configuration from measured primitives.

The selector deliberately does not optimize measured end-to-end throughput.
It admits only configurations whose outcome build meets the target deadline
with a guard interval and whose target interference stays below a limit.  It
then ranks admitted configurations with a simple SSD round-cost model:

    max(target_verify, outcome_build)
      + P(hit) * select_hit
      + P(miss) * select_miss_jit

The expected useful output of one verification round is 1 + mean accepted
draft tokens.  A fixed host/runtime tax can be supplied to account for
scheduler and frontend work that is common to all configurations.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


LABEL_RE = re.compile(
    r"target(?P<target>\d+)_draft(?P<draft>\d+)_f(?P<fanout>\d+)"
)


def parse_root(value: str) -> tuple[str, Path]:
    family, separator, raw_path = value.partition("=")
    if not separator or not family or not raw_path:
        raise argparse.ArgumentTypeError("--root must be FAMILY=PATH")
    return family, Path(raw_path)


def parse_k_value(value: str) -> tuple[int, float]:
    raw_k, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("value must be K=NUMBER")
    try:
        draft_length = int(raw_k)
        number = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be K=NUMBER") from exc
    if draft_length <= 0 or number <= 0:
        raise argparse.ArgumentTypeError("K and NUMBER must be positive")
    return draft_length, number


def percentile_metric(overall: dict, name: str, percentile: str) -> float:
    value = overall.get(name, {}).get(percentile)
    if value is None:
        raise ValueError(f"Missing {name}.{percentile}")
    return float(value)


def load_point(family: str, analysis_path: Path) -> dict | None:
    analysis = json.loads(analysis_path.read_text())
    overall = analysis["overall"]
    build_p50 = overall.get("outcome_cache", {}).get("median_ms")
    build_p95 = overall.get("outcome_cache", {}).get("p95_ms")
    if build_p50 is None or build_p95 is None:
        # A sequential-SD run has no outcome build and is not selectable here.
        return None

    label = analysis["label"]
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Cannot parse SSD configuration label {label!r}")
    summary = json.loads(
        (analysis_path.parent / "records.jsonl.summary.json").read_text()
    )
    rows = analysis.get("rows", [])
    if not rows or not rows[0].get("fan_outs"):
        raise ValueError(f"Cannot infer draft length from {analysis_path}")
    draft_length = len(rows[0]["fan_outs"]) - 1

    miss_rate = float(overall["cache_miss_rate"])
    hit_p50 = percentile_metric(overall, "select_hit", "median_ms")
    miss_p50 = percentile_metric(
        overall, "select_miss_jit", "median_ms"
    )
    verify_ms = float(overall["target_verify_ms_per_round"])
    acceptance = float(overall["accepted_draft_per_round"])
    rounds = int(overall["rounds"])
    throughput = float(summary["throughput_tok_s"])
    observed_round_ms = (
        1000.0 * float(summary["total_tokens"]) / throughput / rounds
    )
    selection_ms = (1.0 - miss_rate) * hit_p50 + miss_rate * miss_p50
    core_round_ms = max(verify_ms, float(build_p50)) + selection_ms

    return {
        "name": f"{family}:{label}",
        "family": family,
        "label": label,
        "target_mps_pct": int(match.group("target")),
        "draft_mps_pct": int(match.group("draft")),
        "fanout": int(match.group("fanout")),
        "draft_length": draft_length,
        "verify_ms": verify_ms,
        "build_p50_ms": float(build_p50),
        "build_p95_ms": float(build_p95),
        "miss_rate": miss_rate,
        "hit_p50_ms": hit_p50,
        "miss_jit_p50_ms": miss_p50,
        "selection_expected_ms": selection_ms,
        "acceptance": acceptance,
        "core_round_ms": core_round_ms,
        "observed_round_ms": observed_round_ms,
        "observed_throughput_tok_s": throughput,
    }


def render(points: list[dict]) -> str:
    rows = [
        "eligible\tname\tK\tpred_tok_s\tobs_tok_s\terror_pct\tverify_ms\t"
        "build_p95_ms\tdeadline_slack_ms\tinterference_pct\tmiss_pct\t"
        "acceptance\treason"
    ]
    for point in points:
        rows.append(
            "\t".join(
                (
                    "yes" if point["eligible"] else "no",
                    point["name"],
                    str(point["draft_length"]),
                    f'{point["predicted_throughput_tok_s"]:.3f}',
                    f'{point["observed_throughput_tok_s"]:.3f}',
                    f'{point["prediction_error_pct"]:+.2f}',
                    f'{point["verify_ms"]:.3f}',
                    f'{point["build_p95_ms"]:.3f}',
                    f'{point["deadline_slack_ms"]:+.3f}',
                    f'{point["interference_pct"]:+.2f}',
                    f'{100.0 * point["miss_rate"]:.2f}',
                    f'{point["acceptance"]:.3f}',
                    point["reason"],
                )
            )
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=parse_root,
        action="append",
        required=True,
        help="Configuration family and artifact root as FAMILY=PATH",
    )
    parser.add_argument(
        "--verify-baseline-ms",
        type=float,
        help="One uncontended verify baseline for every K (legacy shortcut)",
    )
    parser.add_argument(
        "--verify-baseline",
        type=parse_k_value,
        action="append",
        default=[],
        metavar="K=MS",
        help="Uncontended target verify time for one draft length",
    )
    parser.add_argument("--guard-ms", type=float, default=2.0)
    parser.add_argument("--max-target-interference", type=float, default=0.03)
    parser.add_argument("--host-tax-ms", type=float, default=7.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    points = []
    seen = set()
    for family, root in args.root:
        for analysis_path in sorted(root.glob("*/analysis.json")):
            if not (analysis_path.parent / "complete").exists():
                continue
            point = load_point(family, analysis_path)
            if point is None or point["name"] in seen:
                continue
            seen.add(point["name"])
            points.append(point)
    if not points:
        raise RuntimeError("No completed SSD outcome-cache configurations found.")

    if args.verify_baseline_ms is not None and args.verify_baseline:
        parser.error("Use either --verify-baseline-ms or --verify-baseline.")
    draft_lengths = sorted({point["draft_length"] for point in points})
    verify_baselines = {
        draft_length: min(
            point["verify_ms"]
            for point in points
            if point["draft_length"] == draft_length
        )
        for draft_length in draft_lengths
    }
    if args.verify_baseline_ms is not None:
        verify_baselines = {
            draft_length: args.verify_baseline_ms
            for draft_length in draft_lengths
        }
    for draft_length, verify_ms in args.verify_baseline:
        verify_baselines[draft_length] = verify_ms
    for point in points:
        verify_baseline = verify_baselines[point["draft_length"]]
        point["deadline_slack_ms"] = (
            point["verify_ms"] - args.guard_ms - point["build_p95_ms"]
        )
        point["interference"] = point["verify_ms"] / verify_baseline - 1.0
        point["interference_pct"] = 100.0 * point["interference"]
        violations = []
        if point["deadline_slack_ms"] < 0:
            violations.append("deadline")
        if point["interference"] > args.max_target_interference:
            violations.append("interference")
        point["eligible"] = not violations
        point["reason"] = "admit" if not violations else "+".join(violations)
        predicted_round_ms = point["core_round_ms"] + args.host_tax_ms
        point["predicted_round_ms"] = predicted_round_ms
        point["predicted_throughput_tok_s"] = (
            1000.0 * (1.0 + point["acceptance"]) / predicted_round_ms
        )
        point["prediction_error_pct"] = 100.0 * (
            point["predicted_throughput_tok_s"]
            / point["observed_throughput_tok_s"]
            - 1.0
        )

    ranked = sorted(
        points,
        key=lambda point: (
            point["eligible"], point["predicted_throughput_tok_s"]
        ),
        reverse=True,
    )
    admitted = [point for point in ranked if point["eligible"]]
    if not admitted:
        raise RuntimeError("No configuration satisfies the deadline constraints.")
    selected = admitted[0]
    observed_oracle = max(points, key=lambda point: point["observed_throughput_tok_s"])
    admitted_oracle = max(
        admitted, key=lambda point: point["observed_throughput_tok_s"]
    )
    absolute_errors = [abs(point["prediction_error_pct"]) for point in points]
    result = {
        "policy": {
            "verify_baseline_ms_by_k": verify_baselines,
            "guard_ms": args.guard_ms,
            "max_target_interference": args.max_target_interference,
            "host_tax_ms": args.host_tax_ms,
        },
        "selected": selected,
        "observed_oracle": observed_oracle,
        "admitted_oracle": admitted_oracle,
        "selection_regret_pct": 100.0
        * (
            admitted_oracle["observed_throughput_tok_s"]
            / selected["observed_throughput_tok_s"]
            - 1.0
        ),
        "prediction_absolute_error_pct": {
            "median": statistics.median(absolute_errors),
            "max": max(absolute_errors),
        },
        "points": ranked,
    }

    print(render(ranked[: args.top]))
    print(
        f'selected={selected["name"]} '
        f'pred={selected["predicted_throughput_tok_s"]:.3f} '
        f'obs={selected["observed_throughput_tok_s"]:.3f} tok/s '
        f'admitted_oracle={admitted_oracle["name"]} '
        f'regret={result["selection_regret_pct"]:.2f}% '
        f'median_abs_prediction_error='
        f'{result["prediction_absolute_error_pct"]["median"]:.2f}%'
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
