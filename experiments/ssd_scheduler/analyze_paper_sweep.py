#!/usr/bin/env python3
"""Aggregate SSD cache and draft timing metrics from one target-server log."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


KEY_VALUE_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
HIST_RE = re.compile(
    r"SSD request=(\S+) fan_outs=(\([^)]*\)) outcome_hist=(\[[^]]*\]) "
    r"cacheable_hist=(\[[^]]*\]) recovery_rank_hist=(\{[^}]*\})"
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timing_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(rows: list[dict]) -> dict:
    hits = sum(row["hits"] for row in rows)
    misses = sum(row["misses"] for row in rows)
    rounds = sum(row["rounds"] for row in rows)
    accepted = sum(row["accepted_draft"] for row in rows)
    outcome_cache_ms = [
        value for row in rows for value in row["timings"]["outcome_cache"]
    ]
    jit_ms = [value for row in rows for value in row["timings"]["jit"]]
    initial_ms = [value for row in rows for value in row["timings"]["initial"]]
    select_hit_ms = [
        value for row in rows for value in row.get("draft_select_hit_ms", [])
    ]
    select_miss_ms = [
        value for row in rows for value in row.get("draft_select_miss_ms", [])
    ]
    cacheable = sum(sum(row.get("cacheable_hist", [])) for row in rows)
    histogram_rounds = sum(sum(row.get("outcome_hist", [])) for row in rows)
    return {
        "requests": len(rows),
        "rounds": rounds,
        "hits": hits,
        "misses": misses,
        "cache_miss_rate": ratio(misses, hits + misses),
        "cache_hit_rate": ratio(hits, hits + misses),
        "cacheable_outcome_rate": ratio(cacheable, histogram_rounds),
        "accepted_draft_tokens": accepted,
        "accepted_draft_per_round": ratio(accepted, rounds),
        "target_verify_ms_per_round": ratio(
            sum(row["target_verify_ms"] for row in rows), rounds
        ),
        "draft_wait_ms_per_round": ratio(
            sum(row["draft_wait_ms"] for row in rows), rounds
        ),
        "outcome_select_ms_per_round": ratio(
            sum(row["outcome_select_ms"] for row in rows), rounds
        ),
        "client_latency_mean_s": statistics.fmean(
            row["latency_s"] for row in rows
        ),
        "outcome_cache": timing_summary(outcome_cache_ms),
        "jit": timing_summary(jit_ms),
        "initial": timing_summary(initial_ms),
        "select_hit": timing_summary(select_hit_ms),
        "select_miss_jit": timing_summary(select_miss_ms),
        "outcome_cache_components": {
            name: timing_summary(
                [
                    value
                    for row in rows
                    for value in row["timing_components"].get(name, [])
                ]
            )
            for name in (
                "glue_ms",
                "tree_ms",
                "populate_ms",
                "server_prepare_ms",
                "server_generate_ms",
                "server_parse_ms",
                "server_total_ms",
                "transport_ms",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-log", type=Path, required=True)
    parser.add_argument("--draft-log", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log_text = args.target_log.read_text(errors="replace")
    finals = []
    for line in log_text.splitlines():
        if "SSD request=" not in line or "finished=True" not in line:
            continue
        fields = dict(KEY_VALUE_RE.findall(line))
        finals.append(
            {
                "rid": fields["request"],
                "rounds": int(fields["rounds"]),
                "hits": int(fields["hits"]),
                "misses": int(fields["misses"]),
                "jit_drafts": int(fields["jit"]),
                "accepted_draft": int(fields["accepted_draft"]),
                "draft_wait_ms": float(fields["draft_wait_ms"]),
                "outcome_select_ms": float(fields.get("outcome_select_ms", 0.0)),
                "target_verify_ms": float(fields["target_verify_ms"]),
            }
        )

    timings = defaultdict(lambda: defaultdict(list))
    timing_components = defaultdict(lambda: defaultdict(list))
    timing_failures = defaultdict(lambda: defaultdict(int))
    for line in log_text.splitlines():
        if "SSD draft-timing " not in line:
            continue
        fields = dict(KEY_VALUE_RE.findall(line))
        rid, kind = fields["request"], fields["kind"]
        if fields["success"] == "True":
            timings[rid][kind].append(float(fields["elapsed_ms"]))
            for name in (
                "glue_ms",
                "tree_ms",
                "populate_ms",
                "server_prepare_ms",
                "server_generate_ms",
                "server_parse_ms",
                "server_total_ms",
                "transport_ms",
            ):
                if name in fields:
                    timing_components[rid][name].append(float(fields[name]))
        else:
            timing_failures[rid][kind] += 1

    histograms = {}
    for match in HIST_RE.finditer(log_text):
        histograms[match.group(1)] = {
            "fan_outs": list(ast.literal_eval(match.group(2))),
            "outcome_hist": list(ast.literal_eval(match.group(3))),
            "cacheable_hist": list(ast.literal_eval(match.group(4))),
            "recovery_rank_hist": ast.literal_eval(match.group(5)),
        }

    draft_select = defaultdict(lambda: {"hit": [], "miss": []})
    if args.draft_log is not None:
        for line in args.draft_log.read_text(errors="replace").splitlines():
            if "official-ssd-draft: SELECT " not in line:
                continue
            fields = dict(KEY_VALUE_RE.findall(line))
            bucket = "hit" if fields.get("cache_hit") == "True" else "miss"
            draft_select[fields["rid"]][bucket].append(
                float(fields["elapsed_ms"])
            )

    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    if len(finals) < len(records):
        raise RuntimeError(
            f"Found {len(finals)} finished SSD requests but {len(records)} records"
        )
    ignored_warmups = len(finals) - len(records)
    if ignored_warmups:
        finals = finals[-len(records) :]

    rows = []
    for final, record in zip(finals, records):
        rid = final["rid"]
        row = {
            **record,
            **final,
            **histograms.get(rid, {}),
            "timings": {
                kind: timings[rid].get(kind, [])
                for kind in ("initial", "jit", "outcome_cache")
            },
            "timing_components": dict(timing_components[rid]),
            "timing_failures": dict(timing_failures[rid]),
            "draft_select_hit_ms": draft_select[rid]["hit"],
            "draft_select_miss_ms": draft_select[rid]["miss"],
        }
        rows.append(row)

    by_dataset = {
        dataset: aggregate([row for row in rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in rows})
    }
    result = {
        "label": args.label,
        "ignored_warmup_requests": ignored_warmups,
        "overall": aggregate(rows),
        "by_dataset": by_dataset,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"label": args.label, "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
