#!/usr/bin/env python3
"""Analyze expert routing in speculative target-verification forward passes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

import torch


TARGET_VERIFY_MODE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--gpu-experts", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--expert-bytes-per-rank", type=int, default=9_437_184)
    parser.add_argument("--pcie-gib-per-second", type=float, default=24.0)
    parser.add_argument("--gpu-mask-record", type=Path)
    parser.add_argument("--migration-token-threshold", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value < 1 or value > args.num_experts for value in args.gpu_experts):
        parser.error("each --gpu-experts value must be between 1 and num-experts")
    return args


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * q)])


def gini(values: Iterable[int]) -> float:
    ordered = sorted(value for value in values if value > 0)
    if not ordered:
        return 0.0
    total = sum(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(
        ordered
    )


def load_target_verify_records(paths: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for record in payload.get("records", []):
            if int(record.get("rank", 0)) != 0:
                continue
            if int(record.get("forward_mode", -1)) != TARGET_VERIFY_MODE:
                continue
            key = (str(path), int(record["forward_pass_id"]))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    records.sort(key=lambda item: int(item["forward_pass_id"]))
    return records


def load_recorded_gpu_mask(path: Path | None) -> tuple[torch.Tensor | None, dict | None]:
    if path is None:
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    masks = payload.get("gpu_expert_masks")
    if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
        raise RuntimeError(f"Unexpected GPU mask payload in {path}")
    nonzero = masks[masks.any(dim=(1, 2))]
    if nonzero.numel() == 0:
        raise RuntimeError(f"No populated GPU masks found in {path}")
    changes = sum(
        not torch.equal(nonzero[index - 1], nonzero[index])
        for index in range(1, nonzero.shape[0])
    )
    return nonzero[0], {
        "path": str(path),
        "populated_forward_passes": int(nonzero.shape[0]),
        "mask_changes": changes,
        "experts_per_layer_min": int(nonzero[0].sum(dim=1).min()),
        "experts_per_layer_max": int(nonzero[0].sum(dim=1).max()),
    }


def top_indices(counts: torch.Tensor, count: int) -> torch.Tensor:
    # Stable tie breaking is useful when many experts have zero assignments.
    ids = torch.arange(counts.numel(), dtype=torch.float64)
    scores = counts.to(torch.float64) * (counts.numel() + 1) - ids / (
        counts.numel() + 1
    )
    return torch.topk(scores, count, sorted=True).indices.sort().values


def summarize(
    args: argparse.Namespace,
    records: list[dict],
    recorded_gpu_mask: torch.Tensor | None,
    gpu_mask_metadata: dict | None,
) -> dict:
    if not records:
        raise RuntimeError("No rank-0 TARGET_VERIFY records were found")

    per_pass = []
    layer_rows = []
    count_matrices = []
    actual_gpu_coverages = []
    threshold_expert_counts = []
    threshold_assignment_coverages = []
    nonresident_candidate_counts = []
    nonresident_candidate_coverages = []
    for record in records:
        routed = record["topk_ids_of_layer"].to(torch.int64)
        if routed.ndim != 3:
            raise RuntimeError(f"Unexpected route tensor shape: {tuple(routed.shape)}")
        layers, tokens, top_k = routed.shape
        counts = torch.zeros((layers, args.num_experts), dtype=torch.int64)
        valid = (routed >= 0) & (routed < args.num_experts)
        counts.scatter_add_(1, routed.masked_fill(~valid, 0).reshape(layers, -1), valid.reshape(layers, -1).to(torch.int64))
        count_matrices.append(counts)

        active_per_layer = (counts > 0).sum(dim=1)
        max_load_per_layer = counts.max(dim=1).values
        batch_size = len(record.get("extend_seq_lens") or [])
        per_pass.append(
            {
                "forward_pass_id": int(record["forward_pass_id"]),
                "batch_size": batch_size,
                "verify_tokens": tokens,
                "top_k": top_k,
                "active_experts_mean": float(active_per_layer.float().mean()),
                "active_experts_min": int(active_per_layer.min()),
                "active_experts_max": int(active_per_layer.max()),
                "max_tokens_per_expert_mean": float(max_load_per_layer.float().mean()),
                "max_tokens_per_expert_max": int(max_load_per_layer.max()),
            }
        )

        for layer in range(layers):
            active_loads = counts[layer][counts[layer] > 0].tolist()
            assignments = int(counts[layer].sum())
            sorted_loads = sorted(active_loads, reverse=True)
            row = {
                "forward_pass_id": int(record["forward_pass_id"]),
                "layer": layer,
                "verify_tokens": tokens,
                "assignments": assignments,
                "active_experts": len(active_loads),
                "tokens_per_active_expert_mean": statistics.fmean(active_loads),
                "tokens_per_active_expert_p50": percentile(active_loads, 0.50),
                "tokens_per_active_expert_p90": percentile(active_loads, 0.90),
                "tokens_per_active_expert_p99": percentile(active_loads, 0.99),
                "tokens_per_expert_max": max(active_loads),
                "active_load_gini": gini(active_loads),
            }
            threshold_mask = counts[layer] >= args.migration_token_threshold
            threshold_count = int(threshold_mask.sum())
            threshold_coverage = (
                int(counts[layer][threshold_mask].sum()) / assignments
                if assignments
                else 0.0
            )
            threshold_expert_counts.append(threshold_count)
            threshold_assignment_coverages.append(threshold_coverage)
            row["migration_candidate_experts"] = threshold_count
            row["migration_candidate_assignment_coverage"] = threshold_coverage

            if recorded_gpu_mask is not None:
                if tuple(recorded_gpu_mask.shape) != (layers, args.num_experts):
                    raise RuntimeError(
                        "Recorded GPU mask shape does not match route records: "
                        f"{tuple(recorded_gpu_mask.shape)} vs {(layers, args.num_experts)}"
                    )
                resident = recorded_gpu_mask[layer]
                actual_coverage = (
                    int(counts[layer][resident].sum()) / assignments
                    if assignments
                    else 0.0
                )
                nonresident_candidates = threshold_mask & ~resident
                nonresident_count = int(nonresident_candidates.sum())
                nonresident_coverage = (
                    int(counts[layer][nonresident_candidates].sum()) / assignments
                    if assignments
                    else 0.0
                )
                actual_gpu_coverages.append(actual_coverage)
                nonresident_candidate_counts.append(nonresident_count)
                nonresident_candidate_coverages.append(nonresident_coverage)
                row["recorded_gpu_assignment_coverage"] = actual_coverage
                row["nonresident_migration_candidates"] = nonresident_count
                row["nonresident_candidate_assignment_coverage"] = (
                    nonresident_coverage
                )
            for resident in args.gpu_experts:
                row[f"oracle_top{resident}_coverage"] = (
                    sum(sorted_loads[:resident]) / assignments if assignments else 0.0
                )
            layer_rows.append(row)

    residency = {}
    for resident in args.gpu_experts:
        oracle_coverages = []
        lagged_coverages = []
        promotion_counts = []
        previous_sets = None
        for counts in count_matrices:
            current_sets = []
            for layer in range(counts.shape[0]):
                current = set(top_indices(counts[layer], resident).tolist())
                current_sets.append(current)
                assignments = int(counts[layer].sum())
                oracle_coverages.append(
                    sum(int(counts[layer, expert]) for expert in current) / assignments
                    if assignments
                    else 0.0
                )
                if previous_sets is not None:
                    previous = previous_sets[layer]
                    lagged_coverages.append(
                        sum(int(counts[layer, expert]) for expert in previous)
                        / assignments
                        if assignments
                        else 0.0
                    )
                    promotion_counts.append(len(current - previous))
            previous_sets = current_sets

        mean_promotions = statistics.fmean(promotion_counts) if promotion_counts else 0.0
        bytes_per_layer_pass = mean_promotions * args.expert_bytes_per_rank
        transfer_ms = (
            bytes_per_layer_pass
            / (args.pcie_gib_per_second * 1024**3)
            * 1000.0
        )
        residency[str(resident)] = {
            "oracle_assignment_coverage_mean": statistics.fmean(oracle_coverages),
            "lag_one_pass_assignment_coverage_mean": (
                statistics.fmean(lagged_coverages) if lagged_coverages else None
            ),
            "promotions_per_layer_pass_mean": mean_promotions,
            "promotion_bytes_per_rank_per_layer_pass_mean": bytes_per_layer_pass,
            "ideal_pcie_time_ms_per_layer_pass_mean": transfer_ms,
            "ideal_pcie_time_ms_per_48_layer_pass": transfer_ms
            * count_matrices[0].shape[0],
        }

    active_values = [row["active_experts"] for row in layer_rows]
    max_values = [row["tokens_per_expert_max"] for row in layer_rows]
    verify_tokens = [item["verify_tokens"] for item in per_pass]
    return {
        "record_count": len(records),
        "model": {
            "num_layers": int(count_matrices[0].shape[0]),
            "num_experts": args.num_experts,
            "top_k": int(records[0]["topk_ids_of_layer"].shape[2]),
            "expert_bytes_per_rank": args.expert_bytes_per_rank,
        },
        "target_verify": {
            "tokens_mean": statistics.fmean(verify_tokens),
            "tokens_min": min(verify_tokens),
            "tokens_max": max(verify_tokens),
            "active_experts_per_layer_pass_mean": statistics.fmean(active_values),
            "active_experts_per_layer_pass_p50": percentile(active_values, 0.50),
            "active_experts_per_layer_pass_p90": percentile(active_values, 0.90),
            "active_experts_per_layer_pass_max": max(active_values),
            "hottest_expert_tokens_mean": statistics.fmean(max_values),
            "hottest_expert_tokens_p90": percentile(max_values, 0.90),
            "hottest_expert_tokens_max": max(max_values),
            "migration_token_threshold": args.migration_token_threshold,
            "candidate_experts_per_layer_pass_mean": statistics.fmean(
                threshold_expert_counts
            ),
            "candidate_experts_per_layer_pass_p90": percentile(
                threshold_expert_counts, 0.90
            ),
            "candidate_assignment_coverage_mean": statistics.fmean(
                threshold_assignment_coverages
            ),
        },
        "recorded_gpu_residency": (
            {
                **gpu_mask_metadata,
                "assignment_coverage_mean": statistics.fmean(
                    actual_gpu_coverages
                ),
                "nonresident_candidate_experts_per_layer_pass_mean": statistics.fmean(
                    nonresident_candidate_counts
                ),
                "nonresident_candidate_assignment_coverage_mean": statistics.fmean(
                    nonresident_candidate_coverages
                ),
            }
            if recorded_gpu_mask is not None
            else None
        ),
        "residency_simulation": residency,
        "passes": per_pass,
        "layers": layer_rows,
    }


def main() -> None:
    args = parse_args()
    records = load_target_verify_records(args.records)
    recorded_gpu_mask, gpu_mask_metadata = load_recorded_gpu_mask(
        args.gpu_mask_record
    )
    result = summarize(args, records, recorded_gpu_mask, gpu_mask_metadata)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
