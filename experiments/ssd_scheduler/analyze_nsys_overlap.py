#!/usr/bin/env python3
"""Measure cross-process CUDA overlap from two Nsight Systems SQLite exports."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


Interval = tuple[int, int]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def duration(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in intervals)


def intersect(left: list[Interval], right: list[Interval]) -> list[Interval]:
    result: list[Interval] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            result.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return merge_intervals(result)


class Trace:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        row = self.connection.execute(
            "SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
        self.clock_offset_ns = int(row[0])

    def kernels(self) -> tuple[list[Interval], int]:
        rows = self.connection.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
        raw = [
            (self.clock_offset_ns + int(start), self.clock_offset_ns + int(end))
            for start, end in rows
        ]
        return merge_intervals(raw), len(raw)

    def nvtx(self, label: str) -> list[Interval]:
        rows = self.connection.execute(
            """
            SELECT n.start, n.end
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON s.id = n.textId
            WHERE COALESCE(n.text, s.value) = ? AND n.end IS NOT NULL
            ORDER BY n.start
            """,
            (label,),
        )
        return [
            (self.clock_offset_ns + int(start), self.clock_offset_ns + int(end))
            for start, end in rows
        ]


def milliseconds(value_ns: int) -> float:
    return value_ns / 1e6


def nvtx_summary(trace: Trace, label: str) -> dict:
    intervals = trace.nvtx(label)
    values = [milliseconds(end - start) for start, end in intervals]
    return {
        "count": len(values),
        "sum_ms": sum(values),
        "median_ms": statistics.median(values) if values else 0.0,
        "max_ms": max(values, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    target_trace = Trace(args.target)
    draft_trace = Trace(args.draft)
    target_kernels, target_kernel_count = target_trace.kernels()
    draft_kernels, draft_kernel_count = draft_trace.kernels()
    overlap = intersect(target_kernels, draft_kernels)
    gpu_union = merge_intervals([*target_kernels, *draft_kernels])

    target_active_ns = duration(target_kernels)
    draft_active_ns = duration(draft_kernels)
    overlap_ns = duration(overlap)
    gpu_union_ns = duration(gpu_union)
    capture_begin = min(target_kernels[0][0], draft_kernels[0][0])
    capture_end = max(target_kernels[-1][1], draft_kernels[-1][1])

    target_verify_ranges = merge_intervals(target_trace.nvtx("ssd_target_verify"))
    draft_during_target_verify_ns = duration(
        intersect(target_verify_ranges, draft_kernels)
    )

    result = {
        "target_kernel_count": target_kernel_count,
        "draft_kernel_count": draft_kernel_count,
        "capture_span_ms": milliseconds(capture_end - capture_begin),
        "target_cuda_active_ms": milliseconds(target_active_ns),
        "draft_cuda_active_ms": milliseconds(draft_active_ns),
        "cross_process_overlap_ms": milliseconds(overlap_ns),
        "gpu_union_active_ms": milliseconds(gpu_union_ns),
        "overlap_fraction_of_target_active": (
            overlap_ns / target_active_ns if target_active_ns else 0.0
        ),
        "overlap_fraction_of_draft_active": (
            overlap_ns / draft_active_ns if draft_active_ns else 0.0
        ),
        "draft_cuda_during_target_verify_nvtx_ms": milliseconds(
            draft_during_target_verify_ns
        ),
        "nvtx": {
            "target_verify": nvtx_summary(target_trace, "ssd_target_verify"),
            "target_wait_outcome_cache": nvtx_summary(
                target_trace, "ssd_wait_outcome_cache"
            ),
            "target_jit_draft_wait": nvtx_summary(
                target_trace, "ssd_jit_draft_wait"
            ),
            "draft_extend": nvtx_summary(
                draft_trace, "ssd::draft_server::extend"
            ),
            "draft_decode": nvtx_summary(
                draft_trace, "ssd::draft_server::decode"
            ),
        },
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
