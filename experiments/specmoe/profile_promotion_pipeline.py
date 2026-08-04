#!/usr/bin/env python3
"""Measure KT BF16 expert unpack-to-pinned plus PCIe promotion latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from profile_compute_break_even import load_text_model_config
from sglang.srt.layers.moe.benchmark_kt_ep import (
    BenchmarkKTWrapper,
    setup_minimal_server_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/data/qinchong/models/Qwen3.5-122B-A10B"
    )
    parser.add_argument("--kt-weight-path")
    parser.add_argument("--kt-method", default="BF16")
    parser.add_argument("--kt-cpuinfer", type=int, default=120)
    parser.add_argument("--kt-threadpool-count", type=int, default=2)
    parser.add_argument("--gpu-tp-count", type=int, default=2)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    return parser.parse_args()


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[round((len(ordered) - 1) * 0.50)],
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "p99_ms": ordered[round((len(ordered) - 1) * 0.99)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    args = parse_args()
    args.kt_weight_path = args.kt_weight_path or args.model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    setup_minimal_server_args(args.model)
    model_config = load_text_model_config(args.model)
    if model_config.intermediate_size % args.gpu_tp_count:
        raise ValueError("intermediate size must be divisible by gpu-tp-count")

    wrapper = BenchmarkKTWrapper(
        model_config=model_config,
        kt_weight_path=args.kt_weight_path,
        kt_num_gpu_experts=0,
        kt_cpuinfer=args.kt_cpuinfer,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_method=args.kt_method,
        kt_chunked_prefill_size=1,
        device=device,
        layer_idx=args.layer_idx,
        skip_gpu_weights=True,
        override_top_k=1,
    )
    wrapper.load_cpu_weights(layer_idx=args.layer_idx)

    intermediate_per_rank = model_config.intermediate_size // args.gpu_tp_count
    w13_shape = (2 * intermediate_per_rank, model_config.hidden_size)
    w2_shape = (model_config.hidden_size, intermediate_per_rank)
    w13_host = [
        torch.empty(w13_shape, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(args.gpu_tp_count)
    ]
    w2_host = [
        torch.empty(w2_shape, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(args.gpu_tp_count)
    ]
    zeros = [0] * args.gpu_tp_count
    w13_ptrs = [tensor.data_ptr() for tensor in w13_host]
    w2_ptrs = [tensor.data_ptr() for tensor in w2_host]

    w13_gpu = torch.empty(w13_shape, dtype=torch.bfloat16, device=device)
    w2_gpu = torch.empty(w2_shape, dtype=torch.bfloat16, device=device)
    copy_stream = torch.cuda.Stream(device=device)

    def unpack(expert_id: int) -> None:
        wrapper.kt_wrapper.submit_write_weight_scale_to_buffer(
            args.gpu_tp_count,
            expert_id,
            w13_ptrs,
            zeros,
            w2_ptrs,
            zeros,
        )
        wrapper.kt_wrapper.sync_write_weight_scale_to_buffer()

    def h2d_local_rank() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            start.record(copy_stream)
            w13_gpu.copy_(w13_host[0], non_blocking=True)
            w2_gpu.copy_(w2_host[0], non_blocking=True)
            end.record(copy_stream)
        end.synchronize()
        return float(start.elapsed_time(end))

    for index in range(args.warmup_iters):
        unpack(index % model_config.num_experts)
        h2d_local_rank()

    unpack_ms = []
    h2d_ms = []
    combined_ms = []
    for index in range(args.bench_iters):
        expert_id = (index + args.warmup_iters) % model_config.num_experts
        started = time.perf_counter()
        unpack(expert_id)
        unpack_ms.append((time.perf_counter() - started) * 1000.0)

        h2d_ms.append(h2d_local_rank())

        started = time.perf_counter()
        unpack((expert_id + 97) % model_config.num_experts)
        h2d_local_rank()
        combined_ms.append((time.perf_counter() - started) * 1000.0)

    per_rank_bytes = (
        w13_host[0].numel() * w13_host[0].element_size()
        + w2_host[0].numel() * w2_host[0].element_size()
    )
    result = {
        "configuration": {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "gpu_tp_count": args.gpu_tp_count,
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "per_rank_expert_bytes": per_rank_bytes,
            "total_expert_bytes": per_rank_bytes * args.gpu_tp_count,
            "kt_method": args.kt_method,
            "kt_cpuinfer": args.kt_cpuinfer,
            "kt_threadpool_count": args.kt_threadpool_count,
        },
        "unpack_all_tp_ranks": describe(unpack_ms),
        "h2d_one_rank": describe(h2d_ms),
        "sequential_unpack_plus_h2d": describe(combined_ms),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
