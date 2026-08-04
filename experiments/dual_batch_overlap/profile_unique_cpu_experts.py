#!/usr/bin/env python3
"""Profile KT BF16 CPU MoE versus the number of unique active experts.

Each active expert receives exactly one token through a synthetic top-1 route.
Expert IDs rotate between iterations so small working sets do not benchmark a
single expert left resident in the multi-socket LLC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from sglang.srt.layers.moe.benchmark_kt_ep import (
    BenchmarkKTWrapper,
    MoEModelConfig,
    setup_minimal_server_args,
)


DEFAULT_COUNTS = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    120,
    128,
    160,
    192,
    224,
    240,
    256,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/data/qinchong/models/Qwen3.5-122B-A10B"
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=120)
    parser.add_argument("--threadpool-count", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=12)
    parser.add_argument("--bench-iters", type=int, default=80)
    parser.add_argument("--counts", nargs="+", type=int, default=DEFAULT_COUNTS)
    parser.add_argument("--route-top-k", type=int, choices=(1, 8), default=1)
    parser.add_argument("--route-stride", type=int, default=37)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.counts or min(args.counts) < 1 or max(args.counts) > 256:
        parser.error("--counts must be between 1 and 256")
    if len(args.counts) != len(set(args.counts)):
        parser.error("--counts cannot contain duplicates")
    if any(count % args.route_top_k != 0 for count in args.counts):
        parser.error("every expert count must be divisible by --route-top-k")
    if math.gcd(args.route_stride, 256) != 1:
        parser.error("--route-stride must be coprime with 256")
    if args.warmup_iters < 1 or args.bench_iters < 2:
        parser.error("warmup-iters must be positive and bench-iters >= 2")
    return args


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def load_model_config(model: str, top_k: int) -> tuple[MoEModelConfig, object]:
    root = AutoConfig.from_pretrained(model, trust_remote_code=True)
    config = getattr(root, "text_config", root)
    dtype = getattr(config, "dtype", None) or getattr(
        config, "torch_dtype", torch.bfloat16
    )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""))
    return (
        MoEModelConfig(
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=top_k,
            num_layers=int(config.num_hidden_layers),
            params_dtype=dtype,
            first_moe_layer=0,
        ),
        config,
    )


def route_for_iteration(
    *, count: int, top_k: int, iteration: int, stride: int, device: torch.device
) -> torch.Tensor:
    start = (iteration * stride) % 256
    return (
        (torch.arange(count, dtype=torch.int64, device=device) + start) % 256
    ).view(count // top_k, top_k)


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, 0.10),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KT activation/result staging")

    device = torch.device("cuda:0")
    setup_minimal_server_args(args.model)
    model_config, hf_config = load_model_config(args.model, args.route_top_k)
    if model_config.num_experts != 256:
        raise RuntimeError(f"Expected 256 experts, got {model_config.num_experts}")

    wrapper = BenchmarkKTWrapper(
        model_config=model_config,
        kt_weight_path=args.model,
        kt_num_gpu_experts=0,
        kt_cpuinfer=args.cpu_threads,
        kt_threadpool_count=args.threadpool_count,
        kt_method="BF16",
        kt_chunked_prefill_size=max(args.counts),
        device=device,
        layer_idx=args.layer,
        skip_gpu_weights=True,
        override_top_k=args.route_top_k,
    )
    wrapper.load_cpu_weights(layer_idx=args.layer)

    dtype_bytes = torch.empty((), dtype=model_config.params_dtype).element_size()
    expert_weight_bytes = (
        3
        * model_config.hidden_size
        * model_config.intermediate_size
        * dtype_bytes
    )
    rows = []
    route_iteration = 0

    for count in args.counts:
        tokens = count // args.route_top_k
        hidden_states = torch.randn(
            tokens,
            model_config.hidden_size,
            dtype=model_config.params_dtype,
            device=device,
        )
        topk_weights = torch.full(
            (tokens, args.route_top_k),
            1.0 / args.route_top_k,
            dtype=torch.float32,
            device=device,
        )
        routes = [
            route_for_iteration(
                count=count,
                top_k=args.route_top_k,
                iteration=route_iteration + index,
                stride=args.route_stride,
                device=device,
            )
            for index in range(args.warmup_iters + args.bench_iters)
        ]
        route_iteration += args.warmup_iters + args.bench_iters
        if any(int(torch.unique(route).numel()) != count for route in routes):
            raise AssertionError("Synthetic route does not contain unique experts")

        for route in routes[: args.warmup_iters]:
            wrapper.apply_cpu_only(hidden_states, route, topk_weights)
        torch.cuda.synchronize(device)

        elapsed_ms = []
        for route in routes[args.warmup_iters :]:
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            output = wrapper.apply_cpu_only(hidden_states, route, topk_weights)
            torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
            if output.shape != hidden_states.shape:
                raise AssertionError("KT output shape mismatch")

        timing = describe(elapsed_ms)
        weight_bytes = count * expert_weight_bytes
        rows.append(
            {
                "unique_cpu_experts": count,
                "tokens": tokens,
                "expert_assignments": count,
                "weight_bytes": weight_bytes,
                "weight_gib": weight_bytes / 2**30,
                "timing": timing,
                "effective_weight_bandwidth_gb_per_second_mean": (
                    weight_bytes / (timing["mean_ms"] / 1000.0) / 1e9
                ),
                "effective_weight_bandwidth_gb_per_second_median": (
                    weight_bytes / (timing["median_ms"] / 1000.0) / 1e9
                ),
            }
        )
        print(
            f"experts={count:3d} median={timing['median_ms']:.3f} ms "
            f"mean={timing['mean_ms']:.3f} ms "
            f"BW(median)={rows[-1]['effective_weight_bandwidth_gb_per_second_median']:.1f} GB/s",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "experiment": "kt_bf16_unique_cpu_expert_scaling",
        "configuration": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "model": args.model,
            "layer": args.layer,
            "kt_method": "BF16",
            "cpu_threads": args.cpu_threads,
            "threadpool_count": args.threadpool_count,
            "num_experts": model_config.num_experts,
            "top_k_override": args.route_top_k,
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "dtype": str(model_config.params_dtype),
            "dtype_bytes": dtype_bytes,
            "expert_weight_bytes_node_total": expert_weight_bytes,
            "expert_weight_mib_node_total": expert_weight_bytes / 2**20,
            "warmup_iters": args.warmup_iters,
            "bench_iters": args.bench_iters,
            "route_stride": args.route_stride,
            "counts": args.counts,
            "route_definition": (
                "N/top_k tokens with routes to N distinct experts, one assignment "
                "per expert; the expert-ID window rotates between iterations to "
                "exceed LLC capacity"
            ),
            "timing_boundary": (
                "CUDA-synchronized wall time around BenchmarkKTWrapper.apply_cpu_only; "
                "includes activation staging, native KT CPU work, result copy, and sync"
            ),
            "bandwidth_definition": (
                "N * 3 * hidden_size * intermediate_size * BF16_bytes / elapsed_time; "
                "activation/result traffic and metadata are excluded from the numerator"
            ),
            "model_layer_types": getattr(hf_config, "layer_types", None),
        },
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("RESULT_JSON_BEGIN")
    print(rendered)
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
