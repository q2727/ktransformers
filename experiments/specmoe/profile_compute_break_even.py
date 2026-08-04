#!/usr/bin/env python3
"""Compare one-expert KT CPU and SGLang GPU compute over token counts."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from transformers import AutoConfig

from sglang.srt.layers.moe.benchmark_kt_ep import (
    BenchmarkKTWrapper,
    MoEModelConfig,
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
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument(
        "--token-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--pcie-microseconds-per-expert", type=float, default=352.0)
    return parser.parse_args()


def measure(fn, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[round((len(ordered) - 1) * 0.50)],
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def make_inputs(tokens: int, hidden_size: int, device: torch.device):
    hidden_states = torch.randn(
        tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.zeros((tokens, 1), dtype=torch.int64, device=device)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32, device=device)
    return hidden_states, topk_ids, topk_weights


def load_text_model_config(model_path: str) -> MoEModelConfig:
    root = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = getattr(root, "text_config", root)
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""))
    if dtype is None:
        dtype = torch.bfloat16
    return MoEModelConfig(
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.moe_intermediate_size),
        num_experts=int(config.num_experts),
        top_k=int(config.num_experts_per_tok),
        num_layers=int(config.num_hidden_layers),
        params_dtype=dtype,
        first_moe_layer=0,
    )


def main() -> None:
    args = parse_args()
    args.kt_weight_path = args.kt_weight_path or args.model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    setup_minimal_server_args(args.model)
    model_config = load_text_model_config(args.model)
    layer_idx = max(args.layer_idx, model_config.first_moe_layer)

    cpu = BenchmarkKTWrapper(
        model_config=model_config,
        kt_weight_path=args.kt_weight_path,
        kt_num_gpu_experts=0,
        kt_cpuinfer=args.kt_cpuinfer,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_method=args.kt_method,
        kt_chunked_prefill_size=max(args.token_counts),
        device=device,
        layer_idx=layer_idx,
        skip_gpu_weights=True,
        override_top_k=1,
    )
    cpu.load_cpu_weights(layer_idx=layer_idx)

    gpu = BenchmarkKTWrapper(
        model_config=model_config,
        kt_weight_path=args.kt_weight_path,
        kt_num_gpu_experts=1,
        kt_cpuinfer=args.kt_cpuinfer,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_method=args.kt_method,
        kt_chunked_prefill_size=max(args.token_counts),
        device=device,
        layer_idx=layer_idx,
        skip_gpu_weights=False,
        override_top_k=1,
    )

    rows = []
    for tokens in args.token_counts:
        hidden_states, topk_ids, topk_weights = make_inputs(
            tokens, model_config.hidden_size, device
        )
        cpu_result = measure(
            lambda: cpu.apply_cpu_only(hidden_states, topk_ids, topk_weights),
            args.warmup_iters,
            args.bench_iters,
        )
        gpu_result = measure(
            lambda: gpu.apply_gpu_only(hidden_states, topk_ids, topk_weights),
            args.warmup_iters,
            args.bench_iters,
        )
        pcie_ms = args.pcie_microseconds_per_expert / 1000.0
        rows.append(
            {
                "tokens_for_expert": tokens,
                "cpu": cpu_result,
                "gpu": gpu_result,
                "pcie_plus_gpu_mean_ms": pcie_ms + gpu_result["mean_ms"],
                "cpu_minus_pcie_gpu_ms": cpu_result["mean_ms"]
                - pcie_ms
                - gpu_result["mean_ms"],
                "migration_profitable_lower_bound": (
                    pcie_ms + gpu_result["mean_ms"] < cpu_result["mean_ms"]
                ),
            }
        )

    result = {
        "configuration": {
            "model": args.model,
            "layer_idx": layer_idx,
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "num_experts": model_config.num_experts,
            "top_k_override": 1,
            "kt_method": args.kt_method,
            "kt_cpuinfer": args.kt_cpuinfer,
            "kt_threadpool_count": args.kt_threadpool_count,
            "pcie_microseconds_per_expert_per_rank": args.pcie_microseconds_per_expert,
            "note": "PCIe+GPU excludes KT CPU-to-pinned staging cost; it is a lower bound.",
        },
        "rows": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
