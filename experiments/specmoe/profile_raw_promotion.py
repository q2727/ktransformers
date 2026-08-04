#!/usr/bin/env python3
"""Measure promotion directly from packed Qwen safetensors to TP GPU buffers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/data/qinchong/models/Qwen3.5-122B-A10B"
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--gpu-tp-count", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--access-mode", choices=("slice", "full-mmap"), default="slice")
    parser.add_argument(
        "--expert-pattern", choices=("random", "fixed", "hotset"), default="random"
    )
    parser.add_argument("--fixed-expert-id", type=int, default=0)
    parser.add_argument("--hotset-size", type=int, default=16)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--validate-kt", action="store_true")
    parser.add_argument("--kt-cpuinfer", type=int, default=120)
    parser.add_argument("--kt-threadpool-count", type=int, default=2)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[round((len(ordered) - 1) * 0.50)],
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "p99_ms": ordered[round((len(ordered) - 1) * 0.99)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def load_model_dimensions(model_dir: Path) -> tuple[int, int, int]:
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    return (
        int(text_config["hidden_size"]),
        int(text_config["moe_intermediate_size"]),
        int(text_config["num_experts"]),
    )


def resolve_weight_file(model_dir: Path, key: str) -> Path:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    try:
        filename = index["weight_map"][key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint does not contain {key}") from exc
    return model_dir / filename


def main() -> None:
    args = parse_args()
    if args.torch_num_threads < 1:
        raise ValueError("torch-num-threads must be positive")
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.gpu_tp_count < 1:
        raise ValueError("gpu-tp-count must be positive")

    model_dir = Path(args.model)
    hidden_size, intermediate_size, num_experts = load_model_dimensions(model_dir)
    if intermediate_size % args.gpu_tp_count:
        raise ValueError("intermediate size must be divisible by gpu-tp-count")

    prefix = f"model.language_model.layers.{args.layer_idx}.mlp.experts"
    gate_up_key = f"{prefix}.gate_up_proj"
    down_key = f"{prefix}.down_proj"
    gate_up_file = resolve_weight_file(model_dir, gate_up_key)
    down_file = resolve_weight_file(model_dir, down_key)

    intermediate_per_rank = intermediate_size // args.gpu_tp_count
    w13_shape = (2 * intermediate_per_rank, hidden_size)
    w2_shape = (hidden_size, intermediate_per_rank)
    w13_host = [
        torch.empty(w13_shape, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(args.gpu_tp_count)
    ]
    w2_host = [
        torch.empty(w2_shape, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(args.gpu_tp_count)
    ]

    device = torch.device("cuda:0")
    w13_gpu = torch.empty(w13_shape, dtype=torch.bfloat16, device=device)
    w2_gpu = torch.empty(w2_shape, dtype=torch.bfloat16, device=device)
    copy_stream = torch.cuda.Stream(device=device)

    gate_handle = safe_open(gate_up_file, framework="pt", device="cpu")
    down_handle = safe_open(down_file, framework="pt", device="cpu")
    if args.access_mode == "slice":
        gate_up_source = gate_handle.get_slice(gate_up_key)
        down_source = down_handle.get_slice(down_key)
        gate_shape = tuple(gate_up_source.get_shape())
        down_shape = tuple(down_source.get_shape())
    else:
        # get_tensor returns a tensor backed by the safetensors mmap. Indexing an
        # expert then produces a view instead of asking safetensors to materialize
        # a new slice on every promotion.
        gate_up_source = gate_handle.get_tensor(gate_up_key)
        down_source = down_handle.get_tensor(down_key)
        gate_shape = tuple(gate_up_source.shape)
        down_shape = tuple(down_source.shape)

    expected_gate_shape = (num_experts, 2 * intermediate_size, hidden_size)
    expected_down_shape = (num_experts, hidden_size, intermediate_size)
    if gate_shape != expected_gate_shape:
        raise ValueError(
            f"Unexpected gate/up shape {gate_shape}, expected {expected_gate_shape}"
        )
    if down_shape != expected_down_shape:
        raise ValueError(
            f"Unexpected down shape {down_shape}, expected {expected_down_shape}"
        )

    def stage(expert_id: int) -> None:
        # Safetensors materializes only one expert. The fused tensor is ordered as
        # all gate rows followed by all up rows, while each TP rank expects its
        # gate shard followed by its up shard.
        gate_up = gate_up_source[expert_id]
        down = down_source[expert_id]
        for rank in range(args.gpu_tp_count):
            start = rank * intermediate_per_rank
            end = start + intermediate_per_rank
            w13_host[rank][:intermediate_per_rank].copy_(gate_up[start:end])
            w13_host[rank][intermediate_per_rank:].copy_(
                gate_up[intermediate_size + start : intermediate_size + end]
            )
            w2_host[rank].copy_(down[:, start:end])

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

    required = args.warmup_iters + args.bench_iters * 2
    generator = torch.Generator().manual_seed(args.seed)
    if not 0 <= args.fixed_expert_id < num_experts:
        raise ValueError("fixed-expert-id must be in the model expert range")
    if not 1 <= args.hotset_size <= num_experts:
        raise ValueError("hotset-size must be in the model expert range")
    if args.expert_pattern == "fixed":
        expert_ids = [args.fixed_expert_id] * required
    else:
        population = (
            num_experts if args.expert_pattern == "random" else args.hotset_size
        )
        order = torch.randperm(population, generator=generator).tolist()
        expert_ids = (order * ((required + population - 1) // population))[:required]

    for expert_id in expert_ids[: args.warmup_iters]:
        stage(expert_id)
        h2d_local_rank()

    stage_ms: list[float] = []
    h2d_ms: list[float] = []
    combined_ms: list[float] = []
    offset = args.warmup_iters
    for index in range(args.bench_iters):
        expert_id = expert_ids[offset + index]
        started = time.perf_counter()
        stage(expert_id)
        stage_ms.append((time.perf_counter() - started) * 1000.0)
        h2d_ms.append(h2d_local_rank())

        combined_id = expert_ids[offset + args.bench_iters + index]
        started = time.perf_counter()
        stage(combined_id)
        h2d_local_rank()
        combined_ms.append((time.perf_counter() - started) * 1000.0)

    per_rank_bytes = (
        w13_host[0].numel() * w13_host[0].element_size()
        + w2_host[0].numel() * w2_host[0].element_size()
    )
    result = {
        "configuration": {
            "model": str(model_dir),
            "layer_idx": args.layer_idx,
            "gpu_tp_count": args.gpu_tp_count,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "per_rank_expert_bytes": per_rank_bytes,
            "total_expert_bytes": per_rank_bytes * args.gpu_tp_count,
            "gate_up_file": gate_up_file.name,
            "down_file": down_file.name,
            "torch_num_threads": torch.get_num_threads(),
            "access_mode": args.access_mode,
            "expert_pattern": args.expert_pattern,
            "fixed_expert_id": args.fixed_expert_id,
            "hotset_size": args.hotset_size,
        },
        "raw_safetensors_to_pinned_all_tp_ranks": describe(stage_ms),
        "h2d_one_rank": describe(h2d_ms),
        "sequential_stage_all_ranks_plus_h2d_one_rank": describe(combined_ms),
        "buffer_checksums": {
            "w13_rank0": float(w13_host[0].float().sum()),
            "w2_rank0": float(w2_host[0].float().sum()),
        },
    }

    if args.validate_kt:
        from profile_compute_break_even import load_text_model_config
        from sglang.srt.layers.moe.benchmark_kt_ep import (
            BenchmarkKTWrapper,
            setup_minimal_server_args,
        )

        setup_minimal_server_args(str(model_dir))
        wrapper = BenchmarkKTWrapper(
            model_config=load_text_model_config(str(model_dir)),
            kt_weight_path=str(model_dir),
            kt_num_gpu_experts=0,
            kt_cpuinfer=args.kt_cpuinfer,
            kt_threadpool_count=args.kt_threadpool_count,
            kt_method="BF16",
            kt_chunked_prefill_size=1,
            device=device,
            layer_idx=args.layer_idx,
            skip_gpu_weights=True,
            override_top_k=1,
        )
        wrapper.load_cpu_weights(layer_idx=args.layer_idx)
        reference_w13 = [
            torch.empty(w13_shape, dtype=torch.bfloat16, pin_memory=True)
            for _ in range(args.gpu_tp_count)
        ]
        reference_w2 = [
            torch.empty(w2_shape, dtype=torch.bfloat16, pin_memory=True)
            for _ in range(args.gpu_tp_count)
        ]
        zeros = [0] * args.gpu_tp_count
        validation = []
        for expert_id in sorted({0, 1, num_experts // 2, num_experts - 1}):
            stage(expert_id)
            wrapper.kt_wrapper.submit_write_weight_scale_to_buffer(
                args.gpu_tp_count,
                expert_id,
                [tensor.data_ptr() for tensor in reference_w13],
                zeros,
                [tensor.data_ptr() for tensor in reference_w2],
                zeros,
            )
            wrapper.kt_wrapper.sync_write_weight_scale_to_buffer()
            for rank in range(args.gpu_tp_count):
                w13_equal = torch.equal(w13_host[rank], reference_w13[rank])
                w2_equal = torch.equal(w2_host[rank], reference_w2[rank])
                validation.append(
                    {
                        "expert_id": expert_id,
                        "tp_rank": rank,
                        "w13_exact": w13_equal,
                        "w2_exact": w2_equal,
                        "w13_max_abs_error": float(
                            (w13_host[rank].float() - reference_w13[rank].float())
                            .abs()
                            .max()
                        ),
                        "w2_max_abs_error": float(
                            (w2_host[rank].float() - reference_w2[rank].float())
                            .abs()
                            .max()
                        ),
                    }
                )
        result["kt_reconstruction_validation"] = validation

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
