# SpecMoE profiling on KTransformers

This directory contains the reproducible experiments for evaluating the idea in
"SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted
Speculative Decoding" on the `gpu08` KTransformers host.

The first target is `Qwen3.5-122B-A10B`: it has 48 MoE layers, 256 experts per
layer, Top-8 routing, and a built-in MTP layer. The server runs MTP/NEXTN with
three draft steps and four draft tokens, so each target verification pass
contains roughly `batch_size * 4` tokens.

## Files

- `launch_qwen35_specmoe.sh`: foreground server launcher with per-token expert
  recording enabled.
- `start_server.sh` / `stop_server.sh`: guarded background service management.
- `drive_workload.py`: synchronized closed-loop requests plus recorder endpoint
  control.
- `analyze_expert_distribution.py`: target-verify-only expert hotness and
  residency simulation.
- `profile_pcie.py`: pinned-host-to-GPU transfer microbenchmark using an expert
  sized payload.
- `profile_compute_break_even.py`: real KT CPU versus fused GPU expert compute.
- `profile_promotion_pipeline.py`: KT packed-weight reconstruction plus H2D.
- `profile_raw_promotion.py`: direct packed-BF16 safetensors promotion and
  exact-value validation against KT reconstruction.
- `run_control_ab.py`: same-process static/dynamic/static warmup and measurement
  sequence with per-run system observations.
- `test_decode_hot_policy.py`: CPU-only token-floor, hysteresis, and residency
  policy tests.
- `RESULTS.md`: measured route, transfer, compute, and online-update results.

Runtime data is written below `experiments/artifacts/specmoe/` and is ignored by
git.

The runtime implementation lives in the bundled SGLang tree:

- `layers/moe/kt_decode_hot.py`: policy, checkpoint staging, TP coordination,
  in-place residency updates, profiling, and exact static restoration.
- `layers/moe/kt_ep_wrapper.py`: CUDA-graph-safe route counters and layer
  registration.
- `speculative/eagle_worker.py` and `eagle_worker_v2.py`: post-target-verify
  update hook.
- `server_args.py`: validated `--kt-decode-hot-*` controls.

The current fast path deliberately supports packed Qwen BF16 weights only.
Other KT layouts need a format-specific staging implementation and validation.

## Baseline run

Only start the service in an exclusive GPU window. The launcher refuses to run
unless both GPUs have enough free memory.

```bash
cd ~/workspace/code/ktransformers

GPU_EXPERTS=16 MAX_RUNNING_REQUESTS=32 \
  ./experiments/specmoe/start_server.sh

tail -f experiments/artifacts/specmoe/services/qwen35-nextn/server.log
```

After the health endpoint becomes ready:

```bash
source .venv/bin/activate

python experiments/specmoe/drive_workload.py \
  --concurrency 32 \
  --num-requests 32 \
  --prompt-tokens 256 \
  --output-tokens 128 \
  --record

python experiments/specmoe/analyze_expert_distribution.py \
  experiments/artifacts/specmoe/expert-records/*.pt \
  --gpu-experts 4 8 16 32 \
  --output experiments/artifacts/specmoe/analysis/b32.json
```

The analysis reports actual target-verification routing, not draft-model or
prefill traffic. Counts are logical expert assignments: one target-verify token
contributes eight assignments because this model uses Top-8 routing.

Use a prompt shorter than 64 tokens for decode-only batch-32 comparisons, or
raise `GPU_PREFILL_THRESHOLD`. A 32 x 128-token prompt exceeds the default
2048-token threshold and invokes the unrelated layerwise full-GPU prefill path.

## Online hot-expert replacement

Start a dynamic-capable server in static mode so CPU weights, NUMA placement,
and CUDA graphs can be warmed before the A/B switch:

```bash
DECODE_HOT_EXPERT_UPDATE=1 \
HOT_CONTROL_MODE=static \
CPU_THREADS=60 KT_NUMA_NODES="1 1" \
RUNTIME_DIR="$PWD/experiments/artifacts/specmoe/services/qwen35-nextn-control" \
HOT_PROFILE_PATH="$PWD/experiments/artifacts/specmoe/decode-hot/control-ab.jsonl" \
HOT_CONTROL_PATH="$PWD/experiments/artifacts/specmoe/decode-hot/control" \
DECODE_HOT_REFRESH_INTERVAL=16 \
./experiments/specmoe/start_server.sh
```

The duplicated node-1 mapping is an optional shared-host isolation setup: each
TP partition gets a 30-thread subpool on NUMA node 1. Omit `CPU_THREADS` and
`KT_NUMA_NODES` to use the normal 60-thread pools on nodes 0 and 1.

The policy can replace at most one slot per layer during each of its first 16
target-verification passes. It continues updating route EMAs on every pass but
only permits later replacements every `DECODE_HOT_REFRESH_INTERVAL` passes.
The default interval of 16 is the measured conservative policy; permitting
replacement every pass is slower than static placement on this host.

Switch modes without restarting. The first target verification backs up the
initial resident weights in CPU RAM. The first verification after switching to
`static` restores the original uniform expert weights and every mapping in
place, then checks the restored state exactly.

```bash
printf '%s\n' dynamic > experiments/artifacts/specmoe/decode-hot/control
printf '%s\n' static > experiments/artifacts/specmoe/decode-hot/control
```

For a decode-focused workload that stays below the full-GPU prefill threshold:

```bash
python experiments/specmoe/drive_workload.py \
  --concurrency 32 --num-requests 32 \
  --prompt-tokens 32 --output-tokens 64 \
  --output experiments/artifacts/specmoe/workloads/control-ab.json
```

The workload JSON includes aggregate CPU busy percentage, load averages, and
NVMe throughput/utilization for the exact request interval, plus a SHA-256 of
each generated output. Do not compare runs whose system observations show
materially different shared-host load.

Output hashes are diagnostic rather than a passed correctness oracle for this
prototype. Greedy outputs were stable within a phase but not bitwise invariant
after changing CPU/GPU residency, even after exact static restoration. See
`RESULTS.md` before treating the path as production ready.

After the dynamic-capable server is ready, run the complete crossover sequence:

```bash
python experiments/specmoe/run_control_ab.py \
  --sequence static,dynamic,static \
  --warmup-runs 2 --measure-runs 3 \
  --concurrency 32 --prompt-tokens 32 --output-tokens 64
```

## PCIe transfer baseline

One BF16 Qwen3.5 expert contains three matrices. Under TP2, each GPU receives
half of the intermediate dimension, or 9,437,184 bytes per expert and rank.

```bash
source .venv/bin/activate
numactl --cpunodebind=0 --membind=0 \
  env CUDA_VISIBLE_DEVICES=0 \
  python experiments/specmoe/profile_pcie.py
```
