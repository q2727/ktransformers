# KTransformers experiments

This directory contains deployment and benchmark helpers created for the
`qinchong@100.64.0.3` test machine. It is intentionally separate from the
upstream KTransformers source tree.

## Layout

```text
experiments/
  models/
    qwen3-coder-next-fp8/  # launch, start, stop, and weight verification
    qwen3.5-122b-a10b/     # launch, start, stop, verification, and runbook
    minimax-m2.5/          # official FP8 launch and verification
    deepseek-v4-flash/     # native MXFP4 launch and isolated environment
  benchmarks/
    common/                # model-independent microbenchmarks
    micro/                 # formal TTFT/TPOT measurements and SVG plots
    mooncake/              # Mooncake ToolAgent trace preparation and replay
```

Runtime logs and PID files are stored inside the repository under:

```text
experiments/artifacts/services/<model>/
```

Mooncake datasets, generated traces, and results are stored under:

```text
experiments/artifacts/mooncake-toolagent/
  datasets/
  traces/
  results/
    qwen3-coder-next-fp8/
    qwen3.5-122b-a10b/
```

Only model weights and model caches live on `/data/qinchong`.

The formal serving matrix excludes Qwen3.5-35B-A3B and currently contains
Qwen3-Coder-Next-FP8, Qwen3.5-122B-A10B BF16, MiniMax-M2.5 FP8, and
DeepSeek-V4-Flash native MXFP4. Each model produces a 1K–128K TTFT curve, an
8K-context batch 1/2/4/6/8 TPOT curve, Mooncake closed-loop c8 capacity, and
Mooncake Poisson results at 55% and 95% of that capacity.

To keep the matrix practical, new TTFT and TPOT points use one measured trial
per point. New Mooncake runs use 128 requests and 4 warm-up requests after the
microbenchmarks have warmed the service. A
completed 256-request run is reused through a documented first-128 derivation
instead of being replayed.

See `benchmarks/mooncake/README.md` for benchmark entry points and result
naming conventions.
