# Mooncake ToolAgent serving benchmarks

## Data layout

```text
~/workspace/code/ktransformers/experiments/artifacts/mooncake-toolagent/
  datasets/                         # source and fixed trace windows
  traces/                           # generated fixed-rate or Poisson inputs
  results/qwen3-coder-next-fp8/     # current FP8 model runs
  results/qwen3.5-122b-a10b/        # earlier model and configuration runs
  results/qwen3.6-27b/              # dense BF16 comparison runs
```

New formal runs use the same contiguous 128-request Mooncake ToolAgent window
for every model (source indices `[11419, 11547)`). Closed-loop capacity is
measured with client concurrency 8. Open-loop Poisson tests use arrival rates
equal to 55% and 95% of that measured capacity. After the model has completed
the microbenchmarks, the fast formal path uses 4 warm-up requests at concurrency
4 with one output token.

Do not replay a completed 256-request run merely to satisfy the 128-request
budget. Use `derive_first_n.py` to select sessions 0 through 127 and recompute
the request metrics. Derived artifacts retain the source path and SHA-256. If
an older run used a different arrival rate, report its actual rate and capacity
fraction rather than relabeling it as an exact 55% or 95% run.

## Entry points

- `prepare_mooncake_openloop.py`: creates a contiguous trace window at one or
  more requested rates.
- `derive_first_n.py`: creates a reproducible first-128 view from an existing
  completed 256-request AIPerf run without issuing inference requests.
- `run_mooncake_capacity.sh`: closed-loop saturation test for
  Qwen3-Coder-Next-FP8.
- `run_mooncake_poisson.sh`: one Poisson arrival-rate run.
- `run_mooncake_slo_rates.sh`: derive and run the formal 55%/95% Poisson loads
  from a completed closed-loop `capacity_req_s.txt`. Required environment
  variables are `MODEL`, `TOKENIZER`, `DATA_ROOT`, `CONFIG_TAG`, and
  `CAPACITY_DIR`.
- `run_qcn_poisson_curve.sh`: the tested `0.10 0.12 0.14` Poisson curve.
- `run_qwen36_poisson_curve.sh`: Qwen3.6-27B non-thinking Poisson curve; its
  default order is `0.20 0.15 0.24` requests/second.
- `analyze_length_buckets.py`: compares TTFT, ITL, E2E, and joint-SLO pass
  rates by input/output length; use `--run-glob` to select a model's runs.
- `run_mooncake_batch_curve.sh`: legacy closed-loop concurrency curve.
- `run_mooncake_openloop_curve.sh`: legacy timestamp-scheduled open-loop run.

All result directories contain the AIPerf summary
`profile_export_aiperf.json`, per-request records `profile_export.jsonl`, and
server/GPU telemetry when enabled. Status and console files use the same run
name beside the result directory.

The SLO used by the current scripts is:

```text
TTFT <= 20 seconds
ITL <= 250 milliseconds
E2E request latency <= 120 seconds
```
