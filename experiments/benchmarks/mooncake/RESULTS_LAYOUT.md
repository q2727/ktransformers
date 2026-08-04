# Mooncake ToolAgent artifacts

```text
datasets/
  toolagent_trace.jsonl
  toolagent_fullctx_256.jsonl
  toolagent_fullctx_512.jsonl
  toolagent_16k_pilot24.jsonl
  analysis/
traces/
  openloop_256/
  openloop_1024/
  poisson_256/
  qcn_capacity/
  qcn_curve/
results/
  qwen3.5-122b-a10b/
  qwen3-coder-next-fp8/
```

The Qwen3.5 directory contains the earlier configuration sweeps, including
interrupted and contaminated runs retained for provenance. The
Qwen3-Coder-Next directory contains the FP8 capacity test, Poisson curve, and
streaming speed microbenchmark.

Run names encode the important serving configuration. Each completed AIPerf
run has `profile_export_aiperf.json` for aggregate metrics and
`profile_export.jsonl` for per-request metrics.
