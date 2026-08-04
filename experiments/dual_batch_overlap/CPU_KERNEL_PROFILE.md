# KT BF16 CPU kernel: unique active expert scaling

## Method

This profile runs the native KT `AVX512-BF16` CPU MoE path on `gpu08` without
loading the full model server.  Layer 0 of Qwen3.5-122B-A10B is used; all 48
MoE layers have the same expert dimensions.

- CPU: dual AMD EPYC 9554.
- KT workers: 120 threads, two pools, 60 threads per NUMA node.
- All 256 experts reside on CPU for this microbenchmark.
- One expert contains `3 * 3072 * 1024` BF16 parameters: 18 MiB node-total.
- The primary scan uses N tokens with top-1 routes to N distinct experts.
- A second scan uses the model's real top-8 route shape: N/8 tokens and N
  distinct expert assignments.
- Expert IDs rotate by a coprime stride between iterations.  This prevents the
  small-count cases from repeatedly measuring one expert retained in LLC.
- Each point is three independent processes, each with 12 warmups and 80
  measured iterations.  The table reports the mean of the three per-run
  medians.
- Timing is CUDA-synchronized wall time around `apply_cpu_only`: activation
  staging, KT submit/CPU compute/sync, and result return are included.  GPU
  expert computation is excluded.
- Effective bandwidth counts only BF16 expert-weight bytes, so it is an
  end-to-end effective weight bandwidth rather than a raw STREAM measurement.

## Primary top-1 scan

| Unique CPU experts | Median time (ms) | Effective BW (GB/s) |
|---:|---:|---:|
| 1 | 0.265 | 71.3 |
| 2 | 0.398 | 94.8 |
| 4 | 0.438 | 172.6 |
| 8 | 0.524 | 288.0 |
| 12 | 0.631 | 359.0 |
| 16 | 0.813 | 371.6 |
| 24 | 1.144 | 395.9 |
| 32 | 1.452 | 416.1 |
| 48 | 2.078 | 436.0 |
| 64 | 2.721 | 444.0 |
| 96 | 3.901 | 464.5 |
| 120 | 4.745 | 477.3 |
| 128 | 5.073 | 476.3 |
| 160 | 6.260 | 482.4 |
| 192 | 7.470 | 485.2 |
| 224 | 8.682 | 487.0 |
| 240 | 9.266 | 488.9 |
| 256 | 9.859 | 490.1 |

The high-count region (64–256 experts) fits

```text
time_ms = 0.306 + 0.03731 * unique_experts
```

The fitted asymptotic effective bandwidth is 505.8 GB/s.  The measured curve
first reaches 90% of its observed maximum at 64 experts and 95% at 120
experts.  This aligns with the 120 CPU workers: beyond roughly one independent
expert task per worker, additional experts mainly extend the bandwidth-bound
work rather than improve utilization.

## Top-8 route-shape validation

| Unique CPU experts | Tokens (`N/8`) | Median time (ms) | Effective BW (GB/s) |
|---:|---:|---:|---:|
| 8 | 1 | 0.521 | 290.0 |
| 16 | 2 | 0.812 | 372.2 |
| 24 | 3 | 1.150 | 393.8 |
| 32 | 4 | 1.467 | 411.8 |
| 48 | 6 | 2.111 | 429.2 |
| 64 | 8 | 2.739 | 441.1 |
| 96 | 12 | 3.833 | 472.8 |
| 120 | 15 | 4.695 | 482.5 |
| 128 | 16 | 4.992 | 484.0 |
| 160 | 20 | 6.149 | 491.2 |
| 192 | 24 | 7.320 | 495.1 |
| 224 | 28 | 8.476 | 498.8 |
| 240 | 30 | 9.083 | 498.7 |
| 256 | 32 | 9.679 | 499.2 |

At equal unique-expert counts, top-1 versus top-8 organization differs by only
1.36% on average (maximum absolute difference 2.37%).  The primary curve is
therefore a good measurement of unique-expert weight scaling rather than an
artifact of synthetic top-1 routing.

## Interpretation

Bandwidth rises rapidly while the workload is too small to occupy both NUMA
nodes and all memory channels.  It then approaches a broad 480–500 GB/s
plateau.  AMD specifies 460.8 GB/s per EPYC 9554 socket, or 921.6 GB/s for two
sockets at the CPU's maximum supported memory rate.  The observed top-8 peak
is therefore 54.2% of that theoretical CPU limit.  Actual DIMM speed was not
available through unprivileged `dmidecode`, so this percentage is a CPU-limit
reference, not a verified installed-DRAM peak percentage.

For the earlier 2.514 ms 40-token measurement, the high-count fits from both
route shapes independently imply about 59 unique cold-expert weight-load
equivalents.  This is an estimate, not the exact route count: that timing trace
did not save top-k IDs, and real target-verification batches can reuse an
expert for multiple tokens.  An exact integer requires rerunning the 8K cycle
with the expert-distribution recorder enabled alongside timing.

## Artifacts

- Raw top-1 runs: `run1.json`, `run2.json`, `run3.json`.
- Raw top-8 runs: `topk8_run1.json`, `topk8_run2.json`, `topk8_run3.json`.
- Aggregate: `summary.json`.
- Profiler: `experiments/dual_batch_overlap/profile_unique_cpu_experts.py`.
- Analyzer: `experiments/dual_batch_overlap/analyze_unique_cpu_experts.py`.
