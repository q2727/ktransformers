# Mooncake serving result summary

Workload: the fixed 256-request Mooncake ToolAgent window at source indices
`11419:11675`.

SLO used by the current runs:

```text
TTFT <= 20 seconds
ITL <= 250 milliseconds
E2E request latency <= 120 seconds
```

## Qwen3-Coder-Next-FP8

Tested serving configuration:

```text
TP=2
GPU experts per layer=100
max running requests=4
max total tokens=256000
Mamba cache slots=378 (automatic)
FP8 GEMM backend=Triton
```

Closed-loop saturation capacity:

```text
256 requests in 1681.02 seconds
0.152288 requests/second
30.44 output tokens/second
```

Poisson curve:

| Target rate | Actual issue rate | Completion rate | TTFT p50/p95 | E2E p95 | Avg/max queue | Joint SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.100 | 0.0944 | 0.0932 | 6.8 / 57.1 s | 109.7 s | 0.75 / 10 | 71.5% |
| 0.120 | 0.1133 | 0.1114 | 7.9 / 103.4 s | 132.1 s | 1.60 / 15 | 61.7% |
| 0.140 | 0.1322 | 0.1292 | 10.3 / 136.8 s | 159.1 s | 2.71 / 20 | 57.4% |
| 0.152 | 0.1435 | 0.1392 | 16.4 / 150.1 s | 174.9 s | 3.86 / 22 | 48.0% |

The measured production-oriented point among these runs is 0.10 requests per
second. A lower 0.08 point is still useful if a higher SLO pass ratio is
required.

Single-request streaming microbenchmark:

| Input/output tokens | TTFT | Decode rate |
| ---: | ---: | ---: |
| 1020 / 500 | 0.404 s | 73.60 tokens/s |
| 2027 / 500 | 0.435 s | 71.93 tokens/s |
| 4041 / 500 | 4.359 s | 71.48 tokens/s |
| 8069 / 128 | 4.660 s | 68.66 tokens/s |
| 16125 / 128 | 5.260 s | 61.38 tokens/s |

## Qwen3.5-122B-A10B

Earlier closed-loop capacity curve:

| Concurrency | Completion rate |
| ---: | ---: |
| 1 | 0.08245 requests/s |
| 2 | 0.10160 requests/s |
| 4 | 0.11417 requests/s |
| 8 | 0.11520 requests/s |

Earlier Poisson runs:

| Target rate | Completion rate | TTFT p50/p95 | E2E p50/p95 | ITL p50/p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.08 | 0.07364 | 5.8 / 196.9 s | 17.2 / 289.7 s | 70.8 / 818.2 ms |
| 0.10 | 0.08860 | 8.4 / 252.6 s | 23.2 / 353.8 s | 97.0 / 1066.4 ms |

## Canonical artifacts

```text
~/workspace/code/ktransformers/experiments/artifacts/mooncake-toolagent/results/qwen3-coder-next-fp8/
~/workspace/code/ktransformers/experiments/artifacts/mooncake-toolagent/results/qwen3.5-122b-a10b/
```

Aggregate metrics are in `profile_export_aiperf.json`; per-request metrics are
in `profile_export.jsonl`.
