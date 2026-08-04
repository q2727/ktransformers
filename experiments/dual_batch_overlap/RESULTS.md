# Dual-Batch Attention–MoE Overlap reproduction

## Outcome

The execution mechanism was reproduced, but the paper's throughput gain was
not reproduced on this Qwen3.5/KTransformers setup.  Splitting a real two-token
decode batch into two one-token microbatches makes the KT BF16 CPU MoE execute
twice per layer.  The extra CPU invocation and synchronization cost is much
larger than the Qwen3.5 GPU-attention interval available to hide it.

The implementation is decode-only.  Prefill/extend remains on the original
path, and no speculative decoding algorithm is configured.

## Platform and fixed configuration

- Target: `gpu08`, dual AMD EPYC 9554, 2 x RTX 4090 48 GiB, 1 TiB RAM.
- Model: Qwen3.5-122B-A10B, BF16 KT CPU experts, 16 GPU-resident experts per
  layer, TP=2.
- Server: eager execution (`--disable-cuda-graph`), radix cache disabled,
  maximum 8 running requests.
- Workload: 32-token prompts, temperature 0, forced 128-token outputs,
  two synchronized requests for the batch-2 modes, six repetitions.
- Both baseline and TBO have speculative decoding disabled
  (`speculative_algorithm=None`).

## Results

| Mode | Requests | Mean system tok/s | Median | Stddev | Mean CPU busy |
|---|---:|---:|---:|---:|---:|
| Single request, unsplit path | 1 | 10.667 | 10.672 | 0.406 | 47.87% |
| Ordinary SGLang dynamic batch | 2 | 20.001 | 19.436 | 2.130 | 47.98% |
| KT dual-batch interleaving | 2 | 10.824 | 10.632 | 0.854 | 48.03% |

Using the six-run means:

- Ordinary batch-2 reaches 1.875x single-request system throughput and 10.000
  tok/s per request.
- Dual-batch interleaving reaches 1.015x single-request system throughput and
  5.412 tok/s per request.
- Dual-batch is 45.9% slower than the ordinary batch-2 baseline.
- The paper reports 33.6 / 21.5 = 1.563x system scaling for its custom FP8
  DeepSeek-R1 engine.

All throughput requests returned exactly 128 tokens with a length finish.  A
separate correctness run stored generated text for two synchronized requests;
baseline and TBO matched exactly for both requests across 64 output tokens.

## Why this platform differs from the paper

The paper runs DeepSeek-R1/Kimi-K2 with a custom CPU FP8 GEMV backend on two
EPYC 9355 CPUs and RTX 5090 GPUs.  Its reported per-layer attention and MoE
times are approximately 350 us and 450 us, so attention is long enough to hide
a substantial fraction of MoE.

This target uses Qwen3.5's 36 GDN linear-attention layers and only 12 full-
attention layers, plus KT's BF16 CPU backend.  Prior component timing on the
same machine found that GPU attention is much shorter than CPU MoE: even in an
8K, 40-target-token cycle, mean attention was 0.480 ms/layer versus 2.514
ms/layer for CPU MoE.  The 36 linear-attention layers averaged only 0.275 ms.
The available overlap window is therefore too small, while ordinary batching
preserves the CPU backend's efficient two-token expert invocation.

## Reproduction

Start either server from the repository root:

```bash
MODE=baseline experiments/dual_batch_overlap/start_server.sh
MODE=dual_batch experiments/dual_batch_overlap/start_server.sh
```

Run the fixed throughput workload:

```bash
source .venv/bin/activate
python experiments/specmoe/drive_workload.py \
  --base-url http://127.0.0.1:30006 \
  --tokenizer /data/qinchong/models/Qwen3.5-122B-A10B \
  --concurrency 2 --num-requests 2 \
  --prompt-tokens 32 --output-tokens 128 \
  --warmup-requests 2 --ignore-eos \
  --output experiments/artifacts/dual_batch_overlap/dual_batch/run1.json
```

Recompute the statistics:

```bash
python experiments/dual_batch_overlap/analyze_results.py
```

The raw JSON results, correctness captures, and server logs are under
`experiments/artifacts/dual_batch_overlap/`.

## Implementation notes

- TBO is activated only for real decode batches with at least two requests.
- Each Qwen3.5 layer is staged as attention, KT CPU MoE begin, and KT CPU MoE
  finish, with a yield between stages.
- Child request-pool, KV-cache, and Mamba-cache metadata are initialized
  independently.
- Qwen3.5 MRoPE positions use shape `[3, tokens]`; the local compatibility
  layer splits the token axis rather than the generic TBO helper's first axis.
- Prefill/extend and unsupported modes remain unsplit.
