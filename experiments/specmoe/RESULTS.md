# SpecMoE on KTransformers: profiling results

## Experimental setup

- Host: `gpu08`, 2 x 48 GiB RTX 4090 (PCIe, no NVLink).
- CPU: 2 x AMD EPYC 9554, 64 cores per socket, 1 TiB RAM.
- Model: Qwen3.5-122B-A10B BF16, 48 MoE layers, 256 experts per
  layer, Top-8 routing, TP2.
- Decode: SGLang NEXTN/MTP, 3 speculative steps, 4 draft tokens,
  maximum request batch 32.
- Baseline residency: 16 uniformly selected GPU experts per layer; all other
  experts execute through the KT AVX512-BF16 CPU path.
- Controlled A/B topology: two 30-thread KT pools, both on NUMA node 1. An
  unrelated shared-host task occupied node 0, so absolute throughput is
  specific to this isolation setup.

All paths below are relative to `experiments/artifacts/specmoe/` on the remote
host. Route-distribution runs used the recorder; throughput runs did not,
because recording measurably perturbs decode.

## Target-verify expert distribution

| Request batch | Verify tokens/pass (mean/max) | Active experts/layer (mean) | Hottest expert tokens (mean/p90/max) | Static-16 coverage | Lag-1 Top-16 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 28.6 / 32 | 95.5 | 12.0 / 19 / 27 | 7.2% | 32.6% |
| 16 | 56.8 / 64 | 123.3 | 23.8 / 38 / 57 | 7.7% | 35.9% |
| 32 | 98.5 / 128 | 118.8 | 41.7 / 75 / 94 | 7.1% | 43.9% |

For full 128-token target-verification passes at batch 32, an average layer
activates 145.2 experts. The hottest expert receives 52.2 tokens on average.
Experts with at least four tokens cover 88.8% of assignments. Static uniform
placement therefore wastes nearly all of the 16 resident slots, while recent
route history is predictive enough to make online replacement meaningful.

Source artifacts: `analysis/b{8,16,32}_g16.json`.

## Transfer and compute break-even

One TP2 BF16 expert shard is 9,437,184 bytes per GPU. Local-NUMA pinned H2D is
0.356-0.359 ms per expert, or about 24.5 GiB/s. Simultaneous transfers to both
GPUs sustain roughly the same per-GPU bandwidth, so the two root complexes do
not contend materially.

An isolated real KT BF16 expert on layer 0 costs approximately 0.32, 0.42,
0.99, 4.36, and 8.48 ms for 1, 2, 4, 8, and 16 tokens respectively. The GPU
kernel plus ideal PCIe transfer is about 0.61 ms, giving an optimistic isolated
break-even around four tokens.

There is an important KT-specific trap: CPU experts are stored in an
AMX-optimized packed layout. Reconstructing one expert for both TP ranks with
`submit_write_weight_scale_to_buffer` takes 6.06 ms on average; including H2D
takes 6.69 ms. A naive online implementation would therefore move the
break-even to roughly 12-16 tokens and lose most of the paper's benefit.

Qwen3.5 also retains the original packed BF16 tensors in its safetensors
checkpoint. The implemented fast path mmaps `gate_up_proj` and `down_proj`,
slices the local TP shard directly into pinned memory, and bypasses the AMX
inverse transform. With a 16-expert hot working set, standalone staging of both
TP shards is 0.17 ms at eight benchmark threads. The server keeps its existing
single PyTorch CPU thread because raising it interferes with the KT worker
pool. In the conservative live run, one all-layer promotion event averaged
62.72 ms of rank-0 checkpoint staging and 105.46 ms end to end.

Four layer-0 experts (IDs 0, 1, 128, and 255) on both TP ranks matched the KT
C++ reconstruction element for element. The live path also matched the actual
initial SGLang GPU weights exactly on both ranks for layers 0, 24, and 47.

Source artifacts: `microbench/pcie_*`, `compute_break_even_layer0.json`,
`promotion_pipeline_layer0.json`, and `raw_promotion_layer0_*.json`.

## Online mechanism

The experimental implementation:

1. Captures a persistent per-layer expert-count buffer in each target CUDA
   graph replay.
2. Reads all 48 count vectors once after target verification and maintains a
   CPU EMA of expert load.
3. Replaces at most one eligible GPU slot per layer. It fills during the first
   16 verification steps, then permits replacement only every 16 steps. A
   token floor, score-ratio hysteresis, and minimum residency age suppress
   marginal or oscillating changes.
4. Loads the selected BF16 shard from the original safetensors mmap into a
   double-buffered pinned staging area, copies it on a dedicated CUDA stream,
   then updates GPU and KT CPU routing maps in place.
5. Keeps tensor addresses stable so captured graphs remain valid. TP0 sends
   one decision tensor to TP1 before both ranks update the corresponding slot.
6. Backs up the initial resident GPU weights in CPU RAM. Same-process `static`
   mode restores weights and every CPU/CUDA/KT mapping without checkpoint I/O.

The mechanism supports both legacy and v2 EAGLE workers. A control file can
switch a running server between `static` and `dynamic`, enabling crossover A/B
runs without reloading CPU weights or recapturing graphs.

This is the KT-compatible subset of SpecMoE. Qwen3.5's existing MTP layer
remains the self-assisted draft; the change concerns target expert residency.
Cold experts continue to execute on KT CPU rather than being migrated on
demand, and the paper's affinity-aware training loss is not reproduced.

## Residency behavior

The refresh gate is essential. The aggressive version attempted replacement
on every verification pass. The conservative version fills once, continuously
updates its EMA, and only considers further replacement every 16 passes.

| Policy | Promotion passes | Total promotions | Full-pass resident coverage | Rank-0 staging per promotion pass | Total update per promotion pass | No-promotion update p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive | 133 / 143 | 4,909 | 28.2% | 92.41 ms | 132.16 ms | 26.63 ms |
| Conservative | 23 / 145 | 1,038 | 32.4% | 62.72 ms | 105.46 ms | 2.98 ms |

For the conservative policy, full-batch coverage p10/p50/p90 is
24.0%/31.2%/41.2%, and assignment-weighted coverage across every pass is
30.3%. It transferred 9.12 GiB per rank during the complete dynamic phase,
versus 43.15 GiB for the aggressive policy.

Source artifacts: `decode-hot/control-node1-conservative.jsonl` and
`decode-hot/control-node1-aggressive.jsonl`.

## Controlled end-to-end A/B

Each phase used two warmup runs followed by three synchronized measurements:
32 requests, 32 prompt tokens, 64 generated tokens, and greedy decoding. The
server remained in one process for each `static,dynamic,static` crossover.

| Policy | Static before | Dynamic | Static after | Delta vs mean static | Dynamic NVMe read |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggressive | 112.14 tok/s | 106.55 tok/s | 112.57 tok/s | -5.17% | 40.99-80.44 MiB/s |
| Conservative | 114.41 tok/s | 124.38 tok/s | 115.72 tok/s | **+8.09%** | 0.000-0.002 MiB/s |

The conservative measurements were:

- Static before: 113.97, 114.84, and 114.40 tok/s.
- Dynamic: 123.34, 123.80, and 126.00 tok/s.
- Static after: 115.29, 116.01, and 115.86 tok/s.

Every dynamic measurement exceeds every static measurement. The reported
8.09% is `124.378 / mean(114.407, 115.723) - 1`. The first static interval
overlapped aggregate node-0 CPU activity (62.2% host CPU busy); excluding that
interval and comparing against the remaining five 25%-busy static samples
still gives +7.89%. The KT pools themselves remained isolated on node 1.

The aggressive policy loses because it performs 4.7x as many promotions,
thrashes checkpoint-backed mmap pages, and spends update time on nearly every
target pass. The conservative policy amortizes the initial fill during warmup
and keeps checkpoint pages resident, so its measured dynamic intervals show no
material NVMe reads.

The initial backup occupies 6.75 GiB per TP rank. The final conservative reset
restored 708 changed slots from RAM in 1.406 s, corresponding to 6.22 GiB per
rank. The historical formal-run profile over-counted this one reset byte field
by 16x; the instrumentation now counts one slot per change.

Source artifacts:

- Conservative: `control-ab-node1-conservative/summary.json`,
  `services/qwen35-nextn-control-node1/server-conservative.log`.
- Aggressive: `control-ab-node1/summary.json`,
  `services/qwen35-nextn-control-node1/server-aggressive.log`.

## Correctness status and limits

Raw checkpoint staging has exact BF16 equality against both KT reconstruction
and sampled initial SGLang GPU weights. Static restoration uses the initial GPU
weights themselves as its source and validates restored mappings and changed
slots on both TP ranks. A post-run validation smoke restored and compared 246
changed slots successfully; its corrected profile reports 2,321,547,264 bytes
per rank and 2.031 s including the exact D2H comparisons.

Greedy output text is not bitwise invariant across residency phases. Dynamic
runs matched 31.2-46.9% of the first static phase's 32 output hashes; the final
static phase matched 40.6%, although all three final-static runs agreed with
each other. This remains unresolved and means the current implementation is a
performance research prototype, not yet a production correctness-qualified
path. The likely boundary is numerical/history sensitivity across the KT CPU
and Triton GPU expert paths, but the experiment does not establish a cause.

Early cross-restart throughput artifacts are also retained, but are not used
for conclusions because KT CPU/NUMA warmup and shared-host load caused nominally
identical runs to range from 60.50 to 189.75 tok/s. Only the controlled
same-process crossover above is considered defensible.

Validation artifact: `control-ab-node1-validate2/summary.json` and
`decode-hot/control-node1-validate.jsonl`.
