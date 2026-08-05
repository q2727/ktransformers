# Official SSD draft-engine integration

Date: 2026-08-06

## Integrated path

- Target: KTransformers/SGLang, `Qwen3-30B-A3B`, CPU-offloaded routed experts.
- Draft: pinned `tanishqkumar/ssd` commit
  `d7eb8fa0edb77a6d0876af1903367b9bb82f54e7`, `Qwen3-0.6B`.
- Placement: one RTX 4090, CUDA MPS 50% target / 50% draft.
- Workload: B=1, greedy, K=5, 128 generated tokens.
- IPC: persistent binary Unix-domain socket; draft KV, outcome logits, and
  outcome tree remain in the draft process.

The service uses the official model implementation, FlashInfer paged KV,
CUDA-graph glue decode, recovery-token forking, tree decode, tensor outcome
cache, and JIT miss path. Only the official engine's mandatory two-rank NCCL
request loop is replaced so target and draft can be independent MPS clients on
one physical GPU.

## Correctness gates

- 8 protocol and legacy-client unit tests passed.
- Real-GPU session smoke passed:
  `INIT -> BUILD -> SELECT miss/JIT -> BUILD` with persistent paged KV.
- Official F=1 output matched the legacy SGLang-draft SSD output for 128/128
  token ids.
- Official F=4 output also matched F=1 and legacy for 128/128 token ids.
- Neither official run logged a protocol fallback, exception, or draft-state
  error.

Pure KT one-token decode diverged from both SSD backends at token 49. Because
official and legacy SSD remain identical, this is not caused by the official
drafter integration. It is the existing CPU-MoE numerical-path difference
between target verify (M=K+1) and ordinary decode (M=1).

## End-to-end result

| Backend | F | Outcome branches | Cache hits / selections | Hit rate | Draft wait | Target verify | E2E | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy SGLang draft | 1 | 6 | 21 / 55 | 38.2% | 995.2 ms | 2851.6 ms | 4.040 s | 31.68 tok/s |
| Official SSD draft | 1 | 6 | 19 / 53 | 35.8% | 480.7 ms | 2746.4 ms | 4.039 s | 31.69 tok/s |
| Official SSD draft | 4 | 24 | 37 / 53 | 69.8% | 279.6 ms | 2755.0 ms | 3.248 s | 39.41 tok/s |

For steady-state official F=4:

- outcome tree total: about 21.0--21.4 ms;
- glue decode: about 3.8--3.9 ms;
- five tree-decode steps: about 17.1--17.4 ms;
- tensor-cache population: about 0.08--0.11 ms;
- Unix IPC/control overhead around BUILD: about 0.37--0.47 ms.

The 24-branch draft work remains well below the roughly 51 ms target-verify
window, so it is normally hidden by spatial overlap. Compared with official
F=1, F=4 reduced E2E latency by 19.6% and increased throughput by 24.4% on this
request. Compared with the full-GPU pure-KT timing (3.951 s), official F=4 was
about 21.6% higher throughput despite reserving half the SMs for the draft.

These are integration gates from one deterministic prompt, not final paper
aggregate numbers. The next performance table should use the paper-aligned
multi-prompt workload and report medians/confidence intervals.
