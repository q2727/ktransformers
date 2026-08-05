# Same-GPU SSD scheduler experiment

This directory runs a first functional Saguaro/SSD scheduler on one physical
GPU with two resident CUDA clients:

- KTransformers target: Qwen3-30B-A3B, MPS 82% (104 effective SMs).
- Official SSD `DraftRunner`: Qwen3-0.6B, MPS 18% (22 effective SMs).
- Default SSD parameters: draft length K=8, recovery fan-out F=1, and K+1
  target verification tokens. Set `SSD_DRAFT_LENGTH` and `SSD_FAN_OUT` at
  launch time to change them; the official tree graph follows the sum of the
  K+1 per-position fan-outs.

`start_ssd.sh` starts the MPS daemon, then the official SSD draft service, then
the target server. The draft service is a separate CUDA client and uses a
binary Unix-domain-socket protocol. Draft KV, outcome logits, and the tensor
outcome cache never leave that process. `stop_ssd.sh` only stops processes
recorded in the run directory. The start script refuses to run when the
selected GPU already has compute clients or the target serving port is
occupied.

Batch-invariant deterministic inference is enabled by default for the target,
where it stabilizes prefill and tree verification. The official draft engine
uses its own greedy sampler and CUDA graphs. `DRAFT_DETERMINISTIC_INFERENCE`
only applies when `SSD_DRAFT_BACKEND=sglang`.

The target-side worker and official drafter implement the SSD state machine:

1. Look up the next draft by `(accepted_draft_count, recovery_token)`.
2. Run the official drafter's in-place JIT path on a miss.
3. Run the official glue decode and branch-tree CUDA graphs on its paged KV;
   retain all `(request, accepted_length, recovery_token)` entries GPU-side.
4. Verify the current linear path with SGLang's target-only tree verifier while
   that independent official SSD process builds the next outcome tree.

The canonical path does not require branch-KV copying. Official SSD's glue
decode writes the current recovery plus K-token path into canonical paged-KV
positions; the outcome tree uses separate lookahead slots. The next SELECT
either reads the matching tensor-cache entry or repairs the same canonical KV
with JIT. Target verification remains the source of truth.

Current implementation boundary: one active greedy text request, no grammar,
TP=1, and one official draft session. K and per-position fan-out are CLI
configurable, although the provided launch point is K=8/F=1.

Initialize the pinned official engine once:

```bash
git submodule update --init third_party/ssd
cd third_party/ssd && uv sync --frozen && cd ../..
```

Example:

```bash
experiments/ssd_scheduler/start_ssd.sh
python experiments/ssd_scheduler/benchmark.py \
  --label ssd --output experiments/artifacts/ssd-scheduler/ssd.json
experiments/ssd_scheduler/stop_ssd.sh
```

The official engine is the default. To retain the previous generic SGLang
draft server for A/B comparisons, launch with `SSD_DRAFT_BACKEND=sglang`.

For the K=5/F=1 comparison, launch SSD with:

```bash
SSD_DRAFT_LENGTH=5 SSD_FAN_OUT=1 experiments/ssd_scheduler/start_ssd.sh
```

`start_baseline.sh` launches the same KTransformers target without a draft
server or speculative decoding and gives it the full GPU. Use
`stop_baseline.sh` to stop that baseline server.

To launch both CUDA clients under Nsight Systems with collection gated by the
servers' `/start_profile` and `/stop_profile` endpoints, set for example
`NSYS_PROFILE_PREFIX=experiments/artifacts/ssd-scheduler/nsys/ssd`. The two
reports are written as `<prefix>_target.nsys-rep` and
`<prefix>_draft.nsys-rep`; the draft report also contains per-forward NVTX
ranges from the profiling hook.

For correctness comparisons, run the benchmark against the existing local
`STANDALONE` worker with the same target, draft, K, and greedy prompts, then
use `compare_outputs.py`. With deterministic target inference enabled, both
the initial and optimized implementations matched that reference for 192/192
tested output tokens. On this KTransformers branch, batched `TARGET_VERIFY`
and ordinary one-token
decode can still choose different greedy tokens because the offloaded CPU-MoE
GEMMs use different M=9 and M=1 numerical paths; the batch-invariant SGLang
mode does not cover the external CPU kernel. The existing `STANDALONE` path is
therefore the relevant scheduler integration reference, while ordinary-decode
equivalence remains a separate KT kernel issue. In the first official-engine
gate, F=1 and F=4 both matched the legacy SSD path for 128/128 tokens. At K=5
and a 50/50 MPS split, F=4 built 24-branch outcome trees in about 21 ms and
increased cache hits from 19/53 to 37/53 on that request; end-to-end latency
fell from 4.04 s to 3.25 s.
