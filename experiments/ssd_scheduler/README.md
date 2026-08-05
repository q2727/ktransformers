# Same-GPU SSD scheduler experiment

This directory runs a first functional Saguaro/SSD scheduler on one physical
GPU with two resident CUDA clients:

- KTransformers target: Qwen3-30B-A3B, MPS 82% (104 effective SMs).
- SGLang draft: Qwen3-0.6B, MPS 18% (22 effective SMs).
- Default SSD parameters: draft length K=8, recovery fan-out F=1, and K+1
  target verification tokens. Set `SSD_DRAFT_LENGTH` and `SSD_FAN_OUT` at
  launch time to change them; the draft CUDA Graph batch size follows K+1.

`start_ssd.sh` starts the MPS daemon, then the radix-cached draft server, then
the target server. `stop_ssd.sh` only stops processes recorded in the run
directory. The start script refuses to run when GPU 0 already has compute
clients or either serving port is occupied.

Batch-invariant deterministic inference is enabled by default for the target,
where it stabilizes prefill and tree verification. It is disabled for the
draft because speculative correctness does not depend on draft numerics and
the batch-invariant kernels are much slower for the 9-branch workload. Override
these independently with `TARGET_DETERMINISTIC_INFERENCE` and
`DRAFT_DETERMINISTIC_INFERENCE`.

The target-side worker implements the SSD state machine used by the reference
implementation:

1. Look up the next draft by `(accepted_draft_count, recovery_token)`.
2. Fall back to an ordinary K-token draft on a miss.
3. Carry the K+1 endpoint distributions returned with every cached draft,
   select F off-path recovery tokens per endpoint, and submit all next branches
   in one asynchronous RPC before target verification.
4. Verify the current linear path with SGLang's target-only tree verifier while
   the independent draft process builds those next-round branches.

The draft server's radix cache supplies shared-prefix and copy-on-write KV
reuse across the branch batch. Generating one extra draft token materializes
the full K-token path KV and returns the final endpoint distribution, avoiding
a separate endpoint-query batch. Target verification remains the source of
truth.

Current implementation boundary: one active greedy text request, no grammar,
TP=1, and prompts no longer than one 2048-token prefill chunk. K and F are CLI
configurable, although the provided launch point is K=8/F=1.

Example:

```bash
experiments/ssd_scheduler/start_ssd.sh
python experiments/ssd_scheduler/benchmark.py \
  --label ssd --output experiments/artifacts/ssd-scheduler/ssd.json
experiments/ssd_scheduler/stop_ssd.sh
```

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
equivalence remains a separate KT kernel issue.
