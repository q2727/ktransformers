# Serving microbenchmarks

`benchmark_micro.py` produces the two controlled serving curves used by this
experiment:

- prefill: TTFT at 1K, 2K, 4K, 8K, 16K, 32K, 64K, and 128K input;
- decode: TPOT at synchronized batch sizes 1, 2, 4, 6, and 8 with an 8K input.

The formal default uses one measured trial per point to keep the full model
matrix practical. Increase `--trials` only for a targeted stability check. The
benchmark requests exact output lengths (`ignore_eos`), reads token counts from
the server's usage record, and calls `/flush_cache` before each measured trial.
Raw per-request JSONL, a summary JSON, and a summary CSV are written under the
requested output directory.
Decode batches use distinct, deterministic 8K prompts for each request so that
identical generations and identical expert-routing paths do not inflate batching
performance. Batch-size points are nested: batch 4 uses the same first four
variants as batch 8.

Decode trials submit all distinct raw 8K prompts in one streaming SGLang
`/generate` batch. This is important: separate concurrent HTTP requests can
arrive one scheduler tick apart, causing request A to decode while request B is
still prefilling and therefore mixing prefill stalls into A's TPOT. One batched
request makes all sequences enter prefill and decode together. The raw prompts
are constructed to exactly 8192 tokenizer tokens. If the server still chunks
those prefills internally, the TPOT clock starts when the last sequence emits
its first token. Tokens produced by earlier sequences before that point are
recorded but excluded. Thus every TPOT sample covers only the interval in which
all batch members are in steady-state decode.

Example:

```bash
.venv/bin/python experiments/benchmarks/micro/benchmark_micro.py prefill \
  --model Qwen3-Coder-Next \
  --tokenizer /data/qinchong/models/Qwen3-Coder-Next-FP8 \
  --chat-template-kwargs-json '{"enable_thinking":false}' \
  --output-dir experiments/artifacts/micro/qwen3-coder-next-fp8/prefill

.venv/bin/python experiments/benchmarks/micro/benchmark_micro.py decode \
  --model Qwen3-Coder-Next \
  --tokenizer /data/qinchong/models/Qwen3-Coder-Next-FP8 \
  --chat-template-kwargs-json '{"enable_thinking":false}' \
  --output-dir experiments/artifacts/micro/qwen3-coder-next-fp8/decode
```

Generate an SVG without additional Python packages:

```bash
.venv/bin/python experiments/benchmarks/micro/plot_micro.py \
  experiments/artifacts/micro/MODEL/prefill/RUN.summary.json
```

For a formal model run, use the idempotent wrapper. It skips a mode only when
it finds a complete summary with all required points:

```bash
MODEL=Qwen3.5-122B-A10B \
TOKENIZER=/data/qinchong/models/Qwen3.5-122B-A10B \
OUTPUT_ROOT=experiments/artifacts/micro/qwen3.5-122b-a10b \
WARMUP_PROMPT_TOKENS=4096 \
CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}' \
experiments/benchmarks/micro/run_formal_micro.sh
```

The local tokenizer and the server receive the same chat-template keyword
arguments. Always record the exact launch configuration beside the results.
For dynamic expert-placement tuning, `--warmup-prompt-tokens` can be set to the
server's full-GPU prefill threshold so that the measured points start from the
same warmed expert map.
