# Serving microbenchmarks

`benchmark_micro.py` produces the two controlled serving curves used by this
experiment:

- prefill: TTFT at 1K, 2K, 4K, 8K, 16K, 32K, 64K, and 128K input;
- decode: TPOT at synchronized batch sizes 1, 2, 4, 6, and 8 with an 8K input.

The defaults use five measured trials per point. The benchmark requests exact
output lengths (`ignore_eos`), reads token counts from the server's usage record,
and calls `/flush_cache` before each measured trial. Raw per-request JSONL, a
summary JSON, and a summary CSV are written under the requested output directory.

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

The local tokenizer and the server receive the same chat-template keyword
arguments. Always record the exact launch configuration beside the results.
