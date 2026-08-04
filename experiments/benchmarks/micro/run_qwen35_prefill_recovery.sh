#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL="Qwen3.5-122B-A10B"
TOKENIZER="/data/qinchong/models/Qwen3.5-122B-A10B"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/artifacts/micro/qwen3.5-122b-a10b}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${REPO_ROOT}/experiments/artifacts/micro/aborted/qwen3.5-122b-a10b_20260724_064349_contaminated_by_dongjiang_ds_init}"
COMMON=(
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --chat-template-kwargs-json '{"enable_thinking":false}'
  --warmup-prompt-tokens 4096
)

curl -fsS --max-time 5 http://127.0.0.1:30005/health >/dev/null
mkdir -p "${OUTPUT_ROOT}/fragments/prefill_64k_one" \
  "${OUTPUT_ROOT}/fragments/prefill_128k_five" "${OUTPUT_ROOT}/prefill"

"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_micro.py" prefill \
  "${COMMON[@]}" \
  --prefill-sizes 65536 \
  --trials 1 \
  --output-dir "${OUTPUT_ROOT}/fragments/prefill_64k_one" \
  > "${OUTPUT_ROOT}/fragments/prefill_64k_one/console.log" 2>&1

"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_micro.py" prefill \
  "${COMMON[@]}" \
  --prefill-sizes 131072 \
  --trials 5 \
  --output-dir "${OUTPUT_ROOT}/fragments/prefill_128k_five" \
  > "${OUTPUT_ROOT}/fragments/prefill_128k_five/console.log" 2>&1

clean_partial="${ARCHIVE_ROOT}/prefill/clean_completed_before_contamination.jsonl"
raw_64="$(find "${OUTPUT_ROOT}/fragments/prefill_64k_one" -maxdepth 1 -name '*.jsonl' -print -quit)"
raw_128="$(find "${OUTPUT_ROOT}/fragments/prefill_128k_five" -maxdepth 1 -name '*.jsonl' -print -quit)"
summary_128="$(find "${OUTPUT_ROOT}/fragments/prefill_128k_five" -maxdepth 1 -name '*.summary.json' -print -quit)"
for path in "${clean_partial}" "${raw_64}" "${raw_128}" "${summary_128}"; do
  [[ -s "${path}" ]] || { echo "Missing recovery input: ${path}" >&2; exit 1; }
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_micro_records.py" \
  --input "${clean_partial}" \
  --input "${raw_64}" \
  --input "${raw_128}" \
  --summary-template "${summary_128}" \
  --output-prefix "${OUTPUT_ROOT}/prefill/prefill_formal_merged" \
  --expected-points 8 \
  --trials 5 \
  --reason "Reused 34 clean records completed before external ds_init; replaced the contaminated 64K trial and measured the previously missing 128K point after the host became idle"

env \
  MODEL="${MODEL}" \
  TOKENIZER="${TOKENIZER}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  WARMUP_PROMPT_TOKENS=4096 \
  CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}' \
  "${SCRIPT_DIR}/run_formal_micro.sh"
