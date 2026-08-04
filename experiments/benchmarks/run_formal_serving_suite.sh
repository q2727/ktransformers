#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MICRO_DIR="${SCRIPT_DIR}/micro"
MOONCAKE_DIR="${SCRIPT_DIR}/mooncake"
BENCH_ROOT="${REPO_ROOT}/experiments/artifacts/mooncake-toolagent"

: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to a local model/tokenizer directory}"
: "${SLUG:?Set SLUG for artifact directories}"
: "${CONFIG_TAG:?Set CONFIG_TAG to the exact server configuration}"

WARMUP_PROMPT_TOKENS="${WARMUP_PROMPT_TOKENS:-1024}"
AIPERF_TOKENIZER="${AIPERF_TOKENIZER:-${TOKENIZER}}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
[[ -n "${CHAT_TEMPLATE_KWARGS_JSON}" ]] || CHAT_TEMPLATE_KWARGS_JSON='{}'
REQUEST_COUNT=128
WARMUP_COUNT="${WARMUP_COUNT:-4}"
DATA_ROOT="${BENCH_ROOT}/results/${SLUG}"
MICRO_ROOT="${REPO_ROOT}/experiments/artifacts/micro/${SLUG}"
CAPACITY_DIR="${DATA_ROOT}/capacity_n${REQUEST_COUNT}_${CONFIG_TAG}_c8"
TRACE_ROOT="${BENCH_ROOT}/traces/${CONFIG_TAG}_slo_rates_n128"

mkdir -p "${DATA_ROOT}" "${MICRO_ROOT}"

env \
  MODEL="${MODEL}" \
  TOKENIZER="${TOKENIZER}" \
  OUTPUT_ROOT="${MICRO_ROOT}" \
  WARMUP_PROMPT_TOKENS="${WARMUP_PROMPT_TOKENS}" \
  CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON}" \
  CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE}" \
  TRIALS=1 \
  "${MICRO_DIR}/run_formal_micro.sh"

if [[ ! -s "${CAPACITY_DIR}/profile_export_aiperf.json" ]] \
  || [[ ! -s "${CAPACITY_DIR}/profile_export.jsonl" ]] \
  || [[ "$(wc -l < "${CAPACITY_DIR}/profile_export.jsonl" 2>/dev/null || printf 0)" -ne 128 ]]; then
  env \
    MODEL="${MODEL}" \
    TOKENIZER="${AIPERF_TOKENIZER}" \
    TOKENIZER_REVISION='' \
    TOKENIZER_ONLINE=1 \
    DATA_ROOT="${DATA_ROOT}" \
    CONFIG_TAG="${CONFIG_TAG}" \
    REQUEST_COUNT="${REQUEST_COUNT}" \
    WARMUP_COUNT="${WARMUP_COUNT}" \
    CONCURRENCY=8 \
    "${MOONCAKE_DIR}/run_mooncake_capacity.sh"
fi

env \
  MODEL="${MODEL}" \
  TOKENIZER="${AIPERF_TOKENIZER}" \
  TOKENIZER_REVISION='' \
  TOKENIZER_ONLINE=1 \
  DATA_ROOT="${DATA_ROOT}" \
  CONFIG_TAG="${CONFIG_TAG}" \
  CAPACITY_DIR="${CAPACITY_DIR}" \
  TRACE_ROOT="${TRACE_ROOT}" \
  REQUEST_COUNT="${REQUEST_COUNT}" \
  WARMUP_COUNT="${WARMUP_COUNT}" \
  "${MOONCAKE_DIR}/run_mooncake_slo_rates.sh"

manifest="${DATA_ROOT}/poisson_slo_rates_n128_${CONFIG_TAG}.json"
mapfile -t rates < <("${REPO_ROOT}/.venv/bin/python" - "${manifest}" <<'PY'
import json
import sys
for rate in json.load(open(sys.argv[1]))["request_rates_per_second"]:
    print(f"{rate:.3f}")
PY
)
if [[ "${#rates[@]}" -ne 2 ]]; then
  echo "Expected two Poisson rates in ${manifest}" >&2
  exit 1
fi

rate_tag() {
  printf '%.3f' "$1" | sed 's/0*$//;s/\.$//;s/\./p/'
}
low_dir="${DATA_ROOT}/poisson_n128_${CONFIG_TAG}_r$(rate_tag "${rates[0]}")"
high_dir="${DATA_ROOT}/poisson_n128_${CONFIG_TAG}_r$(rate_tag "${rates[1]}")"
"${REPO_ROOT}/.venv/bin/python" "${MOONCAKE_DIR}/make_formal_selection.py" \
  --model "${MODEL}" \
  --closed-loop "${CAPACITY_DIR}" \
  --poisson "${rates[0]}:${low_dir}" \
  --poisson "${rates[1]}:${high_dir}" \
  --note "Direct 128-request runs at 55% and 95% of measured closed-loop c8 capacity." \
  --output "${DATA_ROOT}/formal_selection.json"
