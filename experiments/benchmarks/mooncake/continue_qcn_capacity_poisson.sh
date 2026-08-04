#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO}/experiments/artifacts/mooncake-toolagent}"
DATASET_ROOT="${DATASET_ROOT:-${BENCH_ROOT}/datasets}"
DATA_ROOT="${DATA_ROOT:-${BENCH_ROOT}/results/qwen3-coder-next-fp8}"
CAPACITY_RUN="${CAPACITY_RUN:-capacity_n256_qcn_fp8_g100_256k_m378_r4_tp2_c8}"
CAPACITY_STATUS="${DATA_ROOT}/${CAPACITY_RUN}.status.tsv"
CAPACITY_DIR="${DATA_ROOT}/${CAPACITY_RUN}"
TRACE_ROOT="${TRACE_ROOT:-${BENCH_ROOT}/traces/qcn_capacity}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-0000000000000000000000000000000000000002}"

mkdir -p "${DATA_ROOT}" "$(dirname -- "${TRACE_ROOT}")"

while true; do
  if grep -q $'\tcompleted\t' "${CAPACITY_STATUS}" 2>/dev/null; then
    break
  fi
  if grep -q $'\tfailed\t' "${CAPACITY_STATUS}" 2>/dev/null; then
    echo "capacity run failed" >&2
    exit 1
  fi
  sleep 10
done

capacity="$(<"${CAPACITY_DIR}/capacity_req_s.txt")"
rate="$(printf '%.3f' "${capacity}")"
printf '%s\n' "${rate}" > "${CAPACITY_DIR}/poisson_request_rate_req_s.txt"

cd "${REPO}"
"${REPO}/.venv/bin/python" "${SCRIPT_DIR}/prepare_mooncake_openloop.py" \
  --source "${DATASET_ROOT}/toolagent_trace.jsonl" \
  --output-dir "${TRACE_ROOT}" \
  --start-index 11419 \
  --request-count 256 \
  --warmup-count 128 \
  --warmup-output-length 1 \
  --rates "${rate}" \
  > "${TRACE_ROOT}.manifest.stdout.json"

exec env \
  DATA_ROOT="${DATA_ROOT}" \
  TRACE_ROOT="${TRACE_ROOT}" \
  MODEL=Qwen3-Coder-Next \
  TOKENIZER=Qwen/Qwen3-Coder-Next-FP8 \
  TOKENIZER_REVISION="${TOKENIZER_REVISION}" \
  REQUEST_RATE="${rate}" \
  REQUEST_COUNT=256 \
  WARMUP_COUNT=128 \
  WARMUP_CONCURRENCY=4 \
  CLIENT_CONCURRENCY=128 \
  WORKERS_MAX=32 \
  CONFIG_TAG=qcn_fp8_g100_256k_m378_r4_tp2 \
  "${SCRIPT_DIR}/run_mooncake_poisson.sh"
