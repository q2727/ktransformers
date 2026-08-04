#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/experiments/artifacts/mooncake-toolagent}"
DATASET_ROOT="${DATASET_ROOT:-${BENCH_ROOT}/datasets}"

: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to a local path or Hugging Face repository}"
: "${DATA_ROOT:?Set DATA_ROOT to the result directory for this model}"
: "${CONFIG_TAG:?Set CONFIG_TAG to the exact server configuration tag}"
: "${CAPACITY_DIR:?Set CAPACITY_DIR to the completed closed-loop c8 result}"

CAPACITY_FILE="${CAPACITY_FILE:-${CAPACITY_DIR}/capacity_req_s.txt}"
TRACE_ROOT="${TRACE_ROOT:-${BENCH_ROOT}/traces/${CONFIG_TAG}_slo_rates}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
REQUEST_COUNT="${REQUEST_COUNT:-128}"
WARMUP_COUNT="${WARMUP_COUNT:-16}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-4}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-128}"
WORKERS_MAX="${WORKERS_MAX:-32}"
FRACTIONS="${FRACTIONS:-0.55 0.95}"
SOURCE_START_INDEX="${SOURCE_START_INDEX:-11419}"
STATUS="${DATA_ROOT}/poisson_slo_rates_n${REQUEST_COUNT}_${CONFIG_TAG}.status.tsv"
MANIFEST="${DATA_ROOT}/poisson_slo_rates_n${REQUEST_COUNT}_${CONFIG_TAG}.json"

if [[ ! -s "${CAPACITY_FILE}" ]]; then
  echo "Closed-loop capacity file is missing: ${CAPACITY_FILE}" >&2
  exit 1
fi

capacity="$(tr -d '[:space:]' < "${CAPACITY_FILE}")"
read -r -a fraction_values <<< "${FRACTIONS}"
rates=()
for fraction in "${fraction_values[@]}"; do
  rate="$(awk -v capacity="${capacity}" -v fraction="${fraction}" 'BEGIN { printf "%.3f", capacity * fraction }')"
  rates+=("${rate}")
done

mkdir -p "${DATA_ROOT}" "${TRACE_ROOT}"
: > "${STATUS}"
record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "${STATUS}"
}

record_status starting \
  "capacity_req_s=${capacity} fractions=${FRACTIONS} rates=${rates[*]} requests_per_rate=${REQUEST_COUNT}"

cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/prepare_mooncake_openloop.py" \
  --source "${DATASET_ROOT}/toolagent_trace.jsonl" \
  --output-dir "${TRACE_ROOT}" \
  --start-index "${SOURCE_START_INDEX}" \
  --request-count "${REQUEST_COUNT}" \
  --warmup-count "${WARMUP_COUNT}" \
  --warmup-output-length 1 \
  --rates "${rates[@]}" \
  > "${TRACE_ROOT}.manifest.stdout.json"

"${REPO_ROOT}/.venv/bin/python" - \
  "${MANIFEST}" "${capacity}" "${FRACTIONS}" "${rates[*]}" \
  "${CONFIG_TAG}" "${REQUEST_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

path, capacity, fractions, rates, config_tag, request_count = sys.argv[1:]
payload = {
    "closed_loop_capacity_requests_per_second": float(capacity),
    "fractions": [float(value) for value in fractions.split()],
    "request_rates_per_second": [float(value) for value in rates.split()],
    "rate_rounding": "nearest 0.001 requests/second",
    "config_tag": config_tag,
    "request_count_per_rate": int(request_count),
    "arrival_pattern": "poisson",
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

for index in "${!rates[@]}"; do
  rate="${rates[$index]}"
  fraction="${fraction_values[$index]}"
  tag="$(printf '%.3f' "${rate}" | sed 's/0*$//;s/\.$//;s/\./p/')"
  run_name="poisson_n${REQUEST_COUNT}_${CONFIG_TAG}_r${tag}"
  out_dir="${DATA_ROOT}/${run_name}"
  if [[ -s "${out_dir}/profile_export_aiperf.json" ]] \
    && [[ -s "${out_dir}/profile_export.jsonl" ]] \
    && [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -eq "${REQUEST_COUNT}" ]]; then
    record_status skipped "fraction=${fraction} rate=${rate} complete_artifacts_exist"
    continue
  fi

  record_status rate_starting "fraction=${fraction} rate=${rate}"
  env \
    DATA_ROOT="${DATA_ROOT}" \
    TRACE_ROOT="${TRACE_ROOT}" \
    MODEL="${MODEL}" \
    TOKENIZER="${TOKENIZER}" \
    TOKENIZER_REVISION="${TOKENIZER_REVISION}" \
    REQUEST_RATE="${rate}" \
    REQUEST_COUNT="${REQUEST_COUNT}" \
    WARMUP_COUNT="${WARMUP_COUNT}" \
    WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY}" \
    CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY}" \
    WORKERS_MAX="${WORKERS_MAX}" \
    CONFIG_TAG="${CONFIG_TAG}" \
    "${SCRIPT_DIR}/run_mooncake_poisson.sh"
  record_status rate_completed "fraction=${fraction} rate=${rate}"
done

record_status completed "fractions=${FRACTIONS} rates=${rates[*]}"
