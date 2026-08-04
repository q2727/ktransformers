#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO}/experiments/artifacts/mooncake-toolagent}"
DATASET_ROOT="${DATASET_ROOT:-${BENCH_ROOT}/datasets}"
DATA_ROOT="${DATA_ROOT:-${BENCH_ROOT}/results/qwen3-coder-next-fp8}"
TRACE_ROOT="${TRACE_ROOT:-${BENCH_ROOT}/traces/qcn_curve}"
RATES="${RATES:-0.10 0.12 0.14}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-0000000000000000000000000000000000000002}"
CURVE_STATUS="${DATA_ROOT}/poisson_curve_qcn_fp8_g100_256k_m378_r4_tp2.status.tsv"

cd "${REPO}"
mkdir -p "${DATA_ROOT}" "$(dirname -- "${TRACE_ROOT}")"
: > "${CURVE_STATUS}"

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "${CURVE_STATUS}"
}

record_status starting "rates=${RATES} requests_per_rate=256 arrival=poisson"
"${REPO}/.venv/bin/python" "${SCRIPT_DIR}/prepare_mooncake_openloop.py" \
  --source "${DATASET_ROOT}/toolagent_trace.jsonl" \
  --output-dir "${TRACE_ROOT}" \
  --start-index 11419 \
  --request-count 256 \
  --warmup-count 128 \
  --warmup-output-length 1 \
  --rates ${RATES} \
  > "${TRACE_ROOT}.manifest.stdout.json"
record_status traces_ready "trace_root=${TRACE_ROOT}"

for rate in ${RATES}; do
  tag="$(printf '%.3f' "${rate}" | sed 's/0*$//;s/\.$//;s/\./p/')"
  run_name="poisson_n256_qcn_fp8_g100_256k_m378_r4_tp2_r${tag}"
  out_dir="${DATA_ROOT}/${run_name}"
  if [[ -s "${out_dir}/profile_export_aiperf.json" ]] \
    && [[ -s "${out_dir}/profile_export.jsonl" ]] \
    && [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -eq 256 ]]; then
    record_status skipped "rate=${rate} complete_artifacts_exist"
    continue
  fi

  record_status rate_starting "rate=${rate}"
  env \
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
  record_status rate_completed "rate=${rate}"
done

record_status completed "rates=${RATES}"
