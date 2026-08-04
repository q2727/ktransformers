#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/experiments/artifacts/mooncake-toolagent}"
DATASET_ROOT="${DATASET_ROOT:-${BENCH_ROOT}/datasets}"
DATA_ROOT="${DATA_ROOT:-${BENCH_ROOT}/results/qwen3.5-122b-a10b}"
AIPERF="${AIPERF:-/data/qinchong/venvs/aiperf/bin/aiperf}"
TRACE="${TRACE:-${DATASET_ROOT}/toolagent_fullctx_256.jsonl}"
MODEL="${MODEL:-Qwen3.5-122B-A10B}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-122B-A10B}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-0000000000000000000000000000000000000001}"
URL="${URL:-http://127.0.0.1:30005}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8}"
REQUEST_COUNT="${REQUEST_COUNT:-256}"
CONFIG_TAG="${CONFIG_TAG:-g32_131k_r8}"
CPU_SET="${CPU_SET:-60-63,124-127,188-191,252-255}"

export HF_HOME="${HF_HOME:-/data/qinchong/hf-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${DATA_ROOT}"
STATUS="${DATA_ROOT}/batch_curve_${CONFIG_TAG}.status.tsv"
touch "${STATUS}"

record_status() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" >> "${STATUS}"
}

for concurrency in ${CONCURRENCIES}; do
  run_name="batch256_${CONFIG_TAG}_c${concurrency}"
  out_dir="${DATA_ROOT}/${run_name}"
  summary="${out_dir}/profile_export_aiperf.json"

  if [[ -s "${summary}" ]] && [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -eq "${REQUEST_COUNT}" ]]; then
    record_status "${run_name}" "skipped" "complete artifacts already exist"
    continue
  fi

  rm -rf "${out_dir}"
  record_status "${run_name}" "starting" "requests=${REQUEST_COUNT} concurrency=${concurrency}"

  flush_result="$(curl -fsS -X POST "${URL}/flush_cache")"
  record_status "${run_name}" "cache_flushed" "${flush_result//$'\n'/ }"

  set +e
  taskset -c "${CPU_SET}" "${AIPERF}" profile \
    --model "${MODEL}" \
    --url "${URL}" \
    --endpoint-type chat \
    --streaming \
    --use-legacy-max-tokens \
    --use-server-token-count \
    --tokenizer "${TOKENIZER}" \
    --tokenizer-revision "${TOKENIZER_REVISION}" \
    --tokenizer-trust-remote-code \
    --custom-dataset-type mooncake-trace \
    --input-file "${TRACE}" \
    --no-fixed-schedule \
    --concurrency "${concurrency}" \
    --request-count "${REQUEST_COUNT}" \
    --extra-inputs '{"ignore_eos":true,"temperature":0}' \
    --random-seed 42 \
    --workers-max "${concurrency}" \
    --output-artifact-dir "${out_dir}" \
    --export-level records \
    --gpu-telemetry pynvml \
    > "${DATA_ROOT}/${run_name}.console.log" 2>&1
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    record_status "${run_name}" "failed" "aiperf_exit=${rc}"
    exit "${rc}"
  fi
  if [[ ! -s "${summary}" ]] || [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -ne "${REQUEST_COUNT}" ]]; then
    record_status "${run_name}" "failed" "artifact validation failed"
    exit 1
  fi

  record_status "${run_name}" "completed" "records=${REQUEST_COUNT}"
done

record_status "all" "completed" "concurrencies=${CONCURRENCIES}"
