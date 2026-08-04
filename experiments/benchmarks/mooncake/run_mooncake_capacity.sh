#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/experiments/artifacts/mooncake-toolagent}"
DATASET_ROOT="${DATASET_ROOT:-${BENCH_ROOT}/datasets}"
DATA_ROOT="${DATA_ROOT:-${BENCH_ROOT}/results/qwen3-coder-next-fp8}"
AIPERF="${AIPERF:-/data/qinchong/venvs/aiperf/bin/aiperf}"
MODEL="${MODEL:-Qwen3-Coder-Next}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-Coder-Next-FP8}"
TOKENIZER_REVISION="${TOKENIZER_REVISION-0000000000000000000000000000000000000002}"
URL="${URL:-http://127.0.0.1:30005}"
REQUEST_COUNT="${REQUEST_COUNT:-128}"
CONCURRENCY="${CONCURRENCY:-8}"
WARMUP_COUNT="${WARMUP_COUNT:-16}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-4}"
WORKERS_MAX="${WORKERS_MAX:-16}"
CONFIG_TAG="${CONFIG_TAG:-qcn_fp8_g100_256k_m378_r4_tp2}"
CPU_SET="${CPU_SET:-60-63,124-127,188-191,252-255}"
GOODPUT="${GOODPUT:-time_to_first_token:20000 inter_token_latency:250 request_latency:120000}"
EXTRA_INPUTS="${EXTRA_INPUTS:-}"
if [[ -z "${EXTRA_INPUTS}" ]]; then
  EXTRA_INPUTS='{"ignore_eos":true,"temperature":0}'
fi

MEASURE_TRACE="${MEASURE_TRACE:-${DATASET_ROOT}/toolagent_fullctx_256.jsonl}"
WARM_TRACE="${WARM_TRACE:-${BENCH_ROOT}/traces/poisson_256/toolagent_openloop_warm128_r0p08.jsonl}"
RUN_NAME="capacity_n${REQUEST_COUNT}_${CONFIG_TAG}_c${CONCURRENCY}"
WARM_DIR="${DATA_ROOT}/${RUN_NAME}_warmup"
OUT_DIR="${DATA_ROOT}/${RUN_NAME}"
STATUS="${DATA_ROOT}/${RUN_NAME}.status.tsv"
ACTIVE_AIPERF_PID=""

export HF_HOME="${HF_HOME:-/data/qinchong/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ "${TOKENIZER}" = /* || "${TOKENIZER_ONLINE:-0}" == "1" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

tokenizer_args=(--tokenizer "${TOKENIZER}" --tokenizer-trust-remote-code)
if [[ -n "${TOKENIZER_REVISION}" ]]; then
  tokenizer_args+=(--tokenizer-revision "${TOKENIZER_REVISION}")
fi

mkdir -p "${DATA_ROOT}"
: > "${STATUS}"

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "${STATUS}"
}

stop_aiperf_group() {
  local pgid="${1:-}"
  [[ -n "${pgid}" ]] || return 0
  if ps -eo pgid= | awk -v pgid="${pgid}" '$1 == pgid { found=1 } END { exit !found }'; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! ps -eo pgid= | awk -v pgid="${pgid}" '$1 == pgid { found=1 } END { exit !found }'; then
        return 0
      fi
      sleep 0.25
    done
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
}

cleanup() {
  if [[ -n "${ACTIVE_AIPERF_PID}" ]]; then
    stop_aiperf_group "${ACTIVE_AIPERF_PID}"
  fi
}
trap cleanup EXIT INT TERM

run_aiperf() {
  local trace="$1"
  local count="$2"
  local concurrency="$3"
  local output="$4"
  local console="$5"
  shift 5

  rm -rf "${output}"
  setsid taskset -c "${CPU_SET}" env \
    -u TOKENIZER -u AIPERF_TOKENIZER -u TOKENIZER_REVISION -u TOKENIZER_ONLINE \
    -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    "${AIPERF}" profile \
    --model "${MODEL}" \
    --url "${URL}" \
    --endpoint-type chat \
    --streaming \
    --use-legacy-max-tokens \
    --use-server-token-count \
    "${tokenizer_args[@]}" \
    --custom-dataset-type mooncake-trace \
    --input-file "${trace}" \
    --no-fixed-schedule \
    --request-count "${count}" \
    --concurrency "${concurrency}" \
    --extra-inputs "${EXTRA_INPUTS}" \
    --random-seed 42 \
    --workers-max "${WORKERS_MAX}" \
    --output-artifact-dir "${output}" \
    --export-level records \
    "$@" \
    > "${console}" 2>&1 &
  ACTIVE_AIPERF_PID=$!

  set +e
  wait "${ACTIVE_AIPERF_PID}"
  local rc=$?
  set -e
  stop_aiperf_group "${ACTIVE_AIPERF_PID}"
  ACTIVE_AIPERF_PID=""
  return "${rc}"
}

if pgrep -af '/aiperf.*profile' >/dev/null; then
  record_status failed "another AIPerf profile is already active"
  exit 1
fi
if [[ ! -s "${WARM_TRACE}" || ! -s "${MEASURE_TRACE}" ]]; then
  record_status failed "Mooncake trace file is missing"
  exit 1
fi
curl -fsS --max-time 5 "${URL}/health" >/dev/null

record_status starting \
  "mode=closed_loop requests=${REQUEST_COUNT} concurrency=${CONCURRENCY}"
if [[ "${SKIP_CACHE_FLUSH:-0}" == "1" ]]; then
  record_status cache_flush_skipped "server launched with radix cache disabled"
else
  flush_result="$(curl -fsS -X POST "${URL}/flush_cache")"
  record_status cache_flushed "${flush_result//$'\n'/ }"
fi

record_status warmup_starting \
  "requests=${WARMUP_COUNT} concurrency=${WARMUP_CONCURRENCY} output_length=1"
if ! run_aiperf \
  "${WARM_TRACE}" "${WARMUP_COUNT}" "${WARMUP_CONCURRENCY}" \
  "${WARM_DIR}" "${DATA_ROOT}/${RUN_NAME}_warmup.console.log"; then
  record_status failed "warmup AIPerf failed"
  exit 1
fi
if [[ ! -s "${WARM_DIR}/profile_export.jsonl" ]] \
  || [[ "$(wc -l < "${WARM_DIR}/profile_export.jsonl")" -ne "${WARMUP_COUNT}" ]]; then
  record_status failed "warmup artifact validation failed"
  exit 1
fi
record_status warmup_completed "records=${WARMUP_COUNT}"

record_status measurement_starting \
  "requests=${REQUEST_COUNT} concurrency=${CONCURRENCY}"
if ! run_aiperf \
  "${MEASURE_TRACE}" "${REQUEST_COUNT}" "${CONCURRENCY}" \
  "${OUT_DIR}" "${DATA_ROOT}/${RUN_NAME}.console.log" \
  --goodput "${GOODPUT}" --gpu-telemetry pynvml; then
  record_status failed "measurement AIPerf failed"
  exit 1
fi

SUMMARY="${OUT_DIR}/profile_export_aiperf.json"
if [[ ! -s "${SUMMARY}" || ! -s "${OUT_DIR}/profile_export.jsonl" ]] \
  || [[ "$(wc -l < "${OUT_DIR}/profile_export.jsonl")" -ne "${REQUEST_COUNT}" ]]; then
  record_status failed "measurement artifact validation failed"
  exit 1
fi

capacity="$(jq -r '.request_throughput.avg' "${SUMMARY}")"
record_status completed "records=${REQUEST_COUNT} capacity_req_s=${capacity}"
printf '%s\n' "${capacity}" > "${OUT_DIR}/capacity_req_s.txt"
