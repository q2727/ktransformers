#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/experiments/artifacts/mooncake-toolagent}"
DATA_ROOT="${DATA_ROOT:-${BENCH_ROOT}/results/qwen3.5-122b-a10b}"
TRACE_ROOT="${TRACE_ROOT:-${BENCH_ROOT}/traces/poisson_256}"
AIPERF="${AIPERF:-/data/qinchong/venvs/aiperf/bin/aiperf}"
MODEL="${MODEL:-Qwen3.5-122B-A10B}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-122B-A10B}"
TOKENIZER_REVISION="${TOKENIZER_REVISION-0000000000000000000000000000000000000001}"
URL="${URL:-http://127.0.0.1:30005}"
REQUEST_RATE="${REQUEST_RATE:-0.10}"
REQUEST_COUNT="${REQUEST_COUNT:-128}"
WARMUP_COUNT="${WARMUP_COUNT:-16}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-4}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-128}"
WORKERS_MAX="${WORKERS_MAX:-32}"
CONFIG_TAG="${CONFIG_TAG:-g48_262k_m160_r8}"
CPU_SET="${CPU_SET:-60-63,124-127,188-191,252-255}"
GOODPUT="${GOODPUT:-time_to_first_token:20000 inter_token_latency:250 request_latency:120000}"
EXTRA_INPUTS="${EXTRA_INPUTS:-}"
if [[ -z "${EXTRA_INPUTS}" ]]; then
  EXTRA_INPUTS='{"ignore_eos":true,"temperature":0}'
fi

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

rate_tag="$(printf '%.3f' "${REQUEST_RATE}" | sed 's/0*$//;s/\.$//;s/\./p/')"
run_name="poisson_n${REQUEST_COUNT}_${CONFIG_TAG}_r${rate_tag}"
warm_trace="${TRACE_ROOT}/toolagent_openloop_warm${WARMUP_COUNT}_r${rate_tag}.jsonl"
measure_trace="${TRACE_ROOT}/toolagent_openloop_measure${REQUEST_COUNT}_r${rate_tag}.jsonl"
warm_dir="${DATA_ROOT}/${run_name}_warmup"
out_dir="${DATA_ROOT}/${run_name}"
STATUS="${DATA_ROOT}/${run_name}.status.tsv"
ACTIVE_AIPERF_PID=""

mkdir -p "${DATA_ROOT}"
touch "${STATUS}"

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "${STATUS}"
}

stop_aiperf_group() {
  local pgid="${1:-}"
  [[ -n "${pgid}" ]] || return 0
  if pgrep -g "${pgid}" >/dev/null 2>&1; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! pgrep -g "${pgid}" >/dev/null 2>&1; then
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
  local workers="$3"
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
    --extra-inputs "${EXTRA_INPUTS}" \
    --random-seed 42 \
    --workers-max "${workers}" \
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

if pgrep -x aiperf >/dev/null; then
  record_status "failed" "another AIPerf run is already active"
  exit 1
fi
if [[ ! -s "${warm_trace}" || ! -s "${measure_trace}" ]]; then
  record_status "failed" "Poisson trace file is missing"
  exit 1
fi
curl -fsS --max-time 5 "${URL}/health" >/dev/null

record_status "starting" \
  "rate=${REQUEST_RATE} requests=${REQUEST_COUNT} arrival=poisson client_concurrency=${CLIENT_CONCURRENCY}"
if [[ "${SKIP_CACHE_FLUSH:-0}" == "1" ]]; then
  record_status "cache_flush_skipped" "server launched with radix cache disabled"
else
  flush_result="$(curl -fsS -X POST "${URL}/flush_cache")"
  record_status "cache_flushed" "${flush_result//$'\n'/ }"
fi

record_status "warmup_starting" \
  "requests=${WARMUP_COUNT} concurrency=${WARMUP_CONCURRENCY} output_length=1"
if ! run_aiperf \
  "${warm_trace}" "${WARMUP_COUNT}" "${WARMUP_CONCURRENCY}" \
  "${warm_dir}" "${DATA_ROOT}/${run_name}_warmup.console.log" \
  --concurrency "${WARMUP_CONCURRENCY}"; then
  record_status "failed" "warmup AIPerf failed"
  exit 1
fi
if [[ ! -s "${warm_dir}/profile_export.jsonl" ]] \
  || [[ "$(wc -l < "${warm_dir}/profile_export.jsonl")" -ne "${WARMUP_COUNT}" ]]; then
  record_status "failed" "warmup artifact validation failed"
  exit 1
fi
record_status "warmup_completed" "records=${WARMUP_COUNT}"

record_status "measurement_starting" \
  "requests=${REQUEST_COUNT} rate=${REQUEST_RATE} arrival=poisson"
if ! run_aiperf \
  "${measure_trace}" "${REQUEST_COUNT}" "${WORKERS_MAX}" \
  "${out_dir}" "${DATA_ROOT}/${run_name}.console.log" \
  --request-rate "${REQUEST_RATE}" \
  --arrival-pattern poisson \
  --concurrency "${CLIENT_CONCURRENCY}" \
  --goodput "${GOODPUT}" \
  --gpu-telemetry pynvml; then
  record_status "failed" "measurement AIPerf failed"
  exit 1
fi
if [[ ! -s "${out_dir}/profile_export_aiperf.json" ]] \
  || [[ ! -s "${out_dir}/profile_export.jsonl" ]] \
  || [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -ne "${REQUEST_COUNT}" ]]; then
  record_status "failed" "measurement artifact validation failed"
  exit 1
fi
record_status "completed" "records=${REQUEST_COUNT}"
