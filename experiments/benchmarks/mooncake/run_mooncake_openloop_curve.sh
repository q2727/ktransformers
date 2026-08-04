#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/experiments/artifacts/mooncake-toolagent}"
DATA_ROOT="${DATA_ROOT:-${BENCH_ROOT}/results/qwen3.5-122b-a10b}"
TRACE_ROOT="${TRACE_ROOT:-${BENCH_ROOT}/traces/openloop_1024}"
AIPERF="${AIPERF:-/data/qinchong/venvs/aiperf/bin/aiperf}"
MODEL="${MODEL:-Qwen3.5-122B-A10B}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-122B-A10B}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-0000000000000000000000000000000000000001}"
URL="${URL:-http://127.0.0.1:30005}"
RATES="${RATES:-0.08 0.10 0.12}"
REQUEST_COUNT="${REQUEST_COUNT:-1024}"
WARMUP_COUNT="${WARMUP_COUNT:-256}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-128}"
WORKERS_MAX="${WORKERS_MAX:-32}"
CONFIG_TAG="${CONFIG_TAG:-g48_262k_m160_r8}"
CPU_SET="${CPU_SET:-60-63,124-127,188-191,252-255}"
GOODPUT="${GOODPUT:-time_to_first_token:20000 inter_token_latency:250 request_latency:120000}"

export HF_HOME="${HF_HOME:-/data/qinchong/hf-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${DATA_ROOT}"
STATUS="${DATA_ROOT}/openloop_curve_${CONFIG_TAG}_n${REQUEST_COUNT}.status.tsv"
touch "${STATUS}"
ACTIVE_AIPERF_PID=""

record_status() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" >> "${STATUS}"
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
  local out_dir="$3"
  local console_log="$4"
  shift 4

  rm -rf "${out_dir}"
  setsid taskset -c "${CPU_SET}" "${AIPERF}" profile \
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
    --input-file "${trace}" \
    --fixed-schedule \
    --fixed-schedule-auto-offset \
    --concurrency "${CLIENT_CONCURRENCY}" \
    --request-count "${count}" \
    --extra-inputs '{"ignore_eos":true,"temperature":0}' \
    --random-seed 42 \
    --workers-max "${WORKERS_MAX}" \
    --output-artifact-dir "${out_dir}" \
    --export-level records \
    "$@" \
    > "${console_log}" 2>&1 &
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
  record_status "all" "failed" "another AIPerf profile is already running"
  exit 1
fi
curl -fsS --max-time 5 "${URL}/health" >/dev/null

for rate in ${RATES}; do
  tag="$(printf '%.3f' "${rate}" | sed 's/0*$//;s/\.$//;s/\./p/')"
  run_name="openloop_n${REQUEST_COUNT}_${CONFIG_TAG}_r${tag}"
  warm_trace="${TRACE_ROOT}/toolagent_openloop_warm${WARMUP_COUNT}_r${tag}.jsonl"
  measure_trace="${TRACE_ROOT}/toolagent_openloop_measure${REQUEST_COUNT}_r${tag}.jsonl"
  warm_dir="${DATA_ROOT}/${run_name}_warmup"
  out_dir="${DATA_ROOT}/${run_name}"
  summary="${out_dir}/profile_export_aiperf.json"

  if [[ -s "${summary}" ]] \
    && [[ -s "${out_dir}/profile_export.jsonl" ]] \
    && [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -eq "${REQUEST_COUNT}" ]]; then
    record_status "${run_name}" "skipped" "complete artifacts already exist"
    continue
  fi
  if [[ ! -s "${warm_trace}" || ! -s "${measure_trace}" ]]; then
    record_status "${run_name}" "failed" "trace file missing"
    exit 1
  fi

  record_status "${run_name}" "starting" \
    "rate=${rate} requests=${REQUEST_COUNT} client_concurrency=${CLIENT_CONCURRENCY}"
  flush_result="$(curl -fsS -X POST "${URL}/flush_cache")"
  record_status "${run_name}" "cache_flushed" "${flush_result//$'\n'/ }"

  record_status "${run_name}" "warmup_starting" \
    "requests=${WARMUP_COUNT} fixed_schedule_rate=${rate}"
  if ! run_aiperf \
    "${warm_trace}" "${WARMUP_COUNT}" "${warm_dir}" \
    "${DATA_ROOT}/${run_name}_warmup.console.log"; then
    record_status "${run_name}" "failed" "warmup AIPerf failed"
    exit 1
  fi
  if [[ ! -s "${warm_dir}/profile_export.jsonl" ]] \
    || [[ "$(wc -l < "${warm_dir}/profile_export.jsonl")" -ne "${WARMUP_COUNT}" ]]; then
    record_status "${run_name}" "failed" "warmup artifact validation failed"
    exit 1
  fi
  record_status "${run_name}" "warmup_completed" "records=${WARMUP_COUNT}"

  record_status "${run_name}" "measurement_starting" \
    "requests=${REQUEST_COUNT} fixed_schedule_rate=${rate}"
  if ! run_aiperf \
    "${measure_trace}" "${REQUEST_COUNT}" "${out_dir}" \
    "${DATA_ROOT}/${run_name}.console.log" \
    --goodput "${GOODPUT}" \
    --gpu-telemetry pynvml; then
    record_status "${run_name}" "failed" "measurement AIPerf failed"
    exit 1
  fi
  if [[ ! -s "${summary}" ]] \
    || [[ ! -s "${out_dir}/profile_export.jsonl" ]] \
    || [[ "$(wc -l < "${out_dir}/profile_export.jsonl")" -ne "${REQUEST_COUNT}" ]]; then
    record_status "${run_name}" "failed" "measurement artifact validation failed"
    exit 1
  fi
  record_status "${run_name}" "completed" "records=${REQUEST_COUNT}"
done

record_status "all" "completed" "rates=${RATES} requests_per_rate=${REQUEST_COUNT}"
