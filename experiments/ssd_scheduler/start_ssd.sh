#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers}"
TARGET_MODEL="${TARGET_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-0.6B}"
TARGET_PORT="${TARGET_PORT:-30020}"
DRAFT_PORT="${DRAFT_PORT:-30021}"
TARGET_MPS_PERCENT="${TARGET_MPS_PERCENT:-82}"
DRAFT_MPS_PERCENT="${DRAFT_MPS_PERCENT:-18}"
TARGET_DETERMINISTIC_INFERENCE="${TARGET_DETERMINISTIC_INFERENCE:-1}"
DRAFT_DETERMINISTIC_INFERENCE="${DRAFT_DETERMINISTIC_INFERENCE:-0}"
SSD_DRAFT_LENGTH="${SSD_DRAFT_LENGTH:-8}"
SSD_FAN_OUT="${SSD_FAN_OUT:-1}"
SSD_FAN_OUTS="${SSD_FAN_OUTS:-}"
NSYS_PROFILE_PREFIX="${NSYS_PROFILE_PREFIX:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-scheduler}"
RUN_DIR="${OUT_DIR}/run"
LOG_DIR="${OUT_DIR}/logs"

MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/qinchong-mps-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/qinchong-mps-log}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if (( SSD_DRAFT_LENGTH < 1 )); then
  echo "SSD_DRAFT_LENGTH must be at least 1." >&2
  exit 1
fi
SSD_VERIFY_TOKENS=$((SSD_DRAFT_LENGTH + 1))

if [[ -s "${RUN_DIR}/target.pid" || -s "${RUN_DIR}/draft.pid" ]]; then
  echo "SSD pid files already exist under ${RUN_DIR}; run stop_ssd.sh first." >&2
  exit 1
fi
if ss -ltn | grep -qE ":(${TARGET_PORT}|${DRAFT_PORT})[[:space:]]"; then
  echo "Target or draft port is already in use." >&2
  exit 1
fi
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU 0 already has compute clients; refusing to disturb them." >&2
  exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_COLOCATED_BATCH_GEN=1

TARGET_DETERMINISTIC_ARGS=()
DRAFT_DETERMINISTIC_ARGS=()
SSD_FAN_OUT_ARGS=(--speculative-ssd-fan-out "${SSD_FAN_OUT}")
if [[ "${TARGET_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  TARGET_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi
if [[ "${DRAFT_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  DRAFT_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi
if [[ -n "${SSD_FAN_OUTS}" ]]; then
  read -r -a SSD_FAN_OUT_VALUES <<<"${SSD_FAN_OUTS}"
  SSD_FAN_OUT_ARGS+=(--speculative-ssd-fan-outs "${SSD_FAN_OUT_VALUES[@]}")
fi

DRAFT_PROFILE_PREFIX=()
TARGET_PROFILE_PREFIX=()
DRAFT_PYTHONPATH="${PYTHONPATH}"
if [[ -n "${NSYS_PROFILE_PREFIX}" ]]; then
  mkdir -p "$(dirname "${NSYS_PROFILE_PREFIX}")"
  NSYS_COMMON=(
    nsys profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --cpuctxsw=none
    --cuda-graph-trace=node
    --capture-range=cudaProfilerApi
    --capture-range-end=stop-shutdown
    --force-overwrite=true
  )
  DRAFT_PROFILE_PREFIX=("${NSYS_COMMON[@]}" -o "${NSYS_PROFILE_PREFIX}_draft")
  TARGET_PROFILE_PREFIX=("${NSYS_COMMON[@]}" -o "${NSYS_PROFILE_PREFIX}_target")
  DRAFT_PYTHONPATH="${REPO_ROOT}/experiments/ssd_sm_profile/hook:${DRAFT_PYTHONPATH}"
fi

rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
nvidia-cuda-mps-control -d

stop_started_processes() {
  local pid
  for name in target draft; do
    if [[ -s "${RUN_DIR}/${name}.pid" ]]; then
      pid=$(<"${RUN_DIR}/${name}.pid")
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
}
trap stop_started_processes ERR INT TERM

wait_until_ready() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -120 "${log}" >&2
      return 1
    fi
    sleep 1
  done
  tail -120 "${log}" >&2
  return 1
}

DRAFT_LOG="${LOG_DIR}/draft.log"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${DRAFT_MPS_PERCENT}" \
SSD_PHASE_NVTX="$([[ -n "${NSYS_PROFILE_PREFIX}" ]] && echo 1 || echo 0)" \
SSD_PROFILE_ROLE=draft PYTHONPATH="${DRAFT_PYTHONPATH}" setsid \
  "${DRAFT_PROFILE_PREFIX[@]}" python -m sglang.launch_server \
    --host 127.0.0.1 --port "${DRAFT_PORT}" \
    --model "${DRAFT_MODEL}" --served-model-name Qwen3-0.6B-SSD \
    --tensor-parallel-size 1 --attention-backend triton \
    --max-total-tokens 16384 --max-running-requests 32 \
    --chunked-prefill-size 2048 --mem-fraction-static 0.20 \
    --watchdog-timeout 3000 --trust-remote-code \
    --disable-custom-all-reduce \
    "${DRAFT_DETERMINISTIC_ARGS[@]}" \
    --cuda-graph-max-bs 16 --cuda-graph-bs 1 "${SSD_VERIFY_TOKENS}" 16 \
    >"${DRAFT_LOG}" 2>&1 < /dev/null &
DRAFT_PID=$!
echo "${DRAFT_PID}" >"${RUN_DIR}/draft.pid"
wait_until_ready "${DRAFT_PORT}" "${DRAFT_PID}" "${DRAFT_LOG}"

TARGET_LOG="${LOG_DIR}/target.log"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${TARGET_MPS_PERCENT}" setsid \
  "${TARGET_PROFILE_PREFIX[@]}" python -m sglang.launch_server \
    --host 127.0.0.1 --port "${TARGET_PORT}" \
    --model "${TARGET_MODEL}" --served-model-name Qwen3-30B-A3B-SSD \
    --kt-weight-path "${TARGET_MODEL}" --kt-method BF16 \
    --kt-num-gpu-experts 0 --kt-cpuinfer 120 --kt-threadpool-count 2 \
    --kt-numa-nodes 0 1 --kt-gpu-prefill-token-threshold 2048 \
    --tensor-parallel-size 1 --attention-backend triton \
    --max-total-tokens 32768 --max-running-requests 2 \
    --chunked-prefill-size 2048 --mem-fraction-static 0.65 \
    --watchdog-timeout 3000 --disable-shared-experts-fusion \
    --trust-remote-code --enable-p2p-check --disable-custom-all-reduce \
    "${TARGET_DETERMINISTIC_ARGS[@]}" \
    --reasoning-parser qwen3 --cuda-graph-max-bs 1 --cuda-graph-bs 1 \
    --speculative-algorithm SSD \
    --speculative-num-steps "${SSD_DRAFT_LENGTH}" --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens "${SSD_VERIFY_TOKENS}" \
    --speculative-ssd-draft-server-url "http://127.0.0.1:${DRAFT_PORT}" \
    "${SSD_FAN_OUT_ARGS[@]}" \
    >"${TARGET_LOG}" 2>&1 < /dev/null &
TARGET_PID=$!
echo "${TARGET_PID}" >"${RUN_DIR}/target.pid"
wait_until_ready "${TARGET_PORT}" "${TARGET_PID}" "${TARGET_LOG}"

trap - ERR INT TERM
echo "SSD ready: target=http://127.0.0.1:${TARGET_PORT} (${TARGET_MPS_PERCENT}%), draft=http://127.0.0.1:${DRAFT_PORT} (${DRAFT_MPS_PERCENT}%)"
echo "SSD config: K=${SSD_DRAFT_LENGTH}, F=${SSD_FAN_OUT}, verify_tokens=${SSD_VERIFY_TOKENS}"
echo "Logs: ${LOG_DIR}"
