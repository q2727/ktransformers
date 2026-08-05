#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers}"
TARGET_MODEL="${TARGET_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
TARGET_PORT="${TARGET_PORT:-30020}"
TARGET_DETERMINISTIC_INFERENCE="${TARGET_DETERMINISTIC_INFERENCE:-1}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-k5-f1-ab}"
RUN_DIR="${OUT_DIR}/run"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [[ -s "${RUN_DIR}/baseline.pid" ]]; then
  echo "Baseline pid file already exists under ${RUN_DIR}; run stop_baseline.sh first." >&2
  exit 1
fi
if ss -ltn | grep -qE ":${TARGET_PORT}[[:space:]]"; then
  echo "Target port ${TARGET_PORT} is already in use." >&2
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_COLOCATED_BATCH_GEN=1

TARGET_DETERMINISTIC_ARGS=()
if [[ "${TARGET_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  TARGET_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi

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

stop_started_process() {
  if [[ -s "${RUN_DIR}/baseline.pid" ]]; then
    local pid
    pid=$(<"${RUN_DIR}/baseline.pid")
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
}
trap stop_started_process ERR INT TERM

TARGET_LOG="${LOG_DIR}/baseline.log"
setsid python -m sglang.launch_server \
  --host 127.0.0.1 --port "${TARGET_PORT}" \
  --model "${TARGET_MODEL}" --served-model-name Qwen3-30B-A3B-KT \
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
  >"${TARGET_LOG}" 2>&1 < /dev/null &
TARGET_PID=$!
echo "${TARGET_PID}" >"${RUN_DIR}/baseline.pid"
wait_until_ready "${TARGET_PORT}" "${TARGET_PID}" "${TARGET_LOG}"

trap - ERR INT TERM
echo "KTransformers baseline ready: http://127.0.0.1:${TARGET_PORT} (full GPU)"
echo "Logs: ${LOG_DIR}"
