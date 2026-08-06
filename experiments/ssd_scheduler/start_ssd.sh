#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
TARGET_MODEL="${TARGET_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-0.6B}"
TARGET_PORT="${TARGET_PORT:-30020}"
DRAFT_PORT="${DRAFT_PORT:-30021}"
SSD_DRAFT_BACKEND="${SSD_DRAFT_BACKEND:-official}"
SSD_OFFICIAL_ROOT="${SSD_OFFICIAL_ROOT:-${REPO_ROOT}/third_party/ssd}"
SSD_OFFICIAL_PYTHON="${SSD_OFFICIAL_PYTHON:-${SSD_OFFICIAL_ROOT}/.venv/bin/python}"
SSD_DRAFT_MAX_MODEL_LEN="${SSD_DRAFT_MAX_MODEL_LEN:-8192}"
# B=1 with the default 2048-token context needs only eight 256-token KV
# blocks. Two percent leaves ample headroom (30 blocks for Qwen3-1.7B on the
# tested 48-GiB device) without reserving memory for dozens of unsupported
# concurrent requests. Increase this explicitly for longer contexts/models.
SSD_DRAFT_GPU_MEMORY_UTILIZATION="${SSD_DRAFT_GPU_MEMORY_UTILIZATION:-0.02}"
GPU_ID="${GPU_ID:-0}"
TARGET_MPS_PERCENT="${TARGET_MPS_PERCENT:-82}"
DRAFT_MPS_PERCENT="${DRAFT_MPS_PERCENT:-18}"
TARGET_MPS_CLIENT_PRIORITY="${TARGET_MPS_CLIENT_PRIORITY:-}"
DRAFT_MPS_CLIENT_PRIORITY="${DRAFT_MPS_CLIENT_PRIORITY:-}"
TARGET_DETERMINISTIC_INFERENCE="${TARGET_DETERMINISTIC_INFERENCE:-1}"
DRAFT_DETERMINISTIC_INFERENCE="${DRAFT_DETERMINISTIC_INFERENCE:-0}"
SSD_DRAFT_LENGTH="${SSD_DRAFT_LENGTH:-8}"
SSD_FAN_OUT="${SSD_FAN_OUT:-1}"
SSD_FAN_OUTS="${SSD_FAN_OUTS:-}"
SSD_DRAFT_SIDE_CACHE="${SSD_DRAFT_SIDE_CACHE:-0}"
SSD_DISABLE_OUTCOME_CACHE="${SSD_DISABLE_OUTCOME_CACHE:-0}"
DRAFT_SKIP_TOKENIZER_INIT="${DRAFT_SKIP_TOKENIZER_INIT:-0}"
DRAFT_DISABLE_OVERLAP_SCHEDULE="${DRAFT_DISABLE_OVERLAP_SCHEDULE:-1}"
DRAFT_CONTINUOUS_DECODE_STEPS="${DRAFT_CONTINUOUS_DECODE_STEPS:-}"
DRAFT_REQUEST_TIME_STATS="${DRAFT_REQUEST_TIME_STATS:-0}"
NSYS_PROFILE_PREFIX="${NSYS_PROFILE_PREFIX:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-scheduler}"
RUN_DIR="${OUT_DIR}/run"
LOG_DIR="${OUT_DIR}/logs"
SSD_DRAFT_SOCKET="${SSD_DRAFT_SOCKET:-/tmp/ktransformers-ssd-${USER}-gpu${GPU_ID}.sock}"

MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/qinchong-mps-gpu${GPU_ID}-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/qinchong-mps-gpu${GPU_ID}-log}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if (( SSD_DRAFT_LENGTH < 1 )); then
  echo "SSD_DRAFT_LENGTH must be at least 1." >&2
  exit 1
fi
if [[ "${SSD_DRAFT_BACKEND}" != "official" && "${SSD_DRAFT_BACKEND}" != "sglang" ]]; then
  echo "SSD_DRAFT_BACKEND must be either official or sglang." >&2
  exit 1
fi
SSD_VERIFY_TOKENS=$((SSD_DRAFT_LENGTH + 1))
if [[ -z "${DRAFT_CONTINUOUS_DECODE_STEPS}" ]]; then
  DRAFT_CONTINUOUS_DECODE_STEPS="${SSD_VERIFY_TOKENS}"
fi

if [[ -s "${RUN_DIR}/target.pid" || -s "${RUN_DIR}/draft.pid" ]]; then
  echo "SSD pid files already exist under ${RUN_DIR}; run stop_ssd.sh first." >&2
  exit 1
fi
PORT_PATTERN=":(${TARGET_PORT})[[:space:]]"
if [[ "${SSD_DRAFT_BACKEND}" == "sglang" ]]; then
  PORT_PATTERN=":(${TARGET_PORT}|${DRAFT_PORT})[[:space:]]"
fi
if ss -ltn | grep -qE "${PORT_PATTERN}"; then
  echo "Target or draft port is already in use." >&2
  exit 1
fi
if nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU ${GPU_ID} already has compute clients; refusing to disturb them." >&2
  exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SSD_HF_CACHE="${SSD_HF_CACHE:-${HOME}/.cache/huggingface}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-${SSD_OFFICIAL_ROOT}/data}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.9}"
export SGLANG_ENABLE_COLOCATED_BATCH_GEN=1
export SGLANG_SSD_DRAFT_SIDE_CACHE="${SSD_DRAFT_SIDE_CACHE}"
export SGLANG_SSD_DISABLE_OUTCOME_CACHE="${SSD_DISABLE_OUTCOME_CACHE}"

TARGET_DETERMINISTIC_ARGS=()
DRAFT_DETERMINISTIC_ARGS=()
DRAFT_TOKENIZER_ARGS=()
DRAFT_SCHEDULER_ARGS=(
  --num-continuous-decode-steps "${DRAFT_CONTINUOUS_DECODE_STEPS}"
)
SSD_FAN_OUT_ARGS=(--speculative-ssd-fan-out "${SSD_FAN_OUT}")
SSD_FAN_OUT_VALUES=()
SSD_BRANCH_BATCH=$((SSD_VERIFY_TOKENS * SSD_FAN_OUT))
if [[ "${TARGET_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  TARGET_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi
if [[ "${DRAFT_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  DRAFT_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi
if [[ "${DRAFT_SKIP_TOKENIZER_INIT}" == "1" ]]; then
  DRAFT_TOKENIZER_ARGS+=(--skip-tokenizer-init)
fi
if [[ "${DRAFT_DISABLE_OVERLAP_SCHEDULE}" == "1" ]]; then
  DRAFT_SCHEDULER_ARGS+=(--disable-overlap-schedule)
fi
if [[ "${DRAFT_REQUEST_TIME_STATS}" == "1" ]]; then
  DRAFT_SCHEDULER_ARGS+=(--enable-request-time-stats-logging)
fi
if [[ -n "${SSD_FAN_OUTS}" ]]; then
  read -r -a SSD_FAN_OUT_VALUES <<<"${SSD_FAN_OUTS}"
  SSD_FAN_OUT_ARGS+=(--speculative-ssd-fan-outs "${SSD_FAN_OUT_VALUES[@]}")
  SSD_BRANCH_BATCH=0
  for value in "${SSD_FAN_OUT_VALUES[@]}"; do
    SSD_BRANCH_BATCH=$((SSD_BRANCH_BATCH + value))
  done
else
  for _ in $(seq 1 "${SSD_VERIFY_TOKENS}"); do
    SSD_FAN_OUT_VALUES+=("${SSD_FAN_OUT}")
  done
fi

DRAFT_CUDA_GRAPH_MAX_BS="${DRAFT_CUDA_GRAPH_MAX_BS:-16}"
if (( SSD_BRANCH_BATCH > DRAFT_CUDA_GRAPH_MAX_BS )); then
  DRAFT_CUDA_GRAPH_MAX_BS=${SSD_BRANCH_BATCH}
fi
DRAFT_MAX_RUNNING_REQUESTS="${DRAFT_MAX_RUNNING_REQUESTS:-32}"
if (( SSD_BRANCH_BATCH > DRAFT_MAX_RUNNING_REQUESTS )); then
  DRAFT_MAX_RUNNING_REQUESTS=${SSD_BRANCH_BATCH}
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

# The MPS server remaps the single physical GPU selected above to logical
# device 0.  Clients must use that remapped ordinal when GPU_ID is nonzero.
export CUDA_VISIBLE_DEVICES=0

stop_started_processes() {
  local pid
  for name in target draft; do
    if [[ -s "${RUN_DIR}/${name}.pid" ]]; then
      pid=$(<"${RUN_DIR}/${name}.pid")
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
    rm -f "${RUN_DIR}/${name}.pid"
  done
  rm -f "${SSD_DRAFT_SOCKET}"
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
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

wait_until_socket_ready() {
  local socket_path=$1 pid=$2 log=$3
  for _ in $(seq 1 300); do
    if [[ -S "${socket_path}" ]]; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -160 "${log}" >&2
      return 1
    fi
    sleep 1
  done
  tail -160 "${log}" >&2
  return 1
}

DRAFT_LOG="${LOG_DIR}/draft.log"
DRAFT_MPS_ENV=(
  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${DRAFT_MPS_PERCENT}"
)
if [[ -n "${DRAFT_MPS_CLIENT_PRIORITY}" ]]; then
  DRAFT_MPS_ENV+=("CUDA_MPS_CLIENT_PRIORITY=${DRAFT_MPS_CLIENT_PRIORITY}")
fi
if [[ "${SSD_DRAFT_BACKEND}" == "official" ]]; then
  if [[ ! -x "${SSD_OFFICIAL_PYTHON}" ]]; then
    echo "Official SSD Python not found: ${SSD_OFFICIAL_PYTHON}" >&2
    echo "Run 'cd ${SSD_OFFICIAL_ROOT} && uv sync' once before launching." >&2
    exit 1
  fi
  rm -f "${SSD_DRAFT_SOCKET}"
  OFFICIAL_DRAFT_PYTHONPATH="${REPO_ROOT}/third_party/sglang/python/sglang/srt/speculative:${SSD_OFFICIAL_ROOT}"
  env "${DRAFT_MPS_ENV[@]}" \
    SSD_PROFILE_ROLE=draft PYTHONPATH="${OFFICIAL_DRAFT_PYTHONPATH}" setsid \
    "${DRAFT_PROFILE_PREFIX[@]}" "${SSD_OFFICIAL_PYTHON}" \
      "${REPO_ROOT}/experiments/ssd_scheduler/official_ssd_draft_service.py" \
      --model "${DRAFT_MODEL}" --socket "${SSD_DRAFT_SOCKET}" \
      --draft-length "${SSD_DRAFT_LENGTH}" \
      --fan-outs "${SSD_FAN_OUT_VALUES[@]}" \
      --max-model-len "${SSD_DRAFT_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${SSD_DRAFT_GPU_MEMORY_UTILIZATION}" \
      >"${DRAFT_LOG}" 2>&1 < /dev/null &
  SSD_DRAFT_SERVER_URL="unix://${SSD_DRAFT_SOCKET}"
else
  env "${DRAFT_MPS_ENV[@]}" \
    SSD_PHASE_NVTX="$([[ -n "${NSYS_PROFILE_PREFIX}" ]] && echo 1 || echo 0)" \
    SSD_PROFILE_ROLE=draft PYTHONPATH="${DRAFT_PYTHONPATH}" setsid \
    "${DRAFT_PROFILE_PREFIX[@]}" python -m sglang.launch_server \
      --host 127.0.0.1 --port "${DRAFT_PORT}" \
      --model "${DRAFT_MODEL}" --served-model-name Qwen3-0.6B-SSD \
      --tensor-parallel-size 1 --attention-backend triton \
      --max-total-tokens 16384 --max-running-requests "${DRAFT_MAX_RUNNING_REQUESTS}" \
      --chunked-prefill-size 2048 --mem-fraction-static 0.20 \
      --watchdog-timeout 3000 --trust-remote-code \
      --disable-custom-all-reduce \
      "${DRAFT_TOKENIZER_ARGS[@]}" \
      "${DRAFT_SCHEDULER_ARGS[@]}" \
      "${DRAFT_DETERMINISTIC_ARGS[@]}" \
      --cuda-graph-max-bs "${DRAFT_CUDA_GRAPH_MAX_BS}" \
      --cuda-graph-bs 1 "${SSD_VERIFY_TOKENS}" "${SSD_BRANCH_BATCH}" "${DRAFT_CUDA_GRAPH_MAX_BS}" \
      >"${DRAFT_LOG}" 2>&1 < /dev/null &
  SSD_DRAFT_SERVER_URL="http://127.0.0.1:${DRAFT_PORT}"
fi
DRAFT_PID=$!
echo "${DRAFT_PID}" >"${RUN_DIR}/draft.pid"
if [[ "${SSD_DRAFT_BACKEND}" == "official" ]]; then
  wait_until_socket_ready "${SSD_DRAFT_SOCKET}" "${DRAFT_PID}" "${DRAFT_LOG}"
else
  wait_until_ready "${DRAFT_PORT}" "${DRAFT_PID}" "${DRAFT_LOG}"
fi

TARGET_LOG="${LOG_DIR}/target.log"
TARGET_MPS_ENV=(
  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${TARGET_MPS_PERCENT}"
)
if [[ -n "${TARGET_MPS_CLIENT_PRIORITY}" ]]; then
  TARGET_MPS_ENV+=("CUDA_MPS_CLIENT_PRIORITY=${TARGET_MPS_CLIENT_PRIORITY}")
fi
env "${TARGET_MPS_ENV[@]}" setsid \
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
    --speculative-ssd-draft-server-url "${SSD_DRAFT_SERVER_URL}" \
    "${SSD_FAN_OUT_ARGS[@]}" \
    >"${TARGET_LOG}" 2>&1 < /dev/null &
TARGET_PID=$!
echo "${TARGET_PID}" >"${RUN_DIR}/target.pid"
wait_until_ready "${TARGET_PORT}" "${TARGET_PID}" "${TARGET_LOG}"

trap - ERR INT TERM
echo "SSD ready on GPU ${GPU_ID}: target=http://127.0.0.1:${TARGET_PORT} (${TARGET_MPS_PERCENT}%), draft=${SSD_DRAFT_SERVER_URL} (${DRAFT_MPS_PERCENT}%)"
echo "MPS client priority: target=${TARGET_MPS_CLIENT_PRIORITY:-default}, draft=${DRAFT_MPS_CLIENT_PRIORITY:-default}"
echo "SSD config: backend=${SSD_DRAFT_BACKEND}, K=${SSD_DRAFT_LENGTH}, fan_outs=${SSD_FAN_OUT_VALUES[*]}, verify_tokens=${SSD_VERIFY_TOKENS}, branch_batch=${SSD_BRANCH_BATCH}"
echo "SSD outcome cache: $([[ "${SSD_DISABLE_OUTCOME_CACHE}" == "1" ]] && echo disabled || echo enabled)"
echo "Logs: ${LOG_DIR}"
