#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
TARGET_MODEL="${TARGET_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-0.6B}"
DRAFT_SERVED_MODEL_NAME="${DRAFT_SERVED_MODEL_NAME:-$(basename -- "${DRAFT_MODEL}")-SSD}"
TARGET_SERVED_MODEL_NAME="${TARGET_SERVED_MODEL_NAME:-$(basename -- "${TARGET_MODEL}")-SSD}"
TARGET_KT_METHOD="${TARGET_KT_METHOD:-BF16}"
TARGET_FP8_GEMM_BACKEND="${TARGET_FP8_GEMM_BACKEND:-}"
TARGET_MEM_FRACTION_STATIC="${TARGET_MEM_FRACTION_STATIC:-0.65}"
TARGET_MAX_TOTAL_TOKENS="${TARGET_MAX_TOTAL_TOKENS:-32768}"
TARGET_CHUNKED_PREFILL_SIZE="${TARGET_CHUNKED_PREFILL_SIZE:-2048}"
TARGET_CPU_WORKERS="${TARGET_CPU_WORKERS:-120}"
TARGET_THREADPOOL_COUNT="${TARGET_THREADPOOL_COUNT:-2}"
TARGET_NUMA_NODES="${TARGET_NUMA_NODES:-0 1}"
TARGET_READY_TIMEOUT="${TARGET_READY_TIMEOUT:-180}"
DRAFT_READY_TIMEOUT="${DRAFT_READY_TIMEOUT:-300}"
TARGET_PORT="${TARGET_PORT:-30020}"
DRAFT_PORT="${DRAFT_PORT:-30021}"
SSD_DRAFT_BACKEND="${SSD_DRAFT_BACKEND:-auto}"
SSD_OFFICIAL_ROOT="${SSD_OFFICIAL_ROOT:-${REPO_ROOT}/third_party/ssd}"
SSD_OFFICIAL_PYTHON="${SSD_OFFICIAL_PYTHON:-${SSD_OFFICIAL_ROOT}/.venv/bin/python}"
SSD_DRAFT_MAX_MODEL_LEN="${SSD_DRAFT_MAX_MODEL_LEN:-8192}"
# B=1 with the default 8192-token context needs 32 256-token KV blocks. Three
# percent leaves headroom (about 46 blocks for Qwen3-1.7B on the tested
# 48-GiB device) without reserving memory for dozens of unsupported concurrent
# requests. Paper sweeps capped at 2048 tokens safely use 0.02 explicitly.
SSD_DRAFT_GPU_MEMORY_UTILIZATION="${SSD_DRAFT_GPU_MEMORY_UTILIZATION:-0.03}"
GPU_ID="${GPU_ID:-0}"
TARGET_MPS_PERCENT="${TARGET_MPS_PERCENT:-82}"
DRAFT_MPS_PERCENT="${DRAFT_MPS_PERCENT:-18}"
TARGET_MPS_CLIENT_PRIORITY="${TARGET_MPS_CLIENT_PRIORITY:-}"
DRAFT_MPS_CLIENT_PRIORITY="${DRAFT_MPS_CLIENT_PRIORITY:-}"
SSD_USE_MPS="${SSD_USE_MPS:-auto}"
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

MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/${USER}-mps-gpu${GPU_ID}-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/${USER}-mps-gpu${GPU_ID}-log}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if (( SSD_DRAFT_LENGTH < 1 )); then
  echo "SSD_DRAFT_LENGTH must be at least 1." >&2
  exit 1
fi
if [[ ! -r "${DRAFT_MODEL}/config.json" ]]; then
  echo "Draft model config not found: ${DRAFT_MODEL}/config.json" >&2
  exit 1
fi
if [[ ! -r "${TARGET_MODEL}/config.json" ]]; then
  echo "Target model config not found: ${TARGET_MODEL}/config.json" >&2
  exit 1
fi
TARGET_MODEL_TYPE=$(jq -r \
  '.text_config.model_type // .model_type // empty' \
  "${TARGET_MODEL}/config.json")
DRAFT_MODEL_TYPE=$(jq -r \
  '.text_config.model_type // .model_type // empty' \
  "${DRAFT_MODEL}/config.json")
if [[ -z "${TARGET_MODEL_TYPE}" ]]; then
  echo "Could not determine target model type from ${TARGET_MODEL}/config.json" >&2
  exit 1
fi
if [[ -z "${DRAFT_MODEL_TYPE}" ]]; then
  echo "Could not determine draft model type from ${DRAFT_MODEL}/config.json" >&2
  exit 1
fi
if [[ "${SSD_DRAFT_BACKEND}" == "auto" ]]; then
  case "${DRAFT_MODEL_TYPE}" in
    qwen3|llama)
      SSD_DRAFT_BACKEND=official
      ;;
    *)
      SSD_DRAFT_BACKEND=sglang
      ;;
  esac
fi
if [[ "${SSD_DRAFT_BACKEND}" != "official" && "${SSD_DRAFT_BACKEND}" != "sglang" ]]; then
  echo "SSD_DRAFT_BACKEND must be auto, official, or sglang." >&2
  exit 1
fi
if [[ "${SSD_DRAFT_BACKEND}" == "official" && "${DRAFT_MODEL_TYPE}" == qwen3_5* ]]; then
  echo "The official SSD DraftRunner has no Qwen3.5 Gated DeltaNet implementation." >&2
  echo "Use SSD_DRAFT_BACKEND=sglang, or leave it as auto." >&2
  exit 1
fi
if [[ "${SSD_USE_MPS}" == "auto" ]]; then
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    SSD_USE_MPS=1
  else
    SSD_USE_MPS=0
  fi
fi
if [[ "${SSD_USE_MPS}" != "0" && "${SSD_USE_MPS}" != "1" ]]; then
  echo "SSD_USE_MPS must be auto, 0, or 1." >&2
  exit 1
fi
if [[ "${SSD_USE_MPS}" == "1" ]] \
  && ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "SSD_USE_MPS=1, but nvidia-cuda-mps-control is unavailable." >&2
  exit 1
fi
SSD_VERIFY_TOKENS=$((SSD_DRAFT_LENGTH + 1))
if [[ -z "${DRAFT_CONTINUOUS_DECODE_STEPS}" ]]; then
  if [[ "${DRAFT_MODEL_TYPE}" == qwen3_5* ]]; then
    # Hybrid GDN uses request-scoped recurrent state with the no-buffer
    # scheduler. Re-entering a branch batch for K+1 continuous decode steps
    # can reuse stale state indices; one scheduler step keeps ownership clear.
    DRAFT_CONTINUOUS_DECODE_STEPS=1
  else
    DRAFT_CONTINUOUS_DECODE_STEPS="${SSD_VERIFY_TOKENS}"
  fi
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SSD_HF_CACHE="${SSD_HF_CACHE:-${HOME}/.cache/huggingface}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-${SSD_OFFICIAL_ROOT}/data}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-$(nvidia-smi -i "${GPU_ID}" --query-gpu=compute_cap --format=csv,noheader,nounits | tr -d '[:space:]')}"
export SGLANG_ENABLE_COLOCATED_BATCH_GEN=1
export SGLANG_SSD_DRAFT_SIDE_CACHE="${SSD_DRAFT_SIDE_CACHE}"
export SGLANG_SSD_DISABLE_OUTCOME_CACHE="${SSD_DISABLE_OUTCOME_CACHE}"

TARGET_DETERMINISTIC_ARGS=()
TARGET_PRECISION_ARGS=()
TARGET_SGLANG_ENV=()
TARGET_SCHEDULER_ARGS=()
DRAFT_DETERMINISTIC_ARGS=()
DRAFT_TOKENIZER_ARGS=()
DRAFT_SGLANG_ENV=()
DRAFT_SCHEDULER_ARGS=(
  --num-continuous-decode-steps "${DRAFT_CONTINUOUS_DECODE_STEPS}"
)
SSD_FAN_OUT_ARGS=(--speculative-ssd-fan-out "${SSD_FAN_OUT}")
SSD_FAN_OUT_VALUES=()
SSD_BRANCH_BATCH=$((SSD_VERIFY_TOKENS * SSD_FAN_OUT))
if [[ "${TARGET_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  TARGET_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi
if [[ "${TARGET_MODEL_TYPE}" == qwen3_5* ]]; then
  TARGET_SGLANG_ENV+=("SGLANG_DISABLE_CUDNN_CHECK=1")
  # SGLang classifies Qwen3.5 as a conditional-generation/VLM model and its
  # built-in warmup injects an image. SSD forwards that multimodal token prefix
  # to the text-only draft model, where it is not a valid draft request.
  TARGET_SCHEDULER_ARGS+=(--skip-server-warmup)
fi
if [[ -n "${TARGET_FP8_GEMM_BACKEND}" ]]; then
  TARGET_PRECISION_ARGS+=(--fp8-gemm-backend "${TARGET_FP8_GEMM_BACKEND}")
fi
if [[ "${DRAFT_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  DRAFT_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi
if [[ "${DRAFT_SKIP_TOKENIZER_INIT}" == "1" ]]; then
  DRAFT_TOKENIZER_ARGS+=(--skip-tokenizer-init)
fi
if [[ "${DRAFT_MODEL_TYPE}" == qwen3_5* ]]; then
  # This environment has torch 2.9.1 with cuDNN 9.10. The guarded Conv3d path
  # is not executed because the SSD drafter submits text-only requests, but
  # SGLang checks the multimodal wrapper type before serving starts.
  DRAFT_SGLANG_ENV+=("SGLANG_DISABLE_CUDNN_CHECK=1")
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

if [[ "${SSD_USE_MPS}" == "1" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  nvidia-cuda-mps-control -d

  # The MPS server remaps the selected physical GPU to logical device 0.
  export CUDA_VISIBLE_DEVICES=0
fi

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
  if [[ "${SSD_USE_MPS}" == "1" ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap stop_started_processes ERR INT TERM

wait_until_ready() {
  local port=$1 pid=$2 log=$3 attempts=$4
  for _ in $(seq 1 "${attempts}"); do
    # /model_info becomes available before SGLang's asynchronous warmup ends,
    # while /health can inject a one-token generation request. Wait for the
    # explicit warmup-complete marker, then use the read-only endpoint.
    if grep -qF "The server is fired up and ready to roll!" "${log}" \
      && curl -fsS "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1; then
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
  local socket_path=$1 pid=$2 log=$3 attempts=$4
  for _ in $(seq 1 "${attempts}"); do
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
DRAFT_MPS_ENV=()
if [[ "${SSD_USE_MPS}" == "1" ]]; then
  DRAFT_MPS_ENV+=("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${DRAFT_MPS_PERCENT}")
fi
if [[ "${SSD_USE_MPS}" == "1" && -n "${DRAFT_MPS_CLIENT_PRIORITY}" ]]; then
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
  env "${DRAFT_MPS_ENV[@]}" "${DRAFT_SGLANG_ENV[@]}" \
    SSD_PHASE_NVTX="$([[ -n "${NSYS_PROFILE_PREFIX}" ]] && echo 1 || echo 0)" \
    SSD_PROFILE_ROLE=draft PYTHONPATH="${DRAFT_PYTHONPATH}" setsid \
    "${DRAFT_PROFILE_PREFIX[@]}" python -m sglang.launch_server \
      --host 127.0.0.1 --port "${DRAFT_PORT}" \
      --model "${DRAFT_MODEL}" --served-model-name "${DRAFT_SERVED_MODEL_NAME}" \
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
  wait_until_socket_ready \
    "${SSD_DRAFT_SOCKET}" "${DRAFT_PID}" "${DRAFT_LOG}" "${DRAFT_READY_TIMEOUT}"
else
  wait_until_ready \
    "${DRAFT_PORT}" "${DRAFT_PID}" "${DRAFT_LOG}" "${DRAFT_READY_TIMEOUT}"
fi

TARGET_LOG="${LOG_DIR}/target.log"
TARGET_MPS_ENV=()
if [[ "${SSD_USE_MPS}" == "1" ]]; then
  TARGET_MPS_ENV+=("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${TARGET_MPS_PERCENT}")
fi
if [[ "${SSD_USE_MPS}" == "1" && -n "${TARGET_MPS_CLIENT_PRIORITY}" ]]; then
  TARGET_MPS_ENV+=("CUDA_MPS_CLIENT_PRIORITY=${TARGET_MPS_CLIENT_PRIORITY}")
fi
TARGET_NUMA_ARGS=()
if [[ -n "${TARGET_NUMA_NODES}" ]]; then
  read -r -a TARGET_NUMA_NODE_VALUES <<<"${TARGET_NUMA_NODES}"
  TARGET_NUMA_ARGS+=(--kt-numa-nodes "${TARGET_NUMA_NODE_VALUES[@]}")
fi
env "${TARGET_MPS_ENV[@]}" "${TARGET_SGLANG_ENV[@]}" setsid \
  "${TARGET_PROFILE_PREFIX[@]}" python -m sglang.launch_server \
    --host 127.0.0.1 --port "${TARGET_PORT}" \
    --model "${TARGET_MODEL}" --served-model-name "${TARGET_SERVED_MODEL_NAME}" \
    --kt-weight-path "${TARGET_MODEL}" --kt-method "${TARGET_KT_METHOD}" \
    --kt-num-gpu-experts 0 --kt-cpuinfer "${TARGET_CPU_WORKERS}" \
    --kt-threadpool-count "${TARGET_THREADPOOL_COUNT}" \
    "${TARGET_NUMA_ARGS[@]}" --kt-gpu-prefill-token-threshold 2048 \
    --tensor-parallel-size 1 --attention-backend triton \
    --max-total-tokens "${TARGET_MAX_TOTAL_TOKENS}" --max-running-requests 2 \
    --chunked-prefill-size "${TARGET_CHUNKED_PREFILL_SIZE}" \
    --mem-fraction-static "${TARGET_MEM_FRACTION_STATIC}" \
    --watchdog-timeout 3000 --disable-shared-experts-fusion \
    --trust-remote-code --enable-p2p-check --disable-custom-all-reduce \
    "${TARGET_DETERMINISTIC_ARGS[@]}" "${TARGET_PRECISION_ARGS[@]}" \
    "${TARGET_SCHEDULER_ARGS[@]}" \
    --reasoning-parser qwen3 --cuda-graph-max-bs 1 --cuda-graph-bs 1 \
    --speculative-algorithm SSD \
    --speculative-num-steps "${SSD_DRAFT_LENGTH}" --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens "${SSD_VERIFY_TOKENS}" \
    --speculative-ssd-draft-server-url "${SSD_DRAFT_SERVER_URL}" \
    "${SSD_FAN_OUT_ARGS[@]}" \
    >"${TARGET_LOG}" 2>&1 < /dev/null &
TARGET_PID=$!
echo "${TARGET_PID}" >"${RUN_DIR}/target.pid"
wait_until_ready \
  "${TARGET_PORT}" "${TARGET_PID}" "${TARGET_LOG}" "${TARGET_READY_TIMEOUT}"

trap - ERR INT TERM
if [[ "${SSD_USE_MPS}" == "1" ]]; then
  echo "SSD ready on GPU ${GPU_ID}: target=http://127.0.0.1:${TARGET_PORT} (${TARGET_MPS_PERCENT}%), draft=${SSD_DRAFT_SERVER_URL} (${DRAFT_MPS_PERCENT}%)"
  echo "MPS client priority: target=${TARGET_MPS_CLIENT_PRIORITY:-default}, draft=${DRAFT_MPS_CLIENT_PRIORITY:-default}"
else
  echo "SSD ready without MPS on GPU ${GPU_ID}: target=http://127.0.0.1:${TARGET_PORT}, draft=${SSD_DRAFT_SERVER_URL}"
  echo "Warning: SM quotas and cross-process spatial overlap are disabled."
fi
echo "KT CPU config: workers=${TARGET_CPU_WORKERS}, threadpools=${TARGET_THREADPOOL_COUNT}, numa_nodes=${TARGET_NUMA_NODES:-none}"
echo "SSD config: backend=${SSD_DRAFT_BACKEND}, target_model_type=${TARGET_MODEL_TYPE}, draft_model_type=${DRAFT_MODEL_TYPE}, K=${SSD_DRAFT_LENGTH}, fan_outs=${SSD_FAN_OUT_VALUES[*]}, verify_tokens=${SSD_VERIFY_TOKENS}, branch_batch=${SSD_BRANCH_BATCH}"
echo "SSD outcome cache: $([[ "${SSD_DISABLE_OUTCOME_CACHE}" == "1" ]] && echo disabled || echo enabled)"
echo "Logs: ${LOG_DIR}"
