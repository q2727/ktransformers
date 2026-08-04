#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODEL="${MODEL:-/home/qinchong/models/Qwen3.5-122B-A10B}"
TP="${TP:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
RECORD_DIR="${RECORD_DIR:-${REPO_ROOT}/experiments/artifacts/specmoe/expert-records}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
read -r -a kt_numa_nodes <<< "${KT_NUMA_NODES:-0 1}"

if (( ${#kt_numa_nodes[@]} != KT_THREADPOOL_COUNT )); then
  echo "KT_NUMA_NODES has ${#kt_numa_nodes[@]} node(s), but KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT}." >&2
  exit 1
fi

if [[ ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "Model is missing or incomplete: ${MODEL}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${RECORD_DIR}"
if [[ -n "${COMPONENT_TIMING_DIR:-}" || -n "${COMPONENT_TIMING_CONTROL:-}" ]]; then
  if [[ -z "${COMPONENT_TIMING_DIR:-}" || -z "${COMPONENT_TIMING_CONTROL:-}" ]]; then
    echo "COMPONENT_TIMING_DIR and COMPONENT_TIMING_CONTROL must be set together." >&2
    exit 1
  fi
  mkdir -p "${COMPONENT_TIMING_DIR}"
  mkdir -p "$(dirname -- "${COMPONENT_TIMING_CONTROL}")"
  : > "${COMPONENT_TIMING_CONTROL}"
  export SGLANG_COMPONENT_TIMING_DIR="${COMPONENT_TIMING_DIR}"
  export SGLANG_COMPONENT_TIMING_CONTROL="${COMPONENT_TIMING_CONTROL}"
  export SGLANG_COMPONENT_TIMING_BATCH_SIZE="${COMPONENT_TIMING_BATCH_SIZE:-8}"
fi
if [[ -n "${REQUEST_METADATA:-}" ]]; then
  if [[ ! -f "${REQUEST_METADATA}" ]]; then
    echo "Request metadata file does not exist: ${REQUEST_METADATA}" >&2
    exit 1
  fi
  export SGLANG_DUPLICATE_LOAD_REQUEST_METADATA="${REQUEST_METADATA}"
fi

# Use the source tree at the exact submodule revision so profiling changes do
# not mutate the shared virtualenv's site-packages.
export PYTHONPATH="${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#gpu_ids[@]} != TP )); then
    echo "CUDA_VISIBLE_DEVICES has ${#gpu_ids[@]} GPU(s), but TP=${TP}." >&2
    exit 1
  fi
  for gpu_id in "${gpu_ids[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < MIN_FREE_MIB )); then
      echo "GPU ${gpu_id} has only ${free_mib} MiB free; ${MIN_FREE_MIB} MiB is required." >&2
      exit 1
    fi
  done
fi

mkdir -p "${RECORD_DIR}"

extra_args=(
  --schedule-policy "${SCHEDULE_POLICY:-fcfs}"
  --expert-distribution-recorder-mode per_token
  --expert-distribution-recorder-buffer-size "${RECORDER_BUFFER_SIZE:-2000}"
  --record-kt-gpu-expert-distribution
)

if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  extra_args+=(--disable-cuda-graph)
fi
if [[ "${ENABLE_KT_TIMING:-0}" == "1" ]]; then
  export SGLANG_KT_HYBRID_TIMING=1
  export SGLANG_LOGGING_LEVEL=debug
fi
if [[ "${ENABLE_KT_DEEP_TIMING:-0}" == "1" ]]; then
  export SGLANG_KT_HYBRID_TIMING=1
  export SGLANG_KT_HYBRID_TIMING_DEEP=1
  export SGLANG_LOGGING_LEVEL=debug
fi
if [[ "${DYNAMIC_EXPERT_UPDATE:-0}" == "1" ]]; then
  extra_args+=(--kt-enable-dynamic-expert-update)
fi
if [[ "${DECODE_HOT_EXPERT_UPDATE:-0}" == "1" ]]; then
  hot_profile_path="${HOT_PROFILE_PATH:-${REPO_ROOT}/experiments/artifacts/specmoe/decode-hot/profile.jsonl}"
  hot_control_path="${HOT_CONTROL_PATH:-${REPO_ROOT}/experiments/artifacts/specmoe/decode-hot/control}"
  mkdir -p "$(dirname -- "${hot_profile_path}")"
  : > "${hot_profile_path}"
  printf '%s\n' "${HOT_CONTROL_MODE:-dynamic}" > "${hot_control_path}"
  export SGLANG_KT_DECODE_HOT_PROFILE_PATH="${hot_profile_path}"
  export SGLANG_KT_DECODE_HOT_CONTROL_PATH="${hot_control_path}"
  extra_args+=(
    --kt-decode-hot-expert-update
    --kt-decode-hot-min-tokens "${DECODE_HOT_MIN_TOKENS:-8}"
    --kt-decode-hot-max-promotions "${DECODE_HOT_MAX_PROMOTIONS:-1}"
    --kt-decode-hot-ema-decay "${DECODE_HOT_EMA_DECAY:-0.7}"
    --kt-decode-hot-hysteresis "${DECODE_HOT_HYSTERESIS:-1.25}"
    --kt-decode-hot-min-residency "${DECODE_HOT_MIN_RESIDENCY:-16}"
    --kt-decode-hot-refresh-interval "${DECODE_HOT_REFRESH_INTERVAL:-16}"
  )
fi

exec python -m sglang.launch_server \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30006}" \
  --model "${MODEL}" \
  --served-model-name Qwen3.5-122B-A10B-SpecMoE \
  --kt-weight-path "${MODEL}" \
  --kt-method BF16 \
  --kt-num-gpu-experts "${GPU_EXPERTS:-16}" \
  --kt-cpuinfer "${CPU_THREADS:-120}" \
  --kt-threadpool-count "${KT_THREADPOOL_COUNT}" \
  --kt-numa-nodes "${kt_numa_nodes[@]}" \
  --kt-gpu-prefill-token-threshold "${GPU_PREFILL_THRESHOLD:-2048}" \
  --tensor-parallel-size "${TP}" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}" \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}" \
  --speculative-moe-runner-backend triton \
  --attention-backend triton \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-65536}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-32}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.88}" \
  --watchdog-timeout 3000 \
  --disable-shared-experts-fusion \
  --disable-radix-cache \
  --trust-remote-code \
  --enable-mixed-chunk \
  --enable-p2p-check \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  "${extra_args[@]}"
