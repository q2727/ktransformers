#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODEL="${MODEL:-/data/qinchong/models/Qwen3.5-122B-A10B}"
TP="${TP:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
read -r -a kt_numa_nodes <<< "${KT_NUMA_NODES:-0 1}"

if (( ${#kt_numa_nodes[@]} != KT_THREADPOOL_COUNT )); then
  echo "KT_NUMA_NODES and KT_THREADPOOL_COUNT disagree." >&2
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
export SGLANG_KT_CPU_TBO="${ENABLE_DUAL_BATCH:-0}"
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_TBO_DEBUG="${TBO_DEBUG:-0}"

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#gpu_ids[@]} != TP )); then
    echo "CUDA_VISIBLE_DEVICES has ${#gpu_ids[@]} GPU(s), expected ${TP}." >&2
    exit 1
  fi
  for gpu_id in "${gpu_ids[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < MIN_FREE_MIB )); then
      echo "GPU ${gpu_id} has ${free_mib} MiB free; ${MIN_FREE_MIB} MiB required." >&2
      exit 1
    fi
  done
fi

extra_args=()
if [[ "${ENABLE_DUAL_BATCH:-0}" == "1" ]]; then
  extra_args+=(--enable-two-batch-overlap)
fi
if [[ "${DISABLE_CUDA_GRAPH:-1}" == "1" ]]; then
  extra_args+=(--disable-cuda-graph)
fi

exec python -m sglang.launch_server \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30006}" \
  --model "${MODEL}" \
  --served-model-name Qwen3.5-122B-A10B-DualBatch \
  --kt-weight-path "${MODEL}" \
  --kt-method BF16 \
  --kt-num-gpu-experts "${GPU_EXPERTS:-16}" \
  --kt-cpuinfer "${CPU_THREADS:-120}" \
  --kt-threadpool-count "${KT_THREADPOOL_COUNT}" \
  --kt-numa-nodes "${kt_numa_nodes[@]}" \
  --kt-gpu-prefill-token-threshold "${GPU_PREFILL_THRESHOLD:-2048}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend triton \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-65536}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-8}" \
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
