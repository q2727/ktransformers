#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL="${MODEL:-/data/qinchong/models/Qwen3.5-122B-A10B}"
TP="${TP:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-46000}"

if [[ ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "Model is missing or incomplete: ${MODEL}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate
MODEL="${MODEL}" "${SCRIPT_DIR}/verify_qwen35_122b.py" >/dev/null

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# The two Ada GPUs sit behind different CPU sockets and have no NVLink.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

extra_sglang_args=(--schedule-policy "${SCHEDULE_POLICY:-lpm}")
read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS:-1 2 4 6 8}"
extra_sglang_args+=(--cuda-graph-bs "${cuda_graph_bs[@]}")
if [[ "${ENABLE_METRICS:-0}" == "1" ]]; then
  extra_sglang_args+=(--enable-metrics --collect-tokens-histogram)
fi
if [[ "${ENABLE_CACHE_REPORT:-0}" == "1" ]]; then
  extra_sglang_args+=(--enable-cache-report)
fi
if [[ -n "${MAX_MAMBA_CACHE_SIZE:-160}" ]]; then
  extra_sglang_args+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE:-160}")
fi

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#gpu_ids[@]} != TP )); then
    echo "CUDA_VISIBLE_DEVICES has ${#gpu_ids[@]} GPU(s), but TP=${TP}." >&2
    exit 1
  fi
  for gpu_id in "${gpu_ids[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < MIN_FREE_MIB )); then
      echo "GPU ${gpu_id} has only ${free_mib} MiB free; ${MIN_FREE_MIB} MiB is required for a safe startup." >&2
      exit 1
    fi
  done
fi

exec kt run "${MODEL}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30005}" \
  --gpu-experts "${GPU_EXPERTS:-48}" \
  --cpu-threads "${CPU_THREADS:-120}" \
  --numa-nodes 0 --numa-nodes 1 \
  --tp "${TP}" \
  --kt-method BF16 \
  --kt-gpu-prefill-threshold "${GPU_PREFILL_THRESHOLD:-4096}" \
  --attention-backend triton \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-262144}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-8}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}" \
  --watchdog-timeout 3000 \
  --served-model-name Qwen3.5-122B-A10B \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --enable-mixed-chunk \
  --enable-p2p-check \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  "${extra_sglang_args[@]}"
