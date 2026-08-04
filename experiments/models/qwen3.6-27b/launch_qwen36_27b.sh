#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL="${MODEL:-/data/qinchong/models/Qwen3.6-27B}"
TP="${TP:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-46000}"

cd "${REPO_ROOT}"
source .venv/bin/activate
MODEL="${MODEL}" "${SCRIPT_DIR}/verify_qwen36_27b.py" >/dev/null

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

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

exec python -m sglang.launch_server \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30005}" \
  --model "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend triton \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-131072}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-8}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
  --watchdog-timeout 3000 \
  --served-model-name Qwen3.6-27B \
  --trust-remote-code \
  --enable-mixed-chunk \
  --enable-p2p-check \
  --disable-custom-all-reduce \
  --schedule-policy "${SCHEDULE_POLICY:-lpm}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --collect-tokens-histogram \
  --enable-cache-report
