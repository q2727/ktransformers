#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL="${MODEL:-/data/qinchong/models/MiniMax-M2.5}"
TP="${TP:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-46000}"

cd "${REPO_ROOT}"
source .venv/bin/activate
MODEL="${MODEL}" "${SCRIPT_DIR}/verify_minimax_m25.py" >/dev/null

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

extra_args=(--schedule-policy "${SCHEDULE_POLICY:-lpm}")
read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS:-1 2 4 6 8}"
extra_args+=(--cuda-graph-bs "${cuda_graph_bs[@]}")
if [[ "${ENABLE_MIXED_CHUNK:-1}" == "1" ]]; then
  extra_args+=(--enable-mixed-chunk)
fi
if [[ "${ENABLE_METRICS:-1}" == "1" ]]; then
  extra_args+=(--enable-metrics --collect-tokens-histogram)
fi
if [[ "${ENABLE_CACHE_REPORT:-1}" == "1" ]]; then
  extra_args+=(--enable-cache-report)
fi

exec python -m sglang.launch_server \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30005}" \
  --model "${MODEL}" \
  --kt-weight-path "${MODEL}" \
  --kt-cpuinfer "${CPU_THREADS:-120}" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-num-gpu-experts "${GPU_EXPERTS:-30}" \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold "${GPU_PREFILL_THRESHOLD:-400}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.94}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-32768}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-150000}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-8}" \
  --watchdog-timeout 3000 \
  --served-model-name MiniMax-M2.5 \
  --trust-remote-code \
  --enable-p2p-check \
  --disable-custom-all-reduce \
  --disable-shared-experts-fusion \
  "${extra_args[@]}"
