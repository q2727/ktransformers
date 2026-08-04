#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL="${MODEL:-/data/qinchong/models/DeepSeek-V4-Flash}"
PYTHON_BIN="${PYTHON_BIN:-/data/qinchong/venvs/ktransformers-dsv4/bin/python}"
TP="${TP:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-46000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "DeepSeek V4 environment is missing: ${PYTHON_BIN}" >&2
  echo "Run ${SCRIPT_DIR}/setup_dsv4_env.sh first." >&2
  exit 1
fi
MODEL="${MODEL}" "${SCRIPT_DIR}/verify_deepseek_v4_flash.py" >/dev/null

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.9}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"
export SGLANG_DSV4_MODE="${SGLANG_DSV4_MODE:-2604}"
export SGLANG_DSV4_2604_SUBMODE="${SGLANG_DSV4_2604_SUBMODE:-2604B}"

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
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${SCRIPT_DIR}/chat_template.jinja}"
if [[ -n "${CHAT_TEMPLATE_FILE}" ]]; then
  [[ -s "${CHAT_TEMPLATE_FILE}" ]] || {
    echo "Chat template is missing: ${CHAT_TEMPLATE_FILE}" >&2
    exit 1
  }
  extra_args+=(--chat-template "${CHAT_TEMPLATE_FILE}")
fi
read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS:-1 2 4 6 8}"
extra_args+=(--cuda-graph-bs "${cuda_graph_bs[@]}")
if [[ "${ENABLE_MIXED_CHUNK:-0}" == "1" ]]; then
  extra_args+=(--enable-mixed-chunk)
fi
if [[ "${DISABLE_RADIX_CACHE:-1}" == "1" ]]; then
  extra_args+=(--disable-radix-cache)
fi
if [[ "${SKIP_SERVER_WARMUP:-1}" == "1" ]]; then
  extra_args+=(--skip-server-warmup)
fi
if [[ "${ENABLE_METRICS:-1}" == "1" ]]; then
  extra_args+=(--enable-metrics --collect-tokens-histogram)
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30005}" \
  --model "${MODEL}" \
  --kt-weight-path "${MODEL}" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts "${GPU_EXPERTS:-20}" \
  --kt-cpuinfer "${CPU_THREADS:-120}" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-gpu-prefill-token-threshold "${GPU_PREFILL_THRESHOLD:-4096}" \
  --tensor-parallel-size "${TP}" \
  --context-length "${CONTEXT_LENGTH:-196608}" \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS:-8192}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-150000}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-8}" \
  --watchdog-timeout 3000 \
  --served-model-name DeepSeek-V4-Flash \
  --trust-remote-code \
  --enable-p2p-check \
  --disable-custom-all-reduce \
  --disable-shared-experts-fusion \
  "${extra_args[@]}"
