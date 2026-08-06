#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers-ssd}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-0.6B}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/artifacts/ssd-paper-sweep/draft-only}"
GPU_ID="${GPU_ID:-1}"
DRAFT_PORT="${DRAFT_PORT:-31021}"
MPS_PCTS=(18 25 50 100)

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_PID=""
MPS_PIPE_DIR=""
MPS_LOG_DIR=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  if [[ -n "${MPS_PIPE_DIR}" ]]; then
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
      sh -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

for pct in "${MPS_PCTS[@]}"; do
  label="draft_pct${pct}"
  out_dir="${OUT_ROOT}/${label}"
  mkdir -p "${out_dir}"
  MPS_PIPE_DIR="/tmp/qinchong-draft-only-gpu${GPU_ID}-pct${pct}-pipe"
  MPS_LOG_DIR="/tmp/qinchong-draft-only-gpu${GPU_ID}-pct${pct}-log"

  if nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU ${GPU_ID} became busy; stopping before ${label}." >&2
    exit 1
  fi

  rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    nvidia-cuda-mps-control -d

  CUDA_VISIBLE_DEVICES=0 \
  CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
  CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${pct}" \
  setsid python -m sglang.launch_server \
    --host 127.0.0.1 --port "${DRAFT_PORT}" \
    --model "${DRAFT_MODEL}" --served-model-name Qwen3-0.6B-SSD \
    --tensor-parallel-size 1 --attention-backend triton \
    --max-total-tokens 16384 --max-running-requests 32 \
    --chunked-prefill-size 2048 --mem-fraction-static 0.20 \
    --watchdog-timeout 3000 --trust-remote-code --disable-custom-all-reduce \
    --cuda-graph-max-bs 24 --cuda-graph-bs 1 6 12 24 \
    >"${out_dir}/draft.log" 2>&1 < /dev/null &
  SERVER_PID=$!

  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${DRAFT_PORT}/model_info" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -100 "${out_dir}/draft.log" >&2
      exit 1
    fi
    sleep 1
  done

  python experiments/ssd_scheduler/benchmark_draft_fanout.py \
    --url "http://127.0.0.1:${DRAFT_PORT}" \
    --model "${DRAFT_MODEL}" \
    --label "${label}" \
    --output "${out_dir}/results.json"
  cleanup
  echo "DONE ${label}"
done

echo "DRAFT SWEEP COMPLETE: ${OUT_ROOT}"
