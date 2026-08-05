#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-scheduler}"
RUN_DIR="${OUT_DIR}/run"
MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/qinchong-mps-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/qinchong-mps-log}"

for name in target draft; do
  pid_file="${RUN_DIR}/${name}.pid"
  [[ -s "${pid_file}" ]] || continue
  pid=$(<"${pid_file}")
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
done

export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
echo "SSD target, draft, and MPS daemon stopped."
