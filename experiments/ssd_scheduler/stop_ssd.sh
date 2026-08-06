#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-scheduler}"
RUN_DIR="${OUT_DIR}/run"
GPU_ID="${GPU_ID:-0}"
SSD_DRAFT_SOCKET="${SSD_DRAFT_SOCKET:-/tmp/ktransformers-ssd-${USER}-gpu${GPU_ID}.sock}"
MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/qinchong-mps-gpu${GPU_ID}-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/qinchong-mps-gpu${GPU_ID}-log}"

terminate_group() {
  local pid=$1
  kill -0 "${pid}" 2>/dev/null || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 1
  done
  # Kill the children first so a multiprocessing child cannot retain an MPS
  # connection after its launcher exits.  Then kill both the process group and
  # leader explicitly; the latter also covers a launcher whose PGID changed.
  pkill -KILL -P "${pid}" 2>/dev/null || true
  kill -KILL -- "-${pid}" 2>/dev/null || true
  kill -KILL "${pid}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 1
  done
  echo "Warning: process ${pid} survived SIGKILL." >&2
}

for name in target draft; do
  pid_file="${RUN_DIR}/${name}.pid"
  [[ -s "${pid_file}" ]] || continue
  pid=$(<"${pid_file}")
  terminate_group "${pid}"
  rm -f "${pid_file}"
done
rm -f "${SSD_DRAFT_SOCKET}"

export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
if ! timeout 10s bash -c \
  'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1'; then
  # A stale CUDA client can wedge the control request.  This MPS instance has
  # a private pipe directory, whose pidfile lets us clean up only this GPU's
  # daemon and server rather than touching another user's MPS instance.
  if [[ -s "${MPS_PIPE_DIR}/nvidia-cuda-mps-control.pid" ]]; then
    mps_control_pid=$(<"${MPS_PIPE_DIR}/nvidia-cuda-mps-control.pid")
    mps_server_pids=$(pgrep -P "${mps_control_pid}" || true)
    [[ -z "${mps_server_pids}" ]] || kill -TERM ${mps_server_pids} 2>/dev/null || true
    kill -TERM "${mps_control_pid}" 2>/dev/null || true
    sleep 2
    [[ -z "${mps_server_pids}" ]] || kill -KILL ${mps_server_pids} 2>/dev/null || true
    kill -KILL "${mps_control_pid}" 2>/dev/null || true
  fi
fi
echo "SSD target, draft, and MPS daemon stopped."
