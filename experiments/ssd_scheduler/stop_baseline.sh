#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-k5-f1-ab}"
PID_FILE="${OUT_DIR}/run/baseline.pid"

if [[ -s "${PID_FILE}" ]]; then
  pid=$(<"${PID_FILE}")
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi

echo "KTransformers baseline stopped."
