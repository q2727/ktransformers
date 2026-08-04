#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODE="${MODE:-baseline}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/experiments/artifacts/dual_batch_overlap/services/${MODE}}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"

if [[ ! -s "${PID_FILE}" ]]; then
  echo "No PID file: ${PID_FILE}"
  exit 0
fi
pid="$(<"${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}"
  for _ in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f "${PID_FILE}"
