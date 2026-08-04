#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODE="${MODE:-baseline}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/experiments/artifacts/dual_batch_overlap/services/${MODE}}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"
LOG_FILE="${LOG_FILE:-${RUNTIME_DIR}/server.log}"
PORT="${PORT:-30006}"

mkdir -p "${RUNTIME_DIR}"
if [[ -s "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "Server already running with PID $(<"${PID_FILE}")." >&2
  exit 1
fi
if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
  echo "Port ${PORT} is already in use." >&2
  exit 1
fi

enable_dual_batch=0
if [[ "${MODE}" == "dual_batch" ]]; then
  enable_dual_batch=1
elif [[ "${MODE}" != "baseline" ]]; then
  echo "MODE must be baseline or dual_batch, got ${MODE}." >&2
  exit 1
fi

setsid env PORT="${PORT}" ENABLE_DUAL_BATCH="${enable_dual_batch}" \
  "${SCRIPT_DIR}/launch_qwen35.sh" >"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
echo "Started ${MODE} server PID ${pid}; log: ${LOG_FILE}"
