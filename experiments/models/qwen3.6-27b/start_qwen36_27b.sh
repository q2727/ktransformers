#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/experiments/artifacts/services/qwen3.6-27b}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"
LOG_FILE="${LOG_FILE:-${RUNTIME_DIR}/server.log}"
PORT="${PORT:-30005}"

mkdir -p "${RUNTIME_DIR}"
if [[ -s "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "Server is already running with PID $(<"${PID_FILE}")." >&2
  exit 1
fi
if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
  echo "Port ${PORT} is already in use." >&2
  exit 1
fi

setsid env PORT="${PORT}" "${SCRIPT_DIR}/launch_qwen36_27b.sh" >"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"
echo "Started PID ${pid}; log: ${LOG_FILE}"
