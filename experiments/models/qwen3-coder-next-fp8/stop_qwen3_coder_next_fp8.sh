#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/experiments/artifacts/services/qwen3-coder-next-fp8}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"
PORT="${PORT:-30005}"

root_pid=""
if [[ -s "${PID_FILE}" ]]; then
  candidate="$(<"${PID_FILE}")"
  if kill -0 "${candidate}" 2>/dev/null; then
    root_pid="${candidate}"
  fi
fi

if [[ -z "${root_pid}" ]]; then
  root_pid="$(pgrep -u "$(id -u)" -f "[k]t run.*/Qwen3-Coder-Next-FP8.*--port ${PORT}" | head -1 || true)"
fi

if [[ -z "${root_pid}" ]]; then
  rm -f "${PID_FILE}"
  echo "No matching Qwen3-Coder-Next-FP8 server is running."
  exit 0
fi

pgid="$(ps -o pgid= -p "${root_pid}" | tr -d ' ')"
kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${root_pid}" 2>/dev/null || true

for _ in {1..30}; do
  kill -0 "${root_pid}" 2>/dev/null || break
  sleep 1
done

if kill -0 "${root_pid}" 2>/dev/null; then
  kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${root_pid}" 2>/dev/null || true
fi

rm -f "${PID_FILE}"
echo "Stopped Qwen3-Coder-Next-FP8 server processes."
