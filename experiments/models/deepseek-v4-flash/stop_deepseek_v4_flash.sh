#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/experiments/artifacts/services/deepseek-v4-flash}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"

if [[ -s "${PID_FILE}" ]]; then
  pid="$(<"${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
fi
pkill -TERM -f '[s]glang.launch_server.*DeepSeek-V4-Flash' 2>/dev/null || true
for _ in {1..30}; do
  if ! pgrep -f '[s]glang.launch_server.*DeepSeek-V4-Flash' >/dev/null; then
    rm -f "${PID_FILE}"
    echo "Stopped DeepSeek-V4-Flash server processes."
    exit 0
  fi
  sleep 1
done
pkill -KILL -f '[s]glang.launch_server.*DeepSeek-V4-Flash' 2>/dev/null || true
rm -f "${PID_FILE}"
echo "Force-stopped DeepSeek-V4-Flash server processes."
