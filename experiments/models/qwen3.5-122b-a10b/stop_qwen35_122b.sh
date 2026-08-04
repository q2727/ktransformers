#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/experiments/artifacts/services/qwen3.5-122b-a10b}"
PID_FILE="${PID_FILE:-${RUNTIME_DIR}/server.pid}"
PORT="${PORT:-30005}"

collect_tree() {
  local parent="$1" child
  for child in $(pgrep -P "${parent}" 2>/dev/null || true); do
    collect_tree "${child}"
  done
  printf '%s\n' "${parent}"
}

pids=""
if [[ -s "${PID_FILE}" ]]; then
  root_pid="$(<"${PID_FILE}")"
  if kill -0 "${root_pid}" 2>/dev/null; then
    pids="$(collect_tree "${root_pid}")"
  fi
fi

if [[ -z "${pids}" ]]; then
  pids="$(pgrep -u "$(id -u)" -f "[s]glang.launch_server.*--port ${PORT}" || true)"
fi

if [[ -z "${pids}" ]]; then
  rm -f "${PID_FILE}"
  echo "No matching server process is running."
  exit 0
fi

for pid in ${pids}; do
  kill -TERM "${pid}" 2>/dev/null || true
done

for _ in {1..10}; do
  alive=0
  for pid in ${pids}; do
    kill -0 "${pid}" 2>/dev/null && alive=1
  done
  (( alive == 0 )) && break
  sleep 1
done

for pid in ${pids}; do
  kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
done

rm -f "${PID_FILE}"
echo "Stopped Qwen3.5-122B-A10B server processes."
