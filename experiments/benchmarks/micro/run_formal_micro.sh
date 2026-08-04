#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

: "${MODEL:?Set MODEL to the served model name}"
: "${TOKENIZER:?Set TOKENIZER to the local model/tokenizer directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT below experiments/artifacts/micro}"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:30005}"
TRIALS="${TRIALS:-1}"
WARMUP_PROMPT_TOKENS="${WARMUP_PROMPT_TOKENS:-1024}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
FLUSH_BETWEEN_TRIALS="${FLUSH_BETWEEN_TRIALS:-1}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-}"
GENERATE_EXTRA_BODY_JSON="${GENERATE_EXTRA_BODY_JSON:-}"
[[ -n "${CHAT_TEMPLATE_KWARGS_JSON}" ]] || CHAT_TEMPLATE_KWARGS_JSON='{}'
[[ -n "${EXTRA_BODY_JSON}" ]] || EXTRA_BODY_JSON='{}'
[[ -n "${GENERATE_EXTRA_BODY_JSON}" ]] || GENERATE_EXTRA_BODY_JSON='{}'

mkdir -p "${OUTPUT_ROOT}/prefill" "${OUTPUT_ROOT}/decode"

complete_summary() {
  local benchmark="$1"
  local expected_points="$2"
  local directory="${OUTPUT_ROOT}/${benchmark}"
  "${PYTHON_BIN}" - "${directory}" "${benchmark}" "${expected_points}" <<'PY'
import json
import sys
from pathlib import Path

directory, benchmark, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for path in sorted(directory.glob("*.summary.json"), reverse=True):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if payload.get("benchmark") == benchmark and len(payload.get("points", [])) == expected:
        print(path)
        break
PY
}

run_one() {
  local benchmark="$1"
  local expected_points="$2"
  local summary
  summary="$(complete_summary "${benchmark}" "${expected_points}")"
  if [[ -z "${summary}" ]]; then
    curl -fsS --max-time 5 "${SERVER_URL}/health" >/dev/null
    local template_args=()
    if [[ -n "${CHAT_TEMPLATE_FILE}" ]]; then
      template_args+=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
    fi
    if [[ "${FLUSH_BETWEEN_TRIALS}" == "0" ]]; then
      template_args+=(--no-flush-between-trials)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_micro.py" "${benchmark}" \
      --model "${MODEL}" \
      --tokenizer "${TOKENIZER}" \
      --api-url "${SERVER_URL}/v1/chat/completions" \
      --decode-endpoint "${SERVER_URL}/generate" \
      --flush-url "${SERVER_URL}/flush_cache" \
      --output-dir "${OUTPUT_ROOT}/${benchmark}" \
      --trials "${TRIALS}" \
      --warmup-prompt-tokens "${WARMUP_PROMPT_TOKENS}" \
      --chat-template-kwargs-json "${CHAT_TEMPLATE_KWARGS_JSON}" \
      "${template_args[@]}" \
      --extra-body-json "${EXTRA_BODY_JSON}" \
      --generate-extra-body-json "${GENERATE_EXTRA_BODY_JSON}" \
      > "${OUTPUT_ROOT}/${benchmark}/formal.console.log" 2>&1
    summary="$(complete_summary "${benchmark}" "${expected_points}")"
  fi
  if [[ -z "${summary}" ]]; then
    echo "No complete ${benchmark} summary was produced" >&2
    exit 1
  fi
  local plot_name
  if [[ "${benchmark}" == "prefill" ]]; then
    plot_name="prefill_ttft.svg"
  else
    plot_name="decode_tpot.svg"
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_micro.py" "${summary}" \
    --output "${OUTPUT_ROOT}/${benchmark}/${plot_name}"
  "${PYTHON_BIN}" - "${summary}" "${OUTPUT_ROOT}/${benchmark}/formal_selection.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
selection_path = Path(sys.argv[2])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
selection = {
    "schema_version": 1,
    "benchmark": payload["benchmark"],
    "model": payload["model"],
    "trials": payload["trials"],
    "point_count": len(payload["points"]),
    "summary_path": str(summary_path),
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
}
selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
PY
  printf '%s\t%s\n' "${benchmark}" "${summary}"
}

run_one prefill 8
run_one decode 5
