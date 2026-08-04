#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
HF_BIN="${HF_BIN:-${REPO_ROOT}/.venv/bin/hf}"
MODEL_ROOT="${MODEL_ROOT:-/data/qinchong/models}"
export HF_HOME="${HF_HOME:-/data/qinchong/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

selection="${1:-all}"
case "${selection}" in
  minimax)
    repositories=("MiniMaxAI/MiniMax-M2.5|MiniMax-M2.5")
    ;;
  deepseek)
    repositories=("deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash")
    ;;
  all)
    repositories=(
      "MiniMaxAI/MiniMax-M2.5|MiniMax-M2.5"
      "deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash"
    )
    ;;
  *)
    echo "Usage: $0 [minimax|deepseek|all]" >&2
    exit 2
    ;;
esac

mkdir -p "${MODEL_ROOT}" "${HF_HOME}"
for item in "${repositories[@]}"; do
  repo="${item%%|*}"
  directory="${item##*|}"
  destination="${MODEL_ROOT}/${directory}"
  echo "Downloading ${repo} to ${destination} via ${HF_ENDPOINT}"
  "${HF_BIN}" download "${repo}" --local-dir "${destination}"
done
