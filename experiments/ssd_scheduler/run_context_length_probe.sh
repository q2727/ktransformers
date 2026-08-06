#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers-ssd}"
TARGET_MODEL="${TARGET_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-1.7B}"
DATASET_DIR="${DATASET_DIR:-/home/qinchong/datasets/ssd-paper}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/artifacts/ssd-context-length-v1}"
GPU_ID="${GPU_ID:-1}"
TARGET_PORT="${TARGET_PORT:-31020}"
DRAFT_PORT="${DRAFT_PORT:-31021}"
TARGET_MPS_PERCENT="${TARGET_MPS_PERCENT:-100}"
DRAFT_MPS_PERCENT="${DRAFT_MPS_PERCENT:-50}"
TARGET_MPS_CLIENT_PRIORITY="${TARGET_MPS_CLIENT_PRIORITY:-0}"
DRAFT_MPS_CLIENT_PRIORITY="${DRAFT_MPS_CLIENT_PRIORITY:-1}"
SSD_DRAFT_LENGTH="${SSD_DRAFT_LENGTH:-5}"
SSD_FAN_OUT="${SSD_FAN_OUT:-4}"
SSD_DRAFT_MAX_MODEL_LEN="${SSD_DRAFT_MAX_MODEL_LEN:-2048}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128,512,1024,1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
source .venv/bin/activate

cleanup() {
  REPO_ROOT="${REPO_ROOT}" GPU_ID="${GPU_ID}" OUT_DIR="${OUT_DIR}" \
    experiments/ssd_scheduler/stop_ssd.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

REPO_ROOT="${REPO_ROOT}" TARGET_MODEL="${TARGET_MODEL}" \
DRAFT_MODEL="${DRAFT_MODEL}" GPU_ID="${GPU_ID}" \
TARGET_PORT="${TARGET_PORT}" DRAFT_PORT="${DRAFT_PORT}" \
TARGET_MPS_PERCENT="${TARGET_MPS_PERCENT}" \
DRAFT_MPS_PERCENT="${DRAFT_MPS_PERCENT}" \
TARGET_MPS_CLIENT_PRIORITY="${TARGET_MPS_CLIENT_PRIORITY}" \
DRAFT_MPS_CLIENT_PRIORITY="${DRAFT_MPS_CLIENT_PRIORITY}" \
SSD_DRAFT_LENGTH="${SSD_DRAFT_LENGTH}" SSD_FAN_OUT="${SSD_FAN_OUT}" \
SSD_DRAFT_MAX_MODEL_LEN="${SSD_DRAFT_MAX_MODEL_LEN}" OUT_DIR="${OUT_DIR}" \
  experiments/ssd_scheduler/start_ssd.sh | tee "${OUT_DIR}/start.stdout"

label="context_k${SSD_DRAFT_LENGTH}_f${SSD_FAN_OUT}_draft${DRAFT_MPS_PERCENT}"
python experiments/ssd_scheduler/benchmark_context_lengths.py \
  --url "http://127.0.0.1:${TARGET_PORT}" \
  --model "${TARGET_MODEL}" \
  --dataset-dir "${DATASET_DIR}" \
  --lengths "${CONTEXT_LENGTHS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --label "${label}" \
  --output "${OUT_DIR}/records.jsonl" \
  | tee "${OUT_DIR}/benchmark.stdout"

sleep 1
python experiments/ssd_scheduler/analyze_paper_sweep.py \
  --target-log "${OUT_DIR}/logs/target.log" \
  --draft-log "${OUT_DIR}/logs/draft.log" \
  --records "${OUT_DIR}/records.jsonl" \
  --label "${label}" \
  --output "${OUT_DIR}/analysis.json" \
  | tee "${OUT_DIR}/analysis.stdout"

cleanup
touch "${OUT_DIR}/complete"
trap - EXIT INT TERM
echo "CONTEXT PROBE COMPLETE: ${OUT_DIR}"
