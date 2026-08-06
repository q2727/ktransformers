#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers-ssd}"
TARGET_MODEL="${TARGET_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-0.6B}"
DATASET_DIR="${DATASET_DIR:-/home/qinchong/datasets/ssd-paper}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/artifacts/ssd-protocol-ab}"
GPU_ID="${GPU_ID:-1}"
TARGET_PORT="${TARGET_PORT:-30020}"
DRAFT_PORT="${DRAFT_PORT:-30021}"
NUM_PER_DATASET="${NUM_PER_DATASET:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

# label, draft-side cache, skip tokenizer, fanout
CONFIGS=(
  "legacy_f4 0 0 4"
  "draftside_f4 1 0 4"
  "optimized_f4 1 1 4"
  "optimized_f1 1 1 1"
)

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"
source .venv/bin/activate

CURRENT_OUT=""
stop_current() {
  if [[ -n "${CURRENT_OUT}" ]]; then
    REPO_ROOT="${REPO_ROOT}" GPU_ID="${GPU_ID}" OUT_DIR="${CURRENT_OUT}" \
      experiments/ssd_scheduler/stop_ssd.sh >/dev/null 2>&1 || true
  fi
}
trap stop_current EXIT INT TERM

for config in "${CONFIGS[@]}"; do
  read -r label draft_side_cache skip_tokenizer fan_out <<<"${config}"
  CURRENT_OUT="${OUT_ROOT}/${label}"
  if [[ -f "${CURRENT_OUT}/complete" ]]; then
    echo "SKIP ${label} (complete)"
    CURRENT_OUT=""
    continue
  fi
  rm -rf "${CURRENT_OUT}"
  mkdir -p "${CURRENT_OUT}"

  echo "START ${label}"
  REPO_ROOT="${REPO_ROOT}" \
  TARGET_MODEL="${TARGET_MODEL}" DRAFT_MODEL="${DRAFT_MODEL}" \
  GPU_ID="${GPU_ID}" TARGET_PORT="${TARGET_PORT}" DRAFT_PORT="${DRAFT_PORT}" \
  TARGET_MPS_PERCENT=50 DRAFT_MPS_PERCENT=50 SSD_DRAFT_LENGTH=5 \
  SSD_FAN_OUT="${fan_out}" SSD_DRAFT_SIDE_CACHE="${draft_side_cache}" \
  DRAFT_SKIP_TOKENIZER_INIT="${skip_tokenizer}" OUT_DIR="${CURRENT_OUT}" \
    experiments/ssd_scheduler/start_ssd.sh | tee "${CURRENT_OUT}/start.stdout"

  python experiments/ssd_scheduler/benchmark_paper_workload.py \
    --url "http://127.0.0.1:${TARGET_PORT}" \
    --model "${TARGET_MODEL}" --dataset-dir "${DATASET_DIR}" \
    --num-per-dataset "${NUM_PER_DATASET}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" --label "${label}" \
    --output "${CURRENT_OUT}/records.jsonl" \
    | tee "${CURRENT_OUT}/benchmark.stdout"

  sleep 1
  python experiments/ssd_scheduler/analyze_paper_sweep.py \
    --target-log "${CURRENT_OUT}/logs/target.log" \
    --records "${CURRENT_OUT}/records.jsonl" --label "${label}" \
    --output "${CURRENT_OUT}/analysis.json" \
    | tee "${CURRENT_OUT}/analysis.stdout"

  stop_current
  touch "${CURRENT_OUT}/complete"
  echo "DONE ${label}"
  CURRENT_OUT=""
done

echo "PROTOCOL A/B COMPLETE: ${OUT_ROOT}"
