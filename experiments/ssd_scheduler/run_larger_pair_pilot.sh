#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
TARGET_MODEL="${TARGET_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-235B-A22B-Instruct-2507-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/qinchong/models/MoE-SpAc/Qwen3-4B-Instruct-2507}"
DATASET_DIR="${DATASET_DIR:-/data/qinchong/datasets/ssd-paper}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/artifacts/largepair-qwen235b-qwen4b-pilot-v1}"
GPU_ID="${GPU_ID:-0}"
TARGET_PORT="${TARGET_PORT:-32020}"
NUM_PER_DATASET="${NUM_PER_DATASET:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# name, target MPS cap, draft MPS cap, fanout, disable outcome cache
CONFIGS=(
  "sync target100_draft100_f1 100 100 1 1"
  "ssd25 target100_draft25_f4 100 25 4 0"
  "ssd50 target100_draft50_f4 100 50 4 0"
  "ssd100 target100_draft100_f4 100 100 4 0"
)

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"
source .venv/bin/activate

CURRENT_OUT=""
cleanup() {
  if [[ -n "${CURRENT_OUT}" ]]; then
    GPU_ID="${GPU_ID}" OUT_DIR="${CURRENT_OUT}" \
      experiments/ssd_scheduler/stop_ssd.sh >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

for config in "${CONFIGS[@]}"; do
  read -r short_name label target_pct draft_pct fanout disable_cache <<<"${config}"
  CURRENT_OUT="${OUT_ROOT}/${short_name}"
  mkdir -p "${CURRENT_OUT}"
  if [[ -f "${CURRENT_OUT}/complete" ]]; then
    echo "SKIP ${label} (complete)"
    CURRENT_OUT=""
    continue
  fi

  target_priority=""
  draft_priority=""
  if [[ "${disable_cache}" == "0" ]]; then
    target_priority=0
    draft_priority=1
  fi

  echo "START ${label}"
  REPO_ROOT="${REPO_ROOT}" \
  TARGET_MODEL="${TARGET_MODEL}" TARGET_KT_METHOD=FP8 \
  TARGET_FP8_GEMM_BACKEND=triton TARGET_MEM_FRACTION_STATIC=0.65 \
  TARGET_MAX_TOTAL_TOKENS=4096 TARGET_CHUNKED_PREFILL_SIZE=2048 \
  DRAFT_MODEL="${DRAFT_MODEL}" GPU_ID="${GPU_ID}" \
  TARGET_PORT="${TARGET_PORT}" TARGET_MPS_PERCENT="${target_pct}" \
  DRAFT_MPS_PERCENT="${draft_pct}" \
  TARGET_MPS_CLIENT_PRIORITY="${target_priority}" \
  DRAFT_MPS_CLIENT_PRIORITY="${draft_priority}" \
  SSD_DRAFT_LENGTH=5 SSD_FAN_OUT="${fanout}" \
  SSD_DISABLE_OUTCOME_CACHE="${disable_cache}" \
  SSD_DRAFT_MAX_MODEL_LEN=2048 SSD_DRAFT_GPU_MEMORY_UTILIZATION=0.02 \
  OUT_DIR="${CURRENT_OUT}" \
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
    --draft-log "${CURRENT_OUT}/logs/draft.log" \
    --records "${CURRENT_OUT}/records.jsonl" --label "${label}" \
    --output "${CURRENT_OUT}/analysis.json" \
    | tee "${CURRENT_OUT}/analysis.stdout"

  cleanup
  touch "${CURRENT_OUT}/complete"
  echo "DONE ${label}"
  CURRENT_OUT=""
done

echo "LARGER-PAIR PILOT COMPLETE: ${OUT_ROOT}"
