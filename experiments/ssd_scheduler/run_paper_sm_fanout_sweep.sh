#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qinchong/workspace/code/ktransformers-ssd}"
TARGET_MODEL="${TARGET_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-30B-A3B}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/qinchong/models/MoE-SpAc/Qwen3-0.6B}"
DATASET_DIR="${DATASET_DIR:-/home/qinchong/datasets/ssd-paper}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/artifacts/ssd-paper-sweep/pilot}"
GPU_ID="${GPU_ID:-1}"
TARGET_PORT="${TARGET_PORT:-31020}"
DRAFT_PORT="${DRAFT_PORT:-31021}"
NUM_PER_DATASET="${NUM_PER_DATASET:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SSD_DRAFT_LENGTH="${SSD_DRAFT_LENGTH:-5}"
SSD_DRAFT_BACKEND="${SSD_DRAFT_BACKEND:-official}"
SSD_DRAFT_MAX_MODEL_LEN="${SSD_DRAFT_MAX_MODEL_LEN:-2048}"
MEASURE_GPU_ENERGY="${MEASURE_GPU_ENERGY:-0}"

# Approximately (target SM, draft SM) = (104,22), (96,32), (64,64)
# on the 128-SM RTX 4090. MPS percentages are the actual control values.
read -r -a SPLITS_RAW <<<"${SSD_SPLITS:-90:10 82:18 75:25 50:50}"
SPLITS=()
for split in "${SPLITS_RAW[@]}"; do
  SPLITS+=("${split/:/ }")
done
read -r -a FANOUTS <<<"${SSD_FANOUTS_SWEEP:-1 2 4 8}"

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

for split in "${SPLITS[@]}"; do
  read -r target_pct draft_pct <<<"${split}"
  for fan_out in "${FANOUTS[@]}"; do
    label="target${target_pct}_draft${draft_pct}_f${fan_out}"
    CURRENT_OUT="${OUT_ROOT}/${label}"
    mkdir -p "${CURRENT_OUT}"
    if [[ -f "${CURRENT_OUT}/complete" ]]; then
      echo "SKIP ${label} (complete)"
      continue
    fi

    echo "START ${label}"
    REPO_ROOT="${REPO_ROOT}" \
    TARGET_MODEL="${TARGET_MODEL}" DRAFT_MODEL="${DRAFT_MODEL}" \
    GPU_ID="${GPU_ID}" TARGET_PORT="${TARGET_PORT}" DRAFT_PORT="${DRAFT_PORT}" \
    TARGET_MPS_PERCENT="${target_pct}" DRAFT_MPS_PERCENT="${draft_pct}" \
    SSD_DRAFT_BACKEND="${SSD_DRAFT_BACKEND}" \
    SSD_DRAFT_MAX_MODEL_LEN="${SSD_DRAFT_MAX_MODEL_LEN}" \
    SSD_DRAFT_LENGTH="${SSD_DRAFT_LENGTH}" SSD_FAN_OUT="${fan_out}" \
    OUT_DIR="${CURRENT_OUT}" \
      experiments/ssd_scheduler/start_ssd.sh | tee "${CURRENT_OUT}/start.stdout"

    ENERGY_ARGS=()
    if [[ "${MEASURE_GPU_ENERGY}" == "1" ]]; then
      ENERGY_ARGS+=(--gpu-index "${GPU_ID}")
      nvidia-smi -i "${GPU_ID}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits \
        >"${CURRENT_OUT}/gpu_process_memory_ready.csv"
    fi

    python experiments/ssd_scheduler/benchmark_paper_workload.py \
      --url "http://127.0.0.1:${TARGET_PORT}" \
      --model "${TARGET_MODEL}" \
      --dataset-dir "${DATASET_DIR}" \
      --num-per-dataset "${NUM_PER_DATASET}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      "${ENERGY_ARGS[@]}" \
      --label "${label}" \
      --output "${CURRENT_OUT}/records.jsonl" \
      | tee "${CURRENT_OUT}/benchmark.stdout"

    if [[ "${MEASURE_GPU_ENERGY}" == "1" ]]; then
      nvidia-smi -i "${GPU_ID}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits \
        >"${CURRENT_OUT}/gpu_process_memory_post.csv"
    fi

    # Let the final asynchronously submitted cache build finish and emit timing.
    sleep 1
    python experiments/ssd_scheduler/analyze_paper_sweep.py \
      --target-log "${CURRENT_OUT}/logs/target.log" \
      --draft-log "${CURRENT_OUT}/logs/draft.log" \
      --records "${CURRENT_OUT}/records.jsonl" \
      --label "${label}" \
      --output "${CURRENT_OUT}/analysis.json" \
      | tee "${CURRENT_OUT}/analysis.stdout"

    stop_current
    touch "${CURRENT_OUT}/complete"
    echo "DONE ${label}"
    CURRENT_OUT=""
  done
done

echo "SWEEP COMPLETE: ${OUT_ROOT}"
