#!/usr/bin/env bash
set -euo pipefail

# Run one full VisionReward model replica per listed GPU.
#
# Activate the VisionReward environment before running this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${VISIONREWARD_ROOT:?Set VISIONREWARD_ROOT to the cloned VisionReward repository}"
: "${MODEL_PATH:?Set MODEL_PATH to the VisionReward-Video model directory}"
MODEL_LABEL="${MODEL_LABEL:-glance_3k_sample_ode}"
case "${MODEL_LABEL}" in
    glance_3k_sample_ode|glance_3k_sample_ode_8step|glance_one_sample_ode|glance_one_sample_dmd)
        variant="${MODEL_LABEL#glance_}"
        DEFAULT_VIDEOS_PATH="${EVAL_ROOT}/output/text_align_glance_${variant}"
        DEFAULT_METRICS_ROOT="${EVAL_ROOT}/output/text_align_metrics_glance_${variant}"
        ;;
    *)
        echo "Unsupported MODEL_LABEL: ${MODEL_LABEL}" >&2
        exit 2
        ;;
esac
VIDEOS_PATH="${VIDEOS_PATH:-${DEFAULT_VIDEOS_PATH}}"
MANIFEST_PATH="${MANIFEST_PATH:-${VIDEOS_PATH}/.text_align/manifest.json}"
RESULTS_DIR="${RESULTS_DIR:-${DEFAULT_METRICS_ROOT}/visionreward}"
GPU_IDS="${GPU_IDS:-0}"

QUESTIONS_PATH="${VISIONREWARD_ROOT}/VisionReward_Video/VisionReward_video_qa_select.txt"
WEIGHTS_PATH="${VISIONREWARD_ROOT}/VisionReward_Video/weight.json"

for required_path in     "${MODEL_PATH}/config.json"     "${QUESTIONS_PATH}"     "${WEIGHTS_PATH}"     "${MANIFEST_PATH}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required file not found: ${required_path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_align.py"     --manifest "${MANIFEST_PATH}"     --videos-dir "${VIDEOS_PATH}"

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
num_shards="${#gpu_array[@]}"
if [[ "${num_shards}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU id" >&2
    exit 2
fi

mkdir -p "${RESULTS_DIR}"
pids=()
for ((rank = 0; rank < num_shards; rank++)); do
    gpu_id="${gpu_array[${rank}]}"
    echo "[VisionReward] shard ${rank}/${num_shards} -> GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}"         "${SCRIPT_DIR}/evaluate_text_align_visionreward.py"         --manifest "${MANIFEST_PATH}"         --videos-dir "${VIDEOS_PATH}"         --model-path "${MODEL_PATH}"         --questions-path "${QUESTIONS_PATH}"         --weights-path "${WEIGHTS_PATH}"         --output-dir "${RESULTS_DIR}"         --device cuda:0         --shard-index "${rank}"         --num-shards "${num_shards}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo "At least one VisionReward shard failed; rerun to resume." >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_text_align_metrics.py"     --manifest "${MANIFEST_PATH}"     --vision-results-dir "${RESULTS_DIR}"     --output "${RESULTS_DIR}/summary.json"     --require-complete

echo "[VisionReward] Complete: ${RESULTS_DIR}"
