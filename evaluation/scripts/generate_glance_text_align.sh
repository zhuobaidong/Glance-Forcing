#!/usr/bin/env bash
set -euo pipefail

# Generate the 100-prompt Dynamic/VisionReward/Instruction suite with Glance.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL_VARIANT:?Set MODEL_VARIANT to 3k_sample_ode, one_sample_ode, or one_sample_dmd}"
: "${GLANCE_ROOT:?Set GLANCE_ROOT to the cloned Glance-Forcing repository}"
: "${GLANCE_CKPT_ROOT:?Set GLANCE_CKPT_ROOT to the Glance-Forcing checkpoint directory}"
case "${MODEL_VARIANT}" in
    3k_sample_ode|one_sample_ode|one_sample_dmd) ;;
    *)
        echo "Unsupported MODEL_VARIANT: ${MODEL_VARIANT}" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${GLANCE_ROOT}/configs/causal_forcing_dmd_chunkwise.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GLANCE_CKPT_ROOT}/chunkwise/ar_diffusion.pt}"
LORA_PATH_1="${LORA_PATH_1:-${GLANCE_CKPT_ROOT}/${MODEL_VARIANT}/slow_lora.pt}"
LORA_PATH_2="${LORA_PATH_2:-${GLANCE_CKPT_ROOT}/${MODEL_VARIANT}/fast_lora.pt}"
PROMPT_PATH="${PROMPT_PATH:-${EVAL_ROOT}/prompts/text_align.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${EVAL_ROOT}/output/text_align_glance_${MODEL_VARIANT}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BASE_SEED="${BASE_SEED:-0}"
STEPS="${STEPS:-4}"
MODEL_LABEL="${MODEL_LABEL:-glance_${MODEL_VARIANT}}"

PROTOCOL_DIR="${OUTPUT_FOLDER}/.text_align"
MANIFEST_PATH="${PROTOCOL_DIR}/manifest.json"
CLEAN_PROMPT_PATH="${PROTOCOL_DIR}/prompts.txt"
OUTPUT_NAME_PATH="${PROTOCOL_DIR}/video_basenames.txt"

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
if [[ ! "${BASE_SEED}" =~ ^[0-9]+$ ]]; then
    echo "BASE_SEED must be a non-negative integer" >&2
    exit 2
fi
if [[ "${STEPS}" != "4" && "${STEPS}" != "8" ]]; then
    echo "STEPS must be 4 or 8" >&2
    exit 2
fi

for required_file in \
    "${GLANCE_ROOT}/infer_glance.py" \
    "${CONFIG_PATH}" \
    "${CHECKPOINT_PATH}" \
    "${LORA_PATH_1}" \
    "${LORA_PATH_2}" \
    "${PROMPT_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_FOLDER}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_text_align.py" \
    --prompts "${PROMPT_PATH}" \
    --output-dir "${PROTOCOL_DIR}" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --aux-checkpoint "${LORA_PATH_1}" \
    --aux-checkpoint "${LORA_PATH_2}" \
    --model-label "${MODEL_LABEL}" \
    --inference-steps "${STEPS}" \
    --base-seed "${BASE_SEED}"

args=(
    "${GLANCE_ROOT}/infer_glance.py"
    --config_path "${CONFIG_PATH}"
    --checkpoint_path "${CHECKPOINT_PATH}"
    --lora_path_1 "${LORA_PATH_1}"
    --lora_path_2 "${LORA_PATH_2}"
    --data_path "${CLEAN_PROMPT_PATH}"
    --output_name_path "${OUTPUT_NAME_PATH}"
    --output_folder "${OUTPUT_FOLDER}"
    --sample_index 0
    --seed "${BASE_SEED}"
    --seed_by_prompt
    --steps "${STEPS}"
)

cd "${GLANCE_ROOT}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        "${args[@]}"
else
    "${PYTHON_BIN}" "${args[@]}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_align.py" \
    --manifest "${MANIFEST_PATH}" \
    --videos-dir "${OUTPUT_FOLDER}"

echo "[Glance/text-align] Complete: ${OUTPUT_FOLDER}"
echo "[Glance/text-align] Manifest: ${MANIFEST_PATH}"
