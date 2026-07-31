#!/usr/bin/env bash
set -euo pipefail

# Generate the 946-prompt x 5-seed VBench suite with Glance-Forcing.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL_VARIANT:?Set MODEL_VARIANT to 3k_sample_ode, one_sample_ode, or one_sample_dmd}"
: "${VBENCH_ROOT:?Set VBENCH_ROOT to the cloned VBench repository}"
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
EXTENDED_PROMPT_PATH="${EXTENDED_PROMPT_PATH:-${EVAL_ROOT}/prompts/all_dimension_extended.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${EVAL_ROOT}/output/vbench_standard_extended_glance_${MODEL_VARIANT}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BASE_SEED="${BASE_SEED:-1000}"
STEPS="${STEPS:-4}"

PROTOCOL_DIR="${OUTPUT_FOLDER}/.vbench"
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
    "${EXTENDED_PROMPT_PATH}" \
    "${VBENCH_ROOT}/prompts/all_dimension.txt" \
    "${VBENCH_ROOT}/vbench/VBench_full_info.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_FOLDER}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vbench_extended.py" \
    --vbench-root "${VBENCH_ROOT}" \
    --extended-prompts "${EXTENDED_PROMPT_PATH}" \
    --output-dir "${PROTOCOL_DIR}"

run_generation() {
    local sample_index="$1"
    local seed=$((BASE_SEED + sample_index))
    local args=(
        "${GLANCE_ROOT}/infer_glance.py"
        --config_path "${CONFIG_PATH}"
        --checkpoint_path "${CHECKPOINT_PATH}"
        --lora_path_1 "${LORA_PATH_1}"
        --lora_path_2 "${LORA_PATH_2}"
        --data_path "${EXTENDED_PROMPT_PATH}"
        --output_name_path "${OUTPUT_NAME_PATH}"
        --output_folder "${OUTPUT_FOLDER}"
        --sample_index "${sample_index}"
        --seed "${seed}"
        --steps "${STEPS}"
    )

    echo "[Glance/VBench] variant=${MODEL_VARIANT}, sample_index=${sample_index}, seed=${seed}, prompts=946, GPUs=${NPROC_PER_NODE}"
    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
        "${PYTHON_BIN}" -m torch.distributed.run \
            --standalone \
            --nproc_per_node="${NPROC_PER_NODE}" \
            "${args[@]}"
    else
        "${PYTHON_BIN}" "${args[@]}"
    fi
}

cd "${GLANCE_ROOT}"
for sample_index in 0 1 2 3 4; do
    run_generation "${sample_index}"
done

missing_count=0
while IFS= read -r basename; do
    for sample_index in 0 1 2 3 4; do
        if [[ ! -f "${OUTPUT_FOLDER}/${basename}-${sample_index}.mp4" ]]; then
            missing_count=$((missing_count + 1))
        fi
    done
done < "${OUTPUT_NAME_PATH}"
if [[ "${missing_count}" -ne 0 ]]; then
    echo "Glance VBench generation is incomplete: ${missing_count} videos missing" >&2
    exit 1
fi

echo "[Glance/VBench] Complete: ${OUTPUT_FOLDER}"
echo "[Glance/VBench] Validated 946 prompts x 5 seeds = 4730 videos"
