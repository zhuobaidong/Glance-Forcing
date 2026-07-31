#!/usr/bin/env bash
set -euo pipefail

# Evaluate the Self-Forcing-style extended-prompt VBench protocol used by the
# Causal Forcing paper. The original VBench checkout is not modified.
#
# Required:
#   VBENCH_ROOT=/path/to/VBench
#   VBENCH_CACHE_DIR=/path/to/vbench_cache
#
# Optional:
#   MODEL_LABEL=glance_3k_sample_ode
#   VIDEOS_PATH=/path/to/generated/videos
#   OUTPUT_PATH=/path/to/evaluation_results
#   NPROC_PER_NODE=1
#   SKIP_STANDARD_EVALUATION=0
#   SKIP_TEMPORAL_FLICKERING=0
#   VBENCH_HF_HOME=/path/to/huggingface/cache
#   RESUME_COMPLETED_DIMENSIONS=1
#   PYTHON_BIN=python

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${VBENCH_ROOT:?Set VBENCH_ROOT to the cloned VBench repository}"
: "${VBENCH_CACHE_DIR:?Set VBENCH_CACHE_DIR to the VBench checkpoint cache directory}"
: "${VBENCH_HF_HOME:?Set VBENCH_HF_HOME to a populated Hugging Face cache directory}"
VBENCH_ROOT="$(cd -- "${VBENCH_ROOT}" && pwd)"
mkdir -p "${VBENCH_CACHE_DIR}"
VBENCH_CACHE_DIR="$(cd -- "${VBENCH_CACHE_DIR}" && pwd)"
export VBENCH_CACHE_DIR

if [[ ! -d "${VBENCH_HF_HOME}" ]]; then
    echo "Hugging Face cache directory not found: ${VBENCH_HF_HOME}" >&2
    exit 1
fi
VBENCH_HF_HOME="$(cd -- "${VBENCH_HF_HOME}" && pwd)"
export HF_HOME="${VBENCH_HF_HOME}"
export HF_HUB_CACHE="${VBENCH_HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${VBENCH_HF_HOME}/hub"
export TRANSFORMERS_CACHE="${VBENCH_HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[VBench] Hugging Face cache: ${VBENCH_HF_HOME} (offline)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_LABEL="${MODEL_LABEL:-glance_3k_sample_ode}"
case "${MODEL_LABEL}" in
    glance_3k_sample_ode|glance_3k_sample_ode_8step|glance_one_sample_ode|glance_one_sample_dmd)
        variant="${MODEL_LABEL#glance_}"
        DEFAULT_VIDEOS_PATH="${EVAL_ROOT}/output/vbench_standard_extended_glance_${variant}"
        DEFAULT_OUTPUT_PATH="${EVAL_ROOT}/output/vbench_evaluation_extended_glance_${variant}"
        ;;
    *)
        echo "Unsupported MODEL_LABEL: ${MODEL_LABEL}" >&2
        exit 2
        ;;
esac
VIDEOS_PATH="${VIDEOS_PATH:-${DEFAULT_VIDEOS_PATH}}"
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_OUTPUT_PATH}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SKIP_STANDARD_EVALUATION="${SKIP_STANDARD_EVALUATION:-0}"
SKIP_TEMPORAL_FLICKERING="${SKIP_TEMPORAL_FLICKERING:-0}"
RESUME_COMPLETED_DIMENSIONS="${RESUME_COMPLETED_DIMENSIONS:-1}"
ZIP_PATH="${ZIP_PATH:-${OUTPUT_PATH}.zip}"
FINAL_SCORE_PATH="${FINAL_SCORE_PATH:-${OUTPUT_PATH}/final_scores.json}"
EXTENDED_PROMPT_PATH="${EXTENDED_PROMPT_PATH:-${EVAL_ROOT}/prompts/all_dimension_extended.txt}"
PROTOCOL_DIR="${PROTOCOL_DIR:-${VIDEOS_PATH}/.vbench}"
OUTPUT_NAME_PATH="${PROTOCOL_DIR}/video_basenames.txt"
EXTENDED_FULL_INFO_PATH="${PROTOCOL_DIR}/VBench_full_info_extended.json"

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
    exit 2
fi

for required_file in \
    "${SCRIPT_DIR}/prepare_vbench_extended.py" \
    "${SCRIPT_DIR}/evaluate_vbench_extended.py" \
    "${SCRIPT_DIR}/cal_vbench_final_score.py" \
    "${SCRIPT_DIR}/verify_vbench_cache.py" \
    "${EXTENDED_PROMPT_PATH}" \
    "${VBENCH_ROOT}/evaluate.py" \
    "${VBENCH_ROOT}/scripts/constant.py" \
    "${VBENCH_ROOT}/prompts/all_dimension.txt" \
    "${VBENCH_ROOT}/vbench/VBench_full_info.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -d "${VIDEOS_PATH}" ]]; then
    echo "Generated video directory not found: ${VIDEOS_PATH}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vbench_extended.py" \
    --vbench-root "${VBENCH_ROOT}" \
    --extended-prompts "${EXTENDED_PROMPT_PATH}" \
    --output-dir "${PROTOCOL_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_vbench_cache.py" \
    --cache-dir "${VBENCH_CACHE_DIR}"

"${PYTHON_BIN}" - <<'PY'
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True,
    local_files_only=True,
)
print(f"[VBench] Offline BERT tokenizer ready: {len(tokenizer)} tokens")
PY

missing_count=0
prompt_count=0
while IFS= read -r video_basename; do
    prompt_count=$((prompt_count + 1))
    for sample_index in 0 1 2 3 4; do
        video_path="${VIDEOS_PATH}/${video_basename}-${sample_index}.mp4"
        if [[ ! -f "${video_path}" ]]; then
            if [[ "${missing_count}" -lt 20 ]]; then
                echo "Missing VBench sample: ${video_path}" >&2
            fi
            missing_count=$((missing_count + 1))
        fi
    done
done < "${OUTPUT_NAME_PATH}"

if [[ "${prompt_count}" -ne 946 ]]; then
    echo "Expected 946 prompt entries, found ${prompt_count}." >&2
    exit 1
fi
if [[ "${missing_count}" -ne 0 ]]; then
    echo "VBench sample validation failed: ${missing_count} files are missing." >&2
    exit 1
fi
echo "[VBench] Sample validation passed: 946 prompts x 5 seeds = 4730 videos."

mkdir -p "${OUTPUT_PATH}"

standard_dimensions=(
    subject_consistency
    background_consistency
    motion_smoothness
    dynamic_degree
    aesthetic_quality
    imaging_quality
    object_class
    multiple_objects
    human_action
    color
    spatial_relationship
    scene
    temporal_style
    appearance_style
    overall_consistency
)

run_evaluate() {
    local dimension="$1"
    local status=0
    local args=(
        "${SCRIPT_DIR}/evaluate_vbench_extended.py"
        --vbench_root "${VBENCH_ROOT}"
        --videos_path "${VIDEOS_PATH}"
        --full_json_dir "${EXTENDED_FULL_INFO_PATH}"
        --dimension "${dimension}"
        --output_path "${OUTPUT_PATH}"
        --load_ckpt_from_local True
    )

    echo "[VBench] Evaluating dimension: ${dimension}"
    set +e
    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
        "${PYTHON_BIN}" -m torch.distributed.run \
            --standalone \
            --nproc_per_node="${NPROC_PER_NODE}" \
            "${args[@]}"
    else
        "${PYTHON_BIN}" "${args[@]}"
    fi
    status=$?
    set -e

    if [[ "${status}" -ne 0 ]]; then
        if has_dimension_result "${dimension}"; then
            echo "[VBench] WARNING: ${dimension} returned status ${status} " \
                 "after writing a valid result; continuing." >&2
            return 0
        fi
        echo "[VBench] Dimension ${dimension} failed before producing a result." >&2
        return "${status}"
    fi
}

has_dimension_result() {
    local dimension="$1"
    "${PYTHON_BIN}" - "${OUTPUT_PATH}" "${dimension}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
dimension = sys.argv[2]
for result_path in output_path.glob("*_eval_results.json"):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if dimension in result:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_or_reuse_dimension() {
    local dimension="$1"
    if [[ "${RESUME_COMPLETED_DIMENSIONS}" == "1" ]] && \
       has_dimension_result "${dimension}"; then
        echo "[VBench] Reusing completed dimension: ${dimension}"
    else
        run_evaluate "${dimension}"
    fi
}

cd "${VBENCH_ROOT}"
if [[ "${SKIP_STANDARD_EVALUATION}" == "1" ]]; then
    echo "[VBench] Reusing the existing 15-dimension evaluation in ${OUTPUT_PATH}."
else
    for dimension in "${standard_dimensions[@]}"; do
        run_or_reuse_dimension "${dimension}"
    done
fi

# The authors state that every prompt, including Temporal Flickering, uses
# exactly five seeds. Therefore these five generated samples are evaluated
# directly; the official 25-candidate static-filter extension is not used.
if [[ "${SKIP_TEMPORAL_FLICKERING}" == "1" ]]; then
    echo "[VBench] Reusing the existing Temporal Flickering evaluation."
else
    run_or_reuse_dimension temporal_flickering
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/cal_vbench_final_score.py" \
    --vbench-root "${VBENCH_ROOT}" \
    --results-dir "${OUTPUT_PATH}" \
    --output "${FINAL_SCORE_PATH}"

"${PYTHON_BIN}" - "${OUTPUT_PATH}" "${ZIP_PATH}" <<'PY'
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve()
archive = Path(sys.argv[2]).expanduser().resolve()
temporary = archive.with_name(f"{archive.name}.tmp")

with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            output.write(path, Path(source.name) / path.relative_to(source))

temporary.replace(archive)
print(f"[VBench] Created result archive with Python: {archive}")
PY

echo "[VBench] Evaluation results: ${OUTPUT_PATH}"
echo "[VBench] Result archive: ${ZIP_PATH}"
