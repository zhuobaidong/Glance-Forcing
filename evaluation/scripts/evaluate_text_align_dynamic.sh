#!/usr/bin/env bash
set -euo pipefail

# Activate the VBench environment before running this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${VBENCH_ROOT:?Set VBENCH_ROOT to the cloned VBench repository}"
: "${VBENCH_CACHE_DIR:?Set VBENCH_CACHE_DIR to the VBench checkpoint cache directory}"
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
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_METRICS_ROOT}/dynamic}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RAFT_CHECKPOINT="${VBENCH_CACHE_DIR}/raft_model/models/raft-things.pth"

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
for required_file in     "${VBENCH_ROOT}/evaluate.py"     "${MANIFEST_PATH}"     "${RAFT_CHECKPOINT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

VBENCH_CACHE_DIR="$(cd -- "${VBENCH_CACHE_DIR}" && pwd)"
export VBENCH_CACHE_DIR
echo "[Dynamic] VBench cache: ${VBENCH_CACHE_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_text_align.py"     --manifest "${MANIFEST_PATH}"     --videos-dir "${VIDEOS_PATH}"

mkdir -p "${OUTPUT_PATH}"
args=(
    "${VBENCH_ROOT}/evaluate.py"
    --dimension dynamic_degree
    --videos_path "${VIDEOS_PATH}"
    --mode custom_input
    --output_path "${OUTPUT_PATH}"
)

cd "${VBENCH_ROOT}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run         --standalone         --nproc_per_node="${NPROC_PER_NODE}"         "${args[@]}"
else
    "${PYTHON_BIN}" "${args[@]}"
fi

echo "[Dynamic] Complete: ${OUTPUT_PATH}"
