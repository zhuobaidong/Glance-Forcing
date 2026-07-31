#!/usr/bin/env bash
set -euo pipefail

# Download every model used by Glance generation and the six-metric evaluation.
# This is intentionally explicit because the evaluation scripts run VBench
# offline after validating the cache.

: "${GLANCE_ROOT:?Set GLANCE_ROOT to this Glance-Forcing checkout}"
: "${GLANCE_CKPT_ROOT:?Set GLANCE_CKPT_ROOT to the checkpoint destination}"
: "${VISIONREWARD_MODEL:?Set VISIONREWARD_MODEL to the model destination}"
: "${VBENCH_CACHE_DIR:?Set VBENCH_CACHE_DIR to the VBench cache destination}"
: "${VBENCH_HF_HOME:?Set VBENCH_HF_HOME to the Hugging Face cache destination}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_BIN="${HF_BIN:-hf}"

for command_name in "${HF_BIN}" wget unzip git; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command not found: ${command_name}" >&2
        exit 1
    fi
done

download_file() {
    local url="$1"
    local output="$2"
    mkdir -p "$(dirname -- "${output}")"
    wget --continue --output-document="${output}" "${url}"
}

echo "[1/5] Wan2.1 base model"
"${HF_BIN}" download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir "${GLANCE_ROOT}/wan_models/Wan2.1-T2V-1.3B"

echo "[2/5] Glance-Forcing base checkpoint and LoRAs"
"${HF_BIN}" download zhuobai/Glance-Forcing \
    chunkwise/ar_diffusion.pt \
    3k_sample_ode/slow_lora.pt \
    3k_sample_ode/fast_lora.pt \
    one_sample_ode/slow_lora.pt \
    one_sample_ode/fast_lora.pt \
    one_sample_dmd/slow_lora.pt \
    one_sample_dmd/fast_lora.pt \
    --local-dir "${GLANCE_CKPT_ROOT}"

echo "[3/5] VisionReward-Video"
"${HF_BIN}" download THUDM/VisionReward-Video \
    --local-dir "${VISIONREWARD_MODEL}"

echo "[4/5] VBench checkpoints"
download_file \
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
    "${VBENCH_CACHE_DIR}/clip_model/ViT-B-32.pt"
download_file \
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
    "${VBENCH_CACHE_DIR}/clip_model/ViT-L-14.pt"
download_file \
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth" \
    "${VBENCH_CACHE_DIR}/grit_model/grit_b_densecap_objectdet.pth"
download_file \
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
    "${VBENCH_CACHE_DIR}/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth"
download_file \
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth" \
    "${VBENCH_CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth"
download_file \
    "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz" \
    "${VBENCH_CACHE_DIR}/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
download_file \
    "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth" \
    "${VBENCH_CACHE_DIR}/caption_model/tag2text_swin_14m.pth"
download_file \
    "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth" \
    "${VBENCH_CACHE_DIR}/amt_model/amt-s.pth"
download_file \
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
    "${VBENCH_CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth"
download_file \
    "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth" \
    "${VBENCH_CACHE_DIR}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
download_file \
    "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
    "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"

dino_dir="${VBENCH_CACHE_DIR}/dino_model/facebookresearch_dino_main"
if [[ ! -f "${dino_dir}/hubconf.py" ]]; then
    git clone https://github.com/facebookresearch/dino.git "${dino_dir}"
fi

raft_model="${VBENCH_CACHE_DIR}/raft_model/models/raft-things.pth"
if [[ ! -f "${raft_model}" ]]; then
    raft_zip="${VBENCH_CACHE_DIR}/raft_model/models.zip"
    download_file \
        "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" \
        "${raft_zip}"
    unzip -o "${raft_zip}" -d "${VBENCH_CACHE_DIR}/raft_model"
fi

echo "[5/5] bert-base-uncased tokenizer"
mkdir -p "${VBENCH_HF_HOME}"
HF_HOME="${VBENCH_HF_HOME}" "${PYTHON_BIN}" - <<'PY'
from transformers import BertTokenizer

BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
print("bert-base-uncased tokenizer downloaded")
PY

"${PYTHON_BIN}" \
    "${GLANCE_ROOT}/evaluation/scripts/verify_vbench_cache.py" \
    --cache-dir "${VBENCH_CACHE_DIR}"

echo "All requested models and VBench checkpoints are ready."
