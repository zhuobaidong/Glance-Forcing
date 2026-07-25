### Installation
```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```
### Download Checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir wan_models/Wan2.1-T2V-14B
# base model
hf download zhuhz22/Causal-Forcing chunkwise/ar_diffusion.pt --local-dir checkpoints
# dataset
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
# slow lora and fast lora
wget https://huggingface.co/zhuobai/Glance-Forcing/resolve/main/fast_lora.pt
wget https://huggingface.co/zhuobai/Glance-Forcing/resolve/main/fast_lora.pt
```

### 💡 如果下载速度较慢，可以试试下面这个镜像站
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Training

首先把 trainer/distillation_lora.py 的 103 行附件换成真实的 lora 本地路径，然后运行下面的指令:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint localhost:29503 \
  train.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --logdir logs/causal_forcing_dmd_chunkwise
```
