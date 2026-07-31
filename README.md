# Glance-forcing-eval

This branch includes the modified multi-GPU `infer_glance.py` and a complete six-metric evaluation workflow. No patch is required. See [evaluation/README.md](evaluation/README.md) for VBench and VisionReward clone commands, tested revisions, video generation, and score aggregation. Model weights, caches, videos, and results are excluded from Git.



This repository is based on [zhuobaidong/Glance-Forcing](https://github.com/zhuobaidong/Glance-Forcing) commit `cc76a1d776751d4063a50614b9a40d558357171c`.

## Included changes

- Modified `infer_glance.py` with correct rank-local GPU placement.
- Complete and non-overlapping distributed prompt partitioning.
- Deterministic per-prompt seeds and collision-safe filenames.
- 4-step and 8-step generation.
- Six-metric evaluation code and prompts under `evaluation/`.

## Installation

```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Evaluation

Follow [evaluation/README.md](evaluation/README.md). It contains the exact `git clone` commands and tested commits for VBench and VisionReward, an automatic model/cache downloader, the checkpoint layout, all generation/evaluation commands, and final six-score aggregation.

## Excluded artifacts

Model weights, VBench/VisionReward caches, Hugging Face caches, generated videos, JSONL shards and evaluation outputs are ignored by Git.

## License

Apache-2.0. See [LICENSE](LICENSE).
