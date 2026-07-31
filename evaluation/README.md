# Glance-Forcing Six-Metric Evaluation

This directory contains the code and prompts needed to reproduce the
six-metric evaluation with the modified `infer_glance.py` already included in
this repository:

- **Total Quality / Semantic**: VBench extended protocol, 946 prompts, five
  seeds per prompt (4,730 videos).
- **Dynamic / VisionReward / Instruction**: `text_align.txt`, 100 prompts,
  one video per prompt.

Model weights, generated videos, caches, and evaluation results are not
included.

## Tested upstream versions

- [Glance-Forcing](https://github.com/zhuobaidong/Glance-Forcing):
  `cc76a1d776751d4063a50614b9a40d558357171c`
- [VBench](https://github.com/Vchitect/VBench):
  `45e79ec14e69a2187202c675d2dbce1a71843d53`
- [VisionReward](https://github.com/zai-org/VisionReward):
  `511960d61a777b2008cd486a85a090ba96792a32`

Generation, VBench, and VisionReward may use separate conda environments.

## 1. Clone evaluation dependencies

Clone the tested VBench and VisionReward revisions next to this repository:

```bash
git clone https://github.com/Vchitect/VBench.git
git -C VBench checkout 45e79ec14e69a2187202c675d2dbce1a71843d53

git clone https://github.com/zai-org/VisionReward.git
git -C VisionReward checkout 511960d61a777b2008cd486a85a090ba96792a32
```

Follow their official environment instructions. No Glance-Forcing patch is
needed: this repository already contains the modified inference code.

Expected checkpoint layout:

```text
Glance-Forcing-checkpoints/
├── chunkwise/ar_diffusion.pt
├── 3k_sample_ode/{slow_lora.pt,fast_lora.pt}
├── one_sample_ode/{slow_lora.pt,fast_lora.pt}
└── one_sample_dmd/{slow_lora.pt,fast_lora.pt}
```

## 2. Set paths and select a model

```bash
export GLANCE_ROOT=/path/to/Glance-forcing-eval
export EVAL_ROOT="$GLANCE_ROOT/evaluation"
export GLANCE_CKPT_ROOT=/path/to/Glance-forcing-eval-checkpoints
export VBENCH_ROOT=/path/to/VBench
export VBENCH_CACHE_DIR=/path/to/vbench_cache
export VBENCH_HF_HOME=/path/to/huggingface-cache
export VISIONREWARD_ROOT=/path/to/VisionReward
export VISIONREWARD_MODEL=/path/to/VisionReward-Video
```

Choose exactly one run configuration:

```bash
# 3K Sample ODE, 4 steps
export MODEL_VARIANT=3k_sample_ode
export RUN_TAG=3k_sample_ode
export STEPS=4

# 3K Sample ODE, 8 steps
# export MODEL_VARIANT=3k_sample_ode
# export RUN_TAG=3k_sample_ode_8step
# export STEPS=8

# One Sample ODE, 4 steps
# export MODEL_VARIANT=one_sample_ode
# export RUN_TAG=one_sample_ode
# export STEPS=4

# One Sample DMD, 4 steps
# export MODEL_VARIANT=one_sample_dmd
# export RUN_TAG=one_sample_dmd
# export STEPS=4

export MODEL_LABEL="glance_${RUN_TAG}"
```

Each `RUN_TAG` has its own output directories, so different runs do not
overwrite one another.

## 3. Download models and evaluation caches

The following command downloads the Wan2.1 base model, all three Glance model
variants, VisionReward-Video, all required VBench checkpoints, and the offline
`bert-base-uncased` tokenizer. It requires network access and substantial disk
space, but can be rerun after an interrupted download.

```bash
"$EVAL_ROOT/scripts/download_models.sh"
```

After downloading, the script validates the VBench checkpoint cache. The later
generation and evaluation scripts also fail early if required files are missing.

## 4. Generate both video suites

Activate the Glance-Forcing generation environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_VARIANT="$MODEL_VARIANT" \
VBENCH_ROOT="$VBENCH_ROOT" \
GLANCE_ROOT="$GLANCE_ROOT" \
GLANCE_CKPT_ROOT="$GLANCE_CKPT_ROOT" \
OUTPUT_FOLDER="$EVAL_ROOT/output/vbench_standard_extended_glance_$RUN_TAG" \
NPROC_PER_NODE=4 \
BASE_SEED=1000 \
STEPS="$STEPS" \
"$EVAL_ROOT/scripts/generate_glance_vbench.sh"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_VARIANT="$MODEL_VARIANT" \
MODEL_LABEL="$MODEL_LABEL" \
GLANCE_ROOT="$GLANCE_ROOT" \
GLANCE_CKPT_ROOT="$GLANCE_CKPT_ROOT" \
OUTPUT_FOLDER="$EVAL_ROOT/output/text_align_glance_$RUN_TAG" \
NPROC_PER_NODE=4 \
BASE_SEED=0 \
STEPS="$STEPS" \
"$EVAL_ROOT/scripts/generate_glance_text_align.sh"
```

Generation validates that the VBench suite has 4,730 videos and the
text-alignment suite has 100 videos.

## 5. Evaluate Total Quality / Semantic

Activate the VBench environment. `VBENCH_CACHE_DIR` contains VBench model
checkpoints; `VBENCH_HF_HOME` must already contain the
`bert-base-uncased` tokenizer because evaluation runs offline.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VBENCH_ROOT="$VBENCH_ROOT" \
VBENCH_CACHE_DIR="$VBENCH_CACHE_DIR" \
VBENCH_HF_HOME="$VBENCH_HF_HOME" \
MODEL_LABEL="$MODEL_LABEL" \
VIDEOS_PATH="$EVAL_ROOT/output/vbench_standard_extended_glance_$RUN_TAG" \
OUTPUT_PATH="$EVAL_ROOT/output/vbench_evaluation_extended_glance_$RUN_TAG" \
NPROC_PER_NODE=4 \
RESUME_COMPLETED_DIMENSIONS=1 \
"$EVAL_ROOT/scripts/evaluate_vbench.sh"
```

The three scores are written to
`output/vbench_evaluation_extended_glance_$RUN_TAG/final_scores.json`.

## 6. Evaluate Dynamic

Still in the VBench environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
VBENCH_ROOT="$VBENCH_ROOT" \
VBENCH_CACHE_DIR="$VBENCH_CACHE_DIR" \
MODEL_LABEL="$MODEL_LABEL" \
VIDEOS_PATH="$EVAL_ROOT/output/text_align_glance_$RUN_TAG" \
OUTPUT_PATH="$EVAL_ROOT/output/text_align_metrics_glance_$RUN_TAG/dynamic" \
NPROC_PER_NODE=1 \
"$EVAL_ROOT/scripts/evaluate_text_align_dynamic.sh"
```

## 7. Evaluate VisionReward / Instruction

Activate the VisionReward environment:

```bash
VISIONREWARD_ROOT="$VISIONREWARD_ROOT" \
MODEL_PATH="$VISIONREWARD_MODEL" \
MODEL_LABEL="$MODEL_LABEL" \
VIDEOS_PATH="$EVAL_ROOT/output/text_align_glance_$RUN_TAG" \
RESULTS_DIR="$EVAL_ROOT/output/text_align_metrics_glance_$RUN_TAG/visionreward" \
GPU_IDS=0,1,2,3 \
"$EVAL_ROOT/scripts/evaluate_text_align_visionreward.sh"
```

The command safely resumes existing JSONL shards.

## 8. Produce one six-score JSON

```bash
python "$EVAL_ROOT/scripts/summarize_text_align_metrics.py" \
  --manifest "$EVAL_ROOT/output/text_align_glance_$RUN_TAG/.text_align/manifest.json" \
  --vision-results-dir "$EVAL_ROOT/output/text_align_metrics_glance_$RUN_TAG/visionreward" \
  --dynamic-results-dir "$EVAL_ROOT/output/text_align_metrics_glance_$RUN_TAG/dynamic" \
  --vbench-summary "$EVAL_ROOT/output/vbench_evaluation_extended_glance_$RUN_TAG/final_scores.json" \
  --output "$EVAL_ROOT/output/six_metrics_glance_$RUN_TAG.json" \
  --require-complete
```

All generated artifacts stay under `output/`, which is ignored by Git.

## Repository contents

```text
prompts/   Extended VBench and text-alignment prompts
scripts/   Generation, validation, evaluation, and summary scripts
```

Do not commit checkpoints, caches, generated MP4 files, JSONL result shards,
tokens, or machine-specific environment files.
