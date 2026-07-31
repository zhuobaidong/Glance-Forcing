#!/usr/bin/env python3
"""Prepare the 100-prompt Causal Forcing text-alignment protocol."""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--aux-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional model weights recorded in the manifest. "
            "Repeat for models composed from multiple checkpoints, such as "
            "Glance slow/fast LoRA adapters."
        ),
    )
    parser.add_argument(
        "--model-label",
        default="causal_forcing",
        help="Logical model/stage label recorded in the manifest",
    )
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--inference-steps", type=int)
    return parser.parse_args()


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def file_identity(path: Path):
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main():
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.sample_index < 0 or args.base_seed < 0:
        raise ValueError("--sample-index and --base-seed must be non-negative")

    raw = args.prompts.read_text(encoding="utf-8")
    prompts = [
        line.rstrip()
        for line in raw.splitlines()
        if line.strip()
    ]
    if len(prompts) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} non-empty prompts, got "
            f"{len(prompts)} from {args.prompts}"
        )
    if len(set(prompts)) != len(prompts):
        raise ValueError("text_align prompts must be unique")

    basenames = [
        f"text_align_{index:03d}"
        for index in range(len(prompts))
    ]
    records = []
    for index, (prompt, basename) in enumerate(zip(prompts, basenames)):
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "effective_seed": args.base_seed + index,
                "video_basename": basename,
                "video_filename": f"{basename}-{args.sample_index}.mp4",
            }
        )

    manifest = {
        "protocol": "causal_forcing_text_align_v1",
        "prompt_source": str(args.prompts.resolve()),
        "prompt_source_sha256": hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest(),
        "num_prompts": len(prompts),
        "sample_index": args.sample_index,
        "base_seed": args.base_seed,
        "seed_mode": "base_seed_plus_prompt_index",
        "model_label": args.model_label,
        "config": file_identity(args.config),
        "checkpoint": file_identity(args.checkpoint),
        "records": records,
    }
    if args.aux_checkpoint:
        manifest["aux_checkpoints"] = [
            file_identity(path) for path in args.aux_checkpoint
        ]
    if args.inference_steps is not None:
        manifest["inference_steps"] = args.inference_steps

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable_keys = (
            "prompt_source_sha256",
            "sample_index",
            "base_seed",
            "seed_mode",
            "model_label",
            "config",
            "checkpoint",
        )
        if args.aux_checkpoint:
            stable_keys += ("aux_checkpoints",)
        if (
            args.inference_steps is not None
            and ("inference_steps" in existing or args.inference_steps != 4)
        ):
            stable_keys += ("inference_steps",)
        changed = [
            key for key in stable_keys
            if existing.get(key) != manifest.get(key)
        ]
        if changed or existing.get("records") != records:
            raise ValueError(
                f"Existing protocol differs ({changed or ['records']}); "
                "use a new OUTPUT_FOLDER instead of reusing old videos"
            )

    write_text(
        args.output_dir / "prompts.txt",
        "\n".join(prompts) + "\n",
    )
    write_text(
        args.output_dir / "video_basenames.txt",
        "\n".join(basenames) + "\n",
    )
    write_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    print(
        f"[text-align] Prepared {len(prompts)} prompts in "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
