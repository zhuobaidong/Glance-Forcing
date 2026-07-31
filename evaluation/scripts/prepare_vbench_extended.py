#!/usr/bin/env python3
"""Prepare collision-safe metadata for the Self-Forcing VBench prompts."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--extended-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def safe_prefix(prompt):
    return prompt[:100].replace("/", "_").replace("\\", "_")


def main():
    args = parse_args()
    original_path = args.vbench_root / "prompts" / "all_dimension.txt"
    full_info_path = args.vbench_root / "vbench" / "VBench_full_info.json"

    original_prompts = read_lines(original_path)
    extended_prompts = read_lines(args.extended_prompts)
    with full_info_path.open("r", encoding="utf-8") as handle:
        full_info = json.load(handle)

    counts = {
        "official prompts": len(original_prompts),
        "extended prompts": len(extended_prompts),
        "VBench metadata entries": len(full_info),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"Prompt/metadata counts do not match: {counts}")
    if any(not prompt for prompt in extended_prompts):
        raise ValueError("The extended prompt file contains an empty line")

    metadata_prompts = [entry["prompt_en"] for entry in full_info]
    if original_prompts != metadata_prompts:
        raise ValueError(
            "VBench all_dimension.txt is not line-aligned with "
            "VBench_full_info.json"
        )

    prefixes = [safe_prefix(prompt) for prompt in extended_prompts]
    prefix_counts = Counter(prefixes)
    video_basenames = []
    for index, prefix in enumerate(prefixes):
        if prefix_counts[prefix] == 1:
            basename = prefix
        else:
            basename = f"{prefix}__vbench_{index:04d}"
        video_basenames.append(basename)

    if len(set(video_basenames)) != len(video_basenames):
        raise ValueError("Failed to construct unique video basenames")

    extended_full_info = []
    manifest = []
    for index, (entry, extended_prompt, basename) in enumerate(
        zip(full_info, extended_prompts, video_basenames)
    ):
        updated_entry = copy.deepcopy(entry)
        updated_entry["original_prompt_en"] = entry["prompt_en"]
        updated_entry["prompt_en"] = extended_prompt
        updated_entry["video_basename"] = basename
        updated_entry["vbench_prompt_index"] = index
        extended_full_info.append(updated_entry)
        manifest.append(
            {
                "index": index,
                "original_prompt": entry["prompt_en"],
                "extended_prompt": extended_prompt,
                "video_basename": basename,
                "dimension": entry["dimension"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    names_path = args.output_dir / "video_basenames.txt"
    metadata_path = args.output_dir / "VBench_full_info_extended.json"
    manifest_path = args.output_dir / "prompt_manifest.json"

    names_path.write_text(
        "".join(f"{name}\n" for name in video_basenames),
        encoding="utf-8",
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(extended_full_info, handle, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    collision_entries = sum(
        count for count in prefix_counts.values() if count > 1
    )
    collision_groups = sum(
        1 for count in prefix_counts.values() if count > 1
    )
    print(f"[VBench] Prepared {len(extended_prompts)} extended prompts")
    print(
        "[VBench] Prompt[:100] collisions: "
        f"{collision_entries} entries in {collision_groups} groups"
    )
    print(f"[VBench] Video-name map: {names_path}")
    print(f"[VBench] Extended metadata: {metadata_path}")


if __name__ == "__main__":
    main()
