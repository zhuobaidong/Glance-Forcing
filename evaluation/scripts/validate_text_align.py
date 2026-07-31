#!/usr/bin/env python3
"""Validate the text-align manifest and its generated videos."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    return parser.parse_args()


def load_manifest(path: Path, expected_count: int):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Missing records list in {path}")
    if manifest.get("num_prompts") != expected_count:
        raise ValueError(
            f"Manifest num_prompts is {manifest.get('num_prompts')}, "
            f"expected {expected_count}"
        )
    if len(records) != expected_count:
        raise ValueError(
            f"Manifest contains {len(records)} records, expected "
            f"{expected_count}"
        )
    indices = [record.get("index") for record in records]
    if indices != list(range(expected_count)):
        raise ValueError("Manifest indices must be contiguous from zero")
    prompts = [record.get("prompt") for record in records]
    filenames = [record.get("video_filename") for record in records]
    if any(
        not isinstance(prompt, str) or not prompt.strip()
        for prompt in prompts
    ):
        raise ValueError("Manifest contains an empty prompt")
    if len(set(prompts)) != expected_count:
        raise ValueError("Manifest prompts are not unique")
    if len(set(filenames)) != expected_count:
        raise ValueError("Manifest video filenames are not unique")
    return manifest


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest, args.expected_count)
    expected_names = {
        record["video_filename"]
        for record in manifest["records"]
    }
    actual_paths = sorted(args.videos_dir.glob("*.mp4"))
    actual_names = {path.name for path in actual_paths}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    empty = sorted(
        path.name for path in actual_paths
        if path.stat().st_size == 0
    )
    if missing or unexpected or empty:
        messages = []
        if missing:
            messages.append(f"missing={missing[:10]}")
        if unexpected:
            messages.append(f"unexpected={unexpected[:10]}")
        if empty:
            messages.append(f"empty={empty[:10]}")
        raise ValueError("Video validation failed: " + "; ".join(messages))
    print(
        f"[text-align] Validation passed: {len(actual_paths)} videos "
        f"in {args.videos_dir}"
    )


if __name__ == "__main__":
    main()
