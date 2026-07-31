#!/usr/bin/env python3
"""Summarize Dynamic, VisionReward, and Instruction metrics."""

import argparse
import json
import statistics
from pathlib import Path

from validate_text_align import load_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vision-results-dir", type=Path, required=True)
    parser.add_argument("--dynamic-results-dir", type=Path)
    parser.add_argument("--vbench-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def load_vision_results(directory: Path):
    records = {}
    sources = []
    paths = sorted(directory.glob("visionreward-part-*.jsonl"))
    if not paths:
        raise ValueError(f"No VisionReward JSONL files in {directory}")
    for path in paths:
        sources.append(str(path))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from error
                index = record.get("index")
                if index in records:
                    if records[index] != record:
                        raise ValueError(
                            f"Conflicting VisionReward index {index}"
                        )
                    continue
                records[index] = record
    return records, sources


def load_dynamic_result(directory: Path):
    candidates = []
    for path in sorted(directory.glob("*_eval_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "dynamic_degree" not in result:
            continue
        value = result["dynamic_degree"]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Malformed dynamic_degree in {path}")
        details = value[1] if len(value) > 1 else []
        candidates.append((path, float(value[0]), details))
    if not candidates:
        raise ValueError(f"No Dynamic result found in {directory}")
    scores = {candidate[1] for candidate in candidates}
    if len(scores) != 1:
        raise ValueError(
            "Conflicting Dynamic result files: "
            + ", ".join(str(item[0]) for item in candidates)
        )
    path, score, details = candidates[-1]
    return score, details, [str(item[0]) for item in candidates]


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest, args.expected_count)
    vision_records, vision_sources = load_vision_results(
        args.vision_results_dir
    )
    expected_indices = set(range(args.expected_count))
    unknown = sorted(set(vision_records) - expected_indices)
    missing = sorted(expected_indices - set(vision_records))
    if unknown:
        raise ValueError(f"Unknown VisionReward indices: {unknown}")
    if args.require_complete and missing:
        raise ValueError(
            f"VisionReward is incomplete; missing {len(missing)} indices"
        )

    ordered = [
        vision_records[index]
        for index in sorted(vision_records)
    ]
    vision_raw = (
        statistics.fmean(
            float(record["visionreward_score"])
            for record in ordered
        )
        if ordered else None
    )
    instruction_values = [
        int(record["instruction_value"])
        for record in ordered
    ]
    instruction_raw = (
        statistics.fmean(instruction_values)
        if instruction_values else None
    )
    yes_count = instruction_values.count(1)
    no_count = instruction_values.count(-1)

    summary = {
        "protocol": manifest["protocol"],
        "model_label": manifest.get("model_label"),
        "checkpoint": manifest.get("checkpoint"),
        "num_prompts": args.expected_count,
        "visionreward": {
            "complete": not missing,
            "num_scored": len(ordered),
            "missing_indices": missing,
            "raw": vision_raw,
            "x100": None if vision_raw is None else vision_raw * 100,
            "sources": vision_sources,
        },
        "instruction": {
            "complete": not missing,
            "num_scored": len(ordered),
            "yes": yes_count,
            "no": no_count,
            "raw": instruction_raw,
            "x100": (
                None
                if instruction_raw is None
                else instruction_raw * 100
            ),
            "sources": vision_sources,
        },
    }

    if args.vbench_summary is not None:
        try:
            vbench_summary = json.loads(
                args.vbench_summary.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid VBench summary: {args.vbench_summary}"
            ) from error
        vbench_metrics = {
            "total": "total_score",
            "quality": "quality_score",
            "semantic": "semantic_score",
        }
        for metric_name, source_key in vbench_metrics.items():
            if source_key not in vbench_summary:
                raise ValueError(
                    f"VBench summary is missing {source_key}"
                )
            raw_value = float(vbench_summary[source_key])
            summary[metric_name] = {
                "raw": raw_value,
                "x100": raw_value * 100,
                "source": str(args.vbench_summary),
            }

    if args.dynamic_results_dir is not None:
        dynamic_raw, details, dynamic_sources = load_dynamic_result(
            args.dynamic_results_dir
        )
        if args.require_complete and len(details) != args.expected_count:
            raise ValueError(
                f"Dynamic result has {len(details)} videos, expected "
                f"{args.expected_count}"
            )
        summary["dynamic"] = {
            "complete": len(details) == args.expected_count,
            "num_scored": len(details),
            "raw": dynamic_raw,
            "x100": dynamic_raw * 100,
            "sources": dynamic_sources,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[text-align] Summary: {args.output}")


if __name__ == "__main__":
    main()
