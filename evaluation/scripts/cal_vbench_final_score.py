#!/usr/bin/env python3
"""Calculate VBench scores directly from one evaluation output directory."""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.vbench_root / "scripts"))
    from constant import (  # pylint: disable=import-error,import-outside-toplevel
        DIM_WEIGHT,
        NORMALIZE_DIC,
        QUALITY_LIST,
        QUALITY_WEIGHT,
        SEMANTIC_LIST,
        SEMANTIC_WEIGHT,
        TASK_INFO,
    )

    expected = {name.replace(" ", "_") for name in TASK_INFO}
    candidates = {}
    for result_path in sorted(args.results_dir.glob("*_eval_results.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid result JSON: {result_path}") from error
        if not isinstance(result, dict):
            continue
        for dimension, value in result.items():
            if dimension not in expected:
                continue
            candidates.setdefault(dimension, []).append((result_path, value))

    missing = sorted(expected - set(candidates))
    if missing:
        raise ValueError(
            "Cannot calculate final score; missing dimensions: "
            + ", ".join(missing)
        )

    raw_scores = {}
    sources = {}
    for dimension in sorted(expected):
        dimension_candidates = candidates[dimension]
        if len(dimension_candidates) > 1:
            paths = ", ".join(str(path) for path, _ in dimension_candidates)
            raise ValueError(
                f"Duplicate result files for {dimension}: {paths}. "
                "Use a clean OUTPUT_PATH or remove the obsolete result."
            )
        result_path, value = dimension_candidates[0]
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"Malformed score for {dimension} in {result_path}"
            )
        raw_scores[dimension.replace("_", " ")] = float(value[0])
        sources[dimension] = str(result_path)

    normalized_weighted = {}
    for dimension in TASK_INFO:
        bounds = NORMALIZE_DIC[dimension]
        normalized = (
            (raw_scores[dimension] - bounds["Min"])
            / (bounds["Max"] - bounds["Min"])
        )
        normalized_weighted[dimension] = normalized * DIM_WEIGHT[dimension]

    quality_score = sum(
        normalized_weighted[name] for name in QUALITY_LIST
    ) / sum(DIM_WEIGHT[name] for name in QUALITY_LIST)
    semantic_score = sum(
        normalized_weighted[name] for name in SEMANTIC_LIST
    ) / sum(DIM_WEIGHT[name] for name in SEMANTIC_LIST)
    total_score = (
        quality_score * QUALITY_WEIGHT
        + semantic_score * SEMANTIC_WEIGHT
    ) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    final_result = {
        "raw_dimension_scores": raw_scores,
        "normalized_weighted_dimension_scores": normalized_weighted,
        "quality_score": quality_score,
        "semantic_score": semantic_score,
        "total_score": total_score,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(final_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("+------------------|------------------+")
    print(f"|     quality score|{quality_score}|")
    print(f"|    semantic score|{semantic_score}|")
    print(f"|       total score|{total_score}|")
    print("+------------------|------------------+")
    print(f"[VBench] Final score JSON: {args.output}")


if __name__ == "__main__":
    main()
