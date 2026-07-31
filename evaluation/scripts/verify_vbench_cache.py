#!/usr/bin/env python3
"""Safely verify original VBench checkpoints without calling torch.load()."""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


# relative path: (exact byte size, official SHA256 when published)
EXPECTED = {
    "clip_model/ViT-B-32.pt": (
        353_976_522,
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    ),
    "clip_model/ViT-L-14.pt": (
        932_768_134,
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    ),
    "grit_model/grit_b_densecap_objectdet.pth": (
        417_381_733,
        "53b6e9b3fd948eac55b574c9b6f94ad0743dff46ba449df7ac2d33009ee92ef1",
    ),
    "umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth": (
        607_024_201,
        "bfee78a03bf806fbc0c216309e8f92792011b7db6556fa860e9d869c1e69fecd",
    ),
    "ViCLIP/ViClip-InternVid-10M-FLT.pth": (
        1_710_545_812,
        "7a4d6ad6eac6632db3693f4b97f9f8f6445b65b1e139a6fc6686150522238c56",
    ),
    "amt_model/amt-s.pth": (
        12_038_359,
        "07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77",
    ),
    "caption_model/tag2text_swin_14m.pth": (
        4_478_705_095,
        "4ce96f0ce98f940a6680d567f66a38ccc9ca8c4e638e5f5c5c2e881a0e3502ac",
    ),
    "dino_model/dino_vitbase16_pretrain.pth": (343_242_485, None),
    "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth": (4_071, None),
    "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth": (108_610_983, None),
    "raft_model/models/raft-things.pth": (21_108_000, None),
}

REQUIRED_CODE = (
    "dino_model/facebookresearch_dino_main/hubconf.py",
    "dino_model/facebookresearch_dino_main/vision_transformer.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(cache_dir, relative_path, expected_size, expected_sha256):
    path = cache_dir / relative_path
    errors = []
    if not path.is_file():
        return [f"missing: {path}"]

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        errors.append(
            f"wrong size: {path} (got {actual_size}, expected {expected_size})"
        )

    with path.open("rb") as stream:
        header = stream.read(512).lower()
    if (header.startswith(b"<!doctype html") or header.startswith(b"<html")
            or b"git-lfs.github.com/spec" in header):
        errors.append(f"HTML/LFS pointer instead of checkpoint: {path}")

    if expected_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"wrong SHA256: {path} (got {actual_sha256}, "
                f"expected {expected_sha256})"
            )

    if not zipfile.is_zipfile(path):
        errors.append(f"invalid or truncated PyTorch ZIP checkpoint: {path}")
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                errors.append(f"bad ZIP CRC in {path}: {bad_member}")
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"bad ZIP structure: {path}: {exc}")

    if not errors:
        print(f"[VBench cache] OK: {relative_path}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    cache_dir = parser.parse_args().cache_dir.expanduser().resolve()

    errors = []
    for relative_path, (size, sha256) in EXPECTED.items():
        errors.extend(verify_checkpoint(cache_dir, relative_path, size, sha256))

    for relative_path in REQUIRED_CODE:
        path = cache_dir / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {path}")
        else:
            print(f"[VBench cache] OK: {relative_path}")

    if errors:
        print("\n[VBench cache] Verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"[VBench cache] All {len(EXPECTED)} checkpoints passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
