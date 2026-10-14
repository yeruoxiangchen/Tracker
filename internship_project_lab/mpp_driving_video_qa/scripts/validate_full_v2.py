#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.data import cache_file
from mpp_video_qa.model import FEATURE_STAGE_PROJECTED, FEATURE_STAGE_QFORMER, FEATURE_STAGES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--feature-stage", choices=FEATURE_STAGES,
                        default=FEATURE_STAGE_PROJECTED)
    args = parser.parse_args()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {}
    for split in ("train", "dev", "eval"):
        path = args.data_dir / (split + ".jsonl")
        rows[split] = load_jsonl(path)
        expected = manifest.get("outputs", {}).get(split, {}).get("sha256")
        if expected and file_sha256(path) != expected:
            raise RuntimeError("dataset fingerprint mismatch: %s" % path)
    segment_sets = {split: {row["segment_id"] for row in values}
                    for split, values in rows.items()}
    for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        if segment_sets[left] & segment_sets[right]:
            raise RuntimeError("segment overlap between %s and %s" % (left, right))
    summary = {"valid": True,
               "questions": {split: len(values) for split, values in rows.items()},
               "segments": {split: len(values) for split, values in segment_sets.items()}}
    if args.cache_dir:
        unique = {row["segment_id"] for values in rows.values() for row in values}
        missing = []
        invalid = []
        expected_hidden = 768 if args.feature_stage == FEATURE_STAGE_QFORMER else 4096
        for segment in unique:
            path = cache_file(args.cache_dir, segment)
            if not path.is_file():
                missing.append(segment)
        cache_summary_path = args.cache_dir / "cache_summary.json"
        cache_summary = (json.loads(cache_summary_path.read_text(encoding="utf-8"))
                         if cache_summary_path.is_file() else {})
        summary_stage = cache_summary.get("feature_stage")
        if summary_stage is None and args.feature_stage == FEATURE_STAGE_PROJECTED:
            summary_stage = FEATURE_STAGE_PROJECTED
        if summary_stage != args.feature_stage:
            invalid.append("cache_summary_feature_stage")
        # Inspect a deterministic sample. Dataset loading enforces this same
        # contract for every record without turning validation into 28k reads.
        for segment in sorted(unique)[:8]:
            path = cache_file(args.cache_dir, segment)
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu")
            features = payload.get("features")
            stage = payload.get("feature_stage")
            if stage is None and getattr(features, "shape", (0,))[-1:] == (4096,):
                stage = FEATURE_STAGE_PROJECTED
            if stage != args.feature_stage or getattr(features, "shape", (0,))[-1:] != (expected_hidden,):
                invalid.append(segment)
        summary["cache"] = {"directory": str(args.cache_dir.resolve()),
                            "segments": len(unique), "missing": len(missing),
                            "missing_examples": missing[:10],
                            "feature_stage": args.feature_stage,
                            "invalid": len(invalid), "invalid_examples": invalid[:10]}
        if missing or invalid:
            print(json.dumps(summary, indent=2))
            raise RuntimeError("visual cache is invalid: %d missing, %d wrong-stage segments" %
                               (len(missing), len(invalid)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
