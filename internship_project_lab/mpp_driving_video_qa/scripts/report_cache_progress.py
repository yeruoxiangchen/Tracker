#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import load_jsonl
from mpp_video_qa.data import cache_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    unique = {}
    for split in ("train", "dev", "eval"):
        for row in load_jsonl(args.data_dir / (split + ".jsonl")):
            unique.setdefault(row["segment_id"], row)
    present = sum(cache_file(args.cache_dir, segment).is_file() for segment in unique)
    rank_progress = []
    for path in sorted(args.cache_dir.glob("progress_rank_*.json")):
        try:
            rank_progress.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    summary_path = args.cache_dir / "cache_summary.json"
    completed = json.loads(summary_path.read_text(encoding="utf-8")) \
        if summary_path.is_file() else None
    result = {"status": "completed" if completed and present == len(unique) else "running_or_not_started",
              "segments_present": present, "segments_total": len(unique),
              "missing": len(unique) - present, "rank_progress": rank_progress,
              "final_summary": completed}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
