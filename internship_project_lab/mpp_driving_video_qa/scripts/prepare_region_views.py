#!/usr/bin/env python3
"""Keep original QA unchanged and build automatic, question-specific view plans."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl, stable_key
from mpp_video_qa.region_views import (REGION_VIEW_PROTOCOL, full_replay, json_atomic,
                                      plan_views, proposal_file, validate_plan)

SPLITS = ("train", "dev", "dev_action", "dev_scenery", "eval")


def prepare(source_dir, proposal_dir, output_dir, max_regions=2, min_score=.35,
            padding=.2, replay_fraction=.2, seed=44):
    source_dir, proposal_dir, output_dir = map(Path, (source_dir, proposal_dir, output_dir))
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("view plans need a new directory; preserve the source QA")
    detector_manifest = json.loads((proposal_dir / "manifest.json").read_text())
    settings = {"max_regions_per_frame": max_regions, "minimum_detection_score": min_score,
                "context_padding": padding, "training_full_replay_fraction": replay_fraction,
                "seed": seed}
    identity = {"protocol": REGION_VIEW_PROTOCOL, "settings": settings,
                "detector_identity": detector_manifest["detector_identity"],
                "source_sha256": {s: file_sha256(source_dir / (s + ".jsonl")) for s in SPLITS}}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if any(old[k] != v for k, v in identity.items()):
            raise ValueError("view directory belongs to different source/planner settings; use a new directory")
    original = {s: load_jsonl(source_dir / (s + ".jsonl")) for s in SPLITS}
    segments = {s: {r["segment_id"] for r in original[s]} for s in ("train", "dev", "eval")}
    for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        if segments[left] & segments[right]:
            raise ValueError("train/dev/eval share segment identities")
    dev_by_id = {r["question_id"]: r for r in original["dev"]}
    components = original["dev_action"] + original["dev_scenery"]
    if len(components) != len(dev_by_id) or {r["question_id"]: r for r in components} != dev_by_id:
        raise ValueError("dev_action and dev_scenery must partition combined dev exactly")
    proposals, regions, split_stats = {}, {}, {}
    for split, rows in original.items():
        if len({r["question_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate question IDs in %s" % split)
        plans = []
        counts = Counter({key: 0 for key in ("records", "records_with_roi", "full_replay_records", "region_slots", "view_slots")})
        for index, row in enumerate(rows, 1):
            segment = row["segment_id"]
            if segment not in proposals:
                path = proposal_file(proposal_dir, segment)
                proposal = json.loads(path.read_text())
                if proposal["detector_identity"] != identity["detector_identity"]:
                    raise ValueError("mixed detector identities: %s" % path)
                proposals[segment] = proposal
            plan = plan_views(row["question_id"], segment, row["question"], row["frames"], proposals[segment],
                              max_regions, min_score, padding,
                              replay=split == "train" and full_replay(row["question_id"], replay_fraction, seed))
            validate_plan(plan, row)
            plans.append(plan)
            region_count = sum(v["kind"] == "region" for v in plan["views"])
            counts["records"] += 1
            counts["records_with_roi"] += int(region_count > 0)
            counts["full_replay_records"] += int(plan["full_replay"])
            counts["region_slots"] += region_count
            counts["view_slots"] += len(plan["views"])
            for view in plan["views"]:
                if view["kind"] != "region":
                    continue
                entry = {key: view[key] for key in ("region_key", "pixel_box", "image_sha256", "box")}
                entry["frame_path"] = row["frames"][view["frame_index"]]
                if view["region_key"] in regions and regions[view["region_key"]] != entry:
                    # Same pixel content can legitimately have a second absolute path.
                    previous = regions[view["region_key"]]
                    if any(previous[k] != entry[k] for k in ("pixel_box", "image_sha256", "box")):
                        raise ValueError("region key collision")
                else:
                    regions[view["region_key"]] = entry
            if index % 1000 == 0:
                print(json.dumps({"split": split, "planned": index, "total": len(rows)}), flush=True)
        dump_jsonl(output_dir / (split + ".jsonl"), rows)
        dump_jsonl(output_dir / (split + ".views.jsonl"), plans)
        split_stats[split] = dict(counts)
        split_stats[split]["mean_views"] = counts["view_slots"] / max(1, len(rows))
    catalog = sorted(regions.values(), key=lambda r: r["region_key"])
    dump_jsonl(output_dir / "regions.jsonl", catalog)
    identity.update({"splits": split_stats, "unique_regions": len(catalog),
                     "region_catalog_sha256": file_sha256(output_dir / "regions.jsonl"),
                     "views_sha256": {s: file_sha256(output_dir / (s + ".views.jsonl")) for s in SPLITS},
                     "qa_values_unchanged": all(original[s] == load_jsonl(output_dir / (s + ".jsonl")) for s in SPLITS),
                     "selection_inputs": "question, original frame pixels, detector proposals; never answers",
                     "split_segment_overlap": 0,
                     "estimated_region_feature_gib": len(catalog) * 257 * 1408 * 2 / 2**30})
    json_atomic(manifest_path, identity)
    print(json.dumps(identity, indent=2), flush=True)
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--proposal-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-regions-per-frame", type=int, default=2)
    p.add_argument("--min-score", type=float, default=.35)
    p.add_argument("--context-padding", type=float, default=.2)
    p.add_argument("--full-replay-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=44)
    a = p.parse_args()
    prepare(a.source_dir, a.proposal_dir, a.output_dir, a.max_regions_per_frame,
            a.min_score, a.context_padding, a.full_replay_fraction, a.seed)


if __name__ == "__main__":
    main()
