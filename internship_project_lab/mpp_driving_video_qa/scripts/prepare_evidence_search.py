#!/usr/bin/env python3
"""Export dev/eval question+image-only inputs, then assemble frozen MPP view plans."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.evidence_search import (INPUT_PROTOCOL, SEARCH_PROTOCOL, content_hash,
    evidence_plan, result_file, validate_input)
from mpp_video_qa.region_views import EVIDENCE_VIEW_PROTOCOL, json_atomic, validate_plan


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("existing identity differs; use a new versioned directory: %s" % path)
    if not path.exists():
        json_atomic(path, value)


def immutable_rows(path, rows):
    path = Path(path)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("existing data differ; use a new versioned directory: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def export_inputs(source_dir, output_dir):
    from PIL import Image
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError("preserve the source dataset in its own directory")
    original = {s: load_jsonl(source / (s + ".jsonl")) for s in ("train", "dev", "eval")}
    segments = {s: {r["segment_id"] for r in rows} for s, rows in original.items()}
    for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        if segments[left] & segments[right]:
            raise ValueError("train/dev/eval segments overlap")
    for split, rows in original.items():
        if len({r["question_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate question IDs in %s" % split)
    if {r["question_id"] for r in original["dev"]} & {r["question_id"] for r in original["eval"]}:
        raise ValueError("dev/eval question IDs overlap")
    identity = {"protocol": INPUT_PROTOCOL, "source_dir": str(source),
                "source_sha256": {s: file_sha256(source / (s + ".jsonl")) for s in original},
                "splits": {}, "answers_in_search_inputs": False,
                "selection_inputs": "complete original question + original frame pixels only",
                "original_qa_preserved": True, "split_segment_overlap": 0}
    existing = output / "input_manifest.json"
    if existing.is_file():
        old = json.loads(existing.read_text())
        if old["source_sha256"] != identity["source_sha256"] or old["source_dir"] != str(source):
            raise ValueError("source QA changed; use a new export directory")
    started, frames = time.monotonic(), {}
    for split in ("dev", "eval"):
        rows, exported = original[split], []
        for index, row in enumerate(rows):
            if len(row["frames"]) != 5:
                raise ValueError("original five frames must be retained")
            metadata = []
            for frame_index, raw_path in enumerate(row["frames"]):
                path = Path(raw_path)
                if not path.is_absolute():
                    raise ValueError("export requires absolute original image paths")
                if raw_path not in frames:
                    with Image.open(path) as image:
                        frames[raw_path] = {"width": image.width, "height": image.height,
                                            "image_sha256": file_sha256(path)}
                metadata.append(dict(frames[raw_path], frame_index=frame_index))
            clean = {"protocol": INPUT_PROTOCOL, "question_id": row["question_id"],
                     "segment_id": row["segment_id"], "question": row["question"],
                     "frames": list(row["frames"]), "frame_metadata": metadata,
                     "source_subset": row.get("source_subset")}
            exported.append(validate_input(clean))
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(rows):
                progress = {"status": "exporting", "split": split, "completed": index + 1,
                            "total": len(rows), "unique_images_checked": len(frames),
                            "elapsed_seconds": time.monotonic() - started,
                            "updated_unix_seconds": time.time()}
                json_atomic(output / "export_progress.json", progress)
                print(json.dumps(progress), flush=True)
        immutable_rows(output / "inputs" / (split + ".jsonl"), exported)
        # Labels live separately for downstream MPP scoring, never in search inputs.
        immutable_rows(output / "original" / (split + ".jsonl"), rows)
        identity["splits"][split] = {
            "records": len(rows), "segments": len(segments[split]),
            "by_source": dict(Counter(r.get("source_subset") for r in rows)),
            "input_sha256": file_sha256(output / "inputs" / (split + ".jsonl")),
            "qa_sha256": file_sha256(output / "original" / (split + ".jsonl"))}
    immutable_json(existing, identity)
    json_atomic(output / "export_progress.json", {"status": "completed", "unique_images_checked": len(frames),
                                                 "updated_unix_seconds": time.time()})
    return identity


def assemble_views(data_dir, search_dir, split):
    data, search = Path(data_dir), Path(search_dir)
    front_manifest = json.loads((data / "input_manifest.json").read_text())
    run = json.loads((search / "run_manifest.json").read_text())
    if run["protocol"] != SEARCH_PROTOCOL or run["input_sha256"] != front_manifest["splits"][split]["input_sha256"]:
        raise ValueError("search did not use this split's answer-free input")
    if run["max_records"] != 0:
        raise ValueError("a bounded interface check is not a complete dev/eval run")
    inputs_path, original_path = data / "inputs" / (split + ".jsonl"), data / "original" / (split + ".jsonl")
    if (file_sha256(inputs_path) != run["input_sha256"] or
            file_sha256(original_path) != front_manifest["splits"][split]["qa_sha256"]):
        raise ValueError("exported question/image input or original QA was modified")
    inputs, originals = load_jsonl(inputs_path), load_jsonl(original_path)
    if len(inputs) != len(originals):
        raise ValueError("input and QA counts differ")
    plans, regions, counts = [], {}, Counter()
    for clean, original in zip(inputs, originals):
        validate_input(clean)
        if any(clean[k] != original[k] for k in ("question_id", "segment_id", "question", "frames")):
            raise ValueError("original question/frame identity changed")
        path = result_file(search, clean["question_id"])
        if not path.is_file():
            raise ValueError("evidence search is incomplete; missing %s" % clean["question_id"])
        result = json.loads(path.read_text())
        plan = evidence_plan(clean, result, run["policy_sha256"], run["policy_identity"]["config"])
        validate_plan(plan, original)
        plans.append(plan)
        counts[result["status"]] += 1
        for view in plan["views"]:
            if view["kind"] != "region":
                continue
            entry = {k: view[k] for k in ("region_key", "pixel_box", "box", "image_sha256")}
            entry["frame_path"] = original["frames"][view["frame_index"]]
            key = entry["region_key"]
            if key in regions and any(regions[key][k] != entry[k] for k in ("pixel_box", "box", "image_sha256")):
                raise ValueError("native crop identity collision")
            regions.setdefault(key, entry)
    folder = data / split
    immutable_rows(folder / "data.jsonl", originals)
    immutable_rows(folder / "views.jsonl", plans)
    immutable_rows(folder / "regions.jsonl", sorted(regions.values(), key=lambda r: r["region_key"]))
    identity = {"protocol": EVIDENCE_VIEW_PROTOCOL, "split": split, "records": len(plans),
                "policy_sha256": run["policy_sha256"], "search_manifest_sha256": file_sha256(search / "run_manifest.json"),
                "qa_values_unchanged": originals == load_jsonl(folder / "data.jsonl"),
                "qa_sha256": file_sha256(folder / "data.jsonl"),
                "views_sha256": file_sha256(folder / "views.jsonl"),
                "catalog_sha256": file_sha256(folder / "regions.jsonl"),
                "search_states": dict(counts), "unique_regions": len(regions),
                "region_slots": sum(v["kind"] == "region" for p in plans for v in p["views"]),
                "max_views": max(len(p["views"]) for p in plans),
                "estimated_region_feature_gib": len(regions) * 257 * 1408 * 2 / 2**30,
                "external_text_passed_to_mpp": False, "failed_search_fallback": "original five frames"}
    immutable_json(folder / "manifest.json", identity)
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source-dir", required=True)
    export.add_argument("--output-dir", required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--data-dir", required=True)
    assemble.add_argument("--search-dir", required=True)
    assemble.add_argument("--split", choices=("dev", "eval"), required=True)
    a = p.parse_args()
    result = (export_inputs(a.source_dir, a.output_dir) if a.action == "export" else
              assemble_views(a.data_dir, a.search_dir, a.split))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
