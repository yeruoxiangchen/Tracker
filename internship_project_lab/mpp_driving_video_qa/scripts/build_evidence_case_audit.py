#!/usr/bin/env python3
"""Deterministic, outcome-stratified dev case audit; never training examples."""
import hashlib
import json
from pathlib import Path
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl
from mpp_video_qa.evidence_search import result_file

RUN = ROOT / "outputs/vlm_evidence_search_v1"
DATA = ROOT / "datasets/processed/lingoqa_vlm_evidence_v1/dev"
OUT = ROOT / "outputs/evidence_case_audit_20261009"
SEED = "20261009-evidence-review"


def put_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("audit manifest already exists with different contents: " + str(path))
    path.write_text(text)


def make_sheet(case):
    row, plan, result = case["row"], case["plan"], case["search"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    lines = []
    for text in ["%02d %s / %s | %s" % (case["case"], case["group"], row["source_subset"], row["question_id"]),
                 "Q: " + row["question"], "Reference: " + " | ".join(row["answers"]),
                 "FULL: " + case["full"]["prediction"], "EVIDENCE: " + case["evidence"]["prediction"]]:
        lines.extend(textwrap.wrap(text, 165))
    header = len(lines) * 25 + 20
    crop_views = [v for v in plan["views"] if v["kind"] == "region"]
    crop_rows = (len(crop_views) + 3) // 4
    sheet = Image.new("RGB", (1920, header + 2 * 356 + crop_rows * 332), "white")
    draw = ImageDraw.Draw(sheet)
    for i, line in enumerate(lines):
        draw.text((12, 10 + 25 * i), line, font=font, fill="black")
    originals = []
    for i, path in enumerate(row["frames"]):
        with Image.open(path) as image:
            original = image.convert("RGB")
        originals.append(original)
        overlay = original.copy()
        marks = ImageDraw.Draw(overlay)
        for v in crop_views:
            if v["frame_index"] == i:
                marks.rectangle(v["pixel_box"], outline="#00ff60", width=8)
        overlay = ImageOps.contain(overlay, (628, 318))
        x, y = (i % 3) * 640 + 6, header + (i // 3) * 356
        draw.text((x, y), "Original T%d (green = selected crops)" % i, font=small, fill="black")
        sheet.paste(overlay, (x, y + 26))
    for i, v in enumerate(crop_views):
        x, y = (i % 4) * 480 + 6, header + 712 + (i // 4) * 332
        crop = originals[v["frame_index"]].crop(v["pixel_box"])
        draw.text((x, y), "%s T%d %s" % (v["candidate_id"], v["frame_index"], v["pixel_box"]), font=small, fill="black")
        sheet.paste(ImageOps.contain(crop, (466, 292)), (x, y + 27))
        crop.close()
    path = OUT / "figures" / ("case_%02d.jpg" % case["case"])
    sheet.save(path, quality=92)
    for original in originals:
        original.close()
    sheet.close()


def main():
    qa = {r["question_id"]: r for r in load_jsonl(DATA / "data.jsonl")}
    plans = {r["question_id"]: r for r in load_jsonl(DATA / "views.jsonl")}
    full = {r["question_id"]: r for r in load_jsonl(RUN / "mpp_full_dev/scored/scored_predictions.jsonl")}
    evidence = {r["question_id"]: r for r in load_jsonl(RUN / "mpp_evidence_dev/scored/scored_predictions.jsonl")}
    quotas = [("regressed", "action", 12), ("regressed", "scenery", 4),
              ("fixed", "action", 4), ("fixed", "scenery", 2),
              ("both_correct", "action", 1), ("both_correct", "scenery", 1)]
    cases, used_segments = [], set()
    for group, source, number in quotas:
        candidates = []
        for qid, row in qa.items():
            if row["source_subset"] != source or plans[qid]["evidence_search"]["status"] != "selected":
                continue
            a, b = full[qid]["lingo_judge_correct"], evidence[qid]["lingo_judge_correct"]
            matches = {"regressed": a and not b, "fixed": not a and b, "both_correct": a and b}
            if matches[group]:
                candidates.append(qid)
        candidates.sort(key=lambda q: hashlib.sha256((SEED + q).encode()).hexdigest())
        added = 0
        for qid in candidates:
            if qa[qid]["segment_id"] in used_segments:
                continue
            used_segments.add(qa[qid]["segment_id"])
            result = json.loads(result_file(RUN / "search_dev", qid).read_text())
            cases.append({"case": len(cases) + 1, "group": group, "row": qa[qid], "plan": plans[qid],
                          "full": full[qid], "evidence": evidence[qid], "search": result})
            added += 1
            if added == number:
                break
        if added != number:
            raise ValueError("not enough distinct segments")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    put_json(OUT / "cases.json", cases)
    put_json(OUT / "selection.json", {"seed": SEED, "quota": quotas, "records": len(cases),
             "distinct_segments": len(used_segments), "selection_uses_outcomes": True,
             "purpose": "visual failure diagnosis only; not an unbiased evaluation or training dataset",
             "question_ids": [c["row"]["question_id"] for c in cases]})
    for filename, key in [("data.jsonl", "row"), ("views.jsonl", "plan")]:
        path = OUT / filename
        text = "".join(json.dumps(c[key], ensure_ascii=False) + "\n" for c in cases)
        if path.exists() and path.read_text() != text:
            raise ValueError("audit input changed")
        path.write_text(text)
    lines = ["# Evidence case audit — 2026-10-09", "",
             "Outcome-stratified dev diagnosis: 16 regressed, 6 fixed, 2 both-correct; 24 distinct segments.",
             "Sampling was fixed before visual review. This is not population accuracy or training data.", ""]
    for case in cases:
        make_sheet(case)
        row = case["row"]
        lines.extend(["## Case %02d — %s / %s" % (case["case"], case["group"], row["source_subset"]), "",
            "Q: " + row["question"], "", "Reference: " + " | ".join(row["answers"]), "",
            "Full: " + case["full"]["prediction"], "", "Evidence: " + case["evidence"]["prediction"], "",
            "Search reason: " + case["search"]["reason"], "",
            "![Case %02d](figures/case_%02d.jpg)" % (case["case"], case["case"]), ""])
    (OUT / "index.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": len(cases), "index": str(OUT / "index.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
