#!/usr/bin/env python3
"""Summarize a selected diagnostic sample, not a new accuracy benchmark."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/evidence_case_audit_20261009"
SOURCE = ROOT / "outputs/vlm_evidence_search_v1"


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {row["question_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate question IDs: %s" % path)
    return indexed


def correct(row):
    return bool(row["lingo_judge_correct"])


def paired(ids, reference, candidate):
    counts = Counter()
    for qid in ids:
        a, b = correct(reference[qid]), correct(candidate[qid])
        counts["fixed" if not a and b else "regressed" if a and not b else
               "both_correct" if a else "both_wrong"] += 1
    return {"records": len(ids),
            "reference_correct": sum(correct(reference[qid]) for qid in ids),
            "candidate_correct": sum(correct(candidate[qid]) for qid in ids),
            "fixed": counts["fixed"], "regressed": counts["regressed"],
            "both_correct": counts["both_correct"], "both_wrong": counts["both_wrong"],
            "net": counts["fixed"] - counts["regressed"]}


def main():
    cases = json.loads((AUDIT / "cases.json").read_text())
    ids = [case["row"]["question_id"] for case in cases]
    runs = {key: read_rows(AUDIT / key / "scored/scored_predictions.jsonl")
            for key in ("full", "duplicate", "roi")}
    for key, rows in runs.items():
        if set(rows) != set(ids):
            raise ValueError("incomplete diagnostic arm: %s" % key)
        for case in cases:
            qid = case["row"]["question_id"]
            if (rows[qid]["question"] != case["row"]["question"] or
                    rows[qid]["references"] != case["row"]["answers"] or
                    rows[qid]["region_observation"]["plan_sha256"] != case["plan"]["plan_sha256"]):
                raise ValueError("diagnostic input mismatch: %s/%s" % (key, qid))
    manifests = {key: json.loads((AUDIT / key / "run_manifest.json").read_text()) for key in runs}
    for field in ("adapter_identity", "projector_identity", "qformer_adapter_sha256",
                  "bridge_init_sha256", "data_sha256", "feature_stage", "precision_protocol",
                  "dtype", "batch_size", "max_new_tokens", "task_instruction"):
        if len({json.dumps(manifests[key][field], sort_keys=True) for key in runs}) != 1:
            raise ValueError("unmatched inference field: %s" % field)

    groups = defaultdict(list)
    for case in cases:
        groups[case["group"]].append(case["row"]["question_id"])
    result = {
        "records": len(ids),
        "selection": json.loads((AUDIT / "selection.json").read_text()),
        "warning": "Outcome-stratified dev sample; counts are diagnostic, not population accuracy.",
        "correct_counts": {key: sum(correct(row) for row in rows.values()) for key, rows in runs.items()},
        "paired": {"full_to_duplicate": paired(ids, runs["full"], runs["duplicate"]),
                   "duplicate_to_roi": paired(ids, runs["duplicate"], runs["roi"]),
                   "full_to_roi": paired(ids, runs["full"], runs["roi"])},
        "by_original_outcome": {group: {key: sum(correct(runs[key][qid]) for qid in qids)
                                        for key in runs} for group, qids in groups.items()},
        "replay": {},
    }
    for current, old in (("full", "full"), ("roi", "evidence")):
        result["replay"][current] = {
            "identical_text": sum(runs[current][c["row"]["question_id"]]["prediction"] ==
                                  c[old]["prediction"] for c in cases),
            "identical_judge": sum(correct(runs[current][c["row"]["question_id"]]) ==
                                   correct(c[old]) for c in cases),
            "changed_cases": [c["case"] for c in cases if
                              runs[current][c["row"]["question_id"]]["prediction"] != c[old]["prediction"]],
        }

    full = read_rows(SOURCE / "mpp_full_dev/scored/scored_predictions.jsonl")
    roi = read_rows(SOURCE / "mpp_evidence_dev/scored/scored_predictions.jsonl")
    plans = read_rows(ROOT / "datasets/processed/lingoqa_vlm_evidence_v1/dev/views.jsonl")
    if set(full) != set(roi) or set(full) != set(plans):
        raise ValueError("source development sets differ")
    strata, whole_slots = defaultdict(list), 0
    for qid, plan in plans.items():
        views = [v for v in plan["views"] if v["kind"] == "region"]
        if not views:
            continue
        whole = [v for v in views if v["pixel_box"] == [0, 0] + plan["frame_sizes"][v["frame_index"]]]
        whole_slots += len(whole)
        group = ("fullsize_only" if len(whole) == len(views) else
                 "mixed_fullsize_and_crop" if whole else "actual_crops_only")
        strata[group].append(qid)
    result["source_dev"] = {
        "records": len(full), "fullsize_selected_slots": whole_slots,
        "selected_view_strata": {key: paired(qids, full, roi) for key, qids in strata.items()},
        "posthoc_lexical_groups": {},
        "lexical_caution": "Question-text regex groups, not validated task categories or deployable routing rules.",
    }
    color_ids = [q for q, r in full.items() if re.search(r"\bcolou?r\b", r["question"].lower())]
    lexical = {"explicit_color": color_ids,
               "explicit_color_scenery": [q for q in color_ids if full[q]["source_subset"] == "scenery"],
               "explicit_distance": [q for q, r in full.items() if re.search(
                   r"\b(how far|how close|distance)\b", r["question"].lower())]}
    for key, qids in lexical.items():
        result["source_dev"]["posthoc_lexical_groups"][key] = paired(qids, full, roi)
    result["cases"] = []
    lines = ["# 24-case matched-input diagnostic — 2026-10-09", "", result["warning"], "",
             "F=original five frames; D=same additional slots filled with full-frame features; R=actual crops.",
             "D and R use identical prompts, coordinates, view order and token budgets.", "",
             "| Case | Original outcome | Full Judge | Duplicate Judge | ROI Judge |", "|---|---|---:|---:|---:|"]
    for case in cases:
        qid = case["row"]["question_id"]
        row = {"case": case["case"], "question_id": qid, "original_outcome": case["group"],
               "arms": {key: {field: runs[key][qid][field] for field in
                              ("prediction", "lingo_judge_logit", "lingo_judge_correct")}
                        for key in runs}}
        result["cases"].append(row)
        lines.append("| %02d | %s | %d | %d | %d |" % (
            case["case"], case["group"], *(int(correct(runs[key][qid])) for key in runs)))
    for case, row in zip(cases, result["cases"]):
        lines.extend(["", "## Case %02d" % case["case"], "", case["row"]["question"], "",
                      "Reference: " + " | ".join(case["row"]["answers"]), ""])
        for key, arm in row["arms"].items():
            lines.extend(["%s — Judge %s; logit %.4f: %s" % (
                key, arm["lingo_judge_correct"], arm["lingo_judge_logit"], arm["prediction"]), ""])
        lines.append("[Original frames and selected crops](figures/case_%02d.jpg)" % case["case"])
    (AUDIT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (AUDIT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "selection")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
