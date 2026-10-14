#!/usr/bin/env python3
"""Select reference-consistent factual contrasts; never label them visually audited."""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import io
import itertools
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl, normalize_answer, stable_key
from mpp_video_qa.preference import PREFERENCE_PROTOCOL, row_identity
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.generate_answers import write_json_atomic


def fact_values(text, question):
    text, question = text.lower(), question.lower()
    result = {}
    # These deliberately narrow patterns describe textual contrasts, not image truth.
    directions = set(re.findall(r"\b(?:turn(?:ing|ed)?|steer(?:ing|ed)?)\s+(?:to\s+)?(?:the\s+)?(left|right)\b", text))
    directions.update(re.findall(r"\b(left|right)[ -]turn\b", text))
    if directions:
        result["turn_direction"] = directions
    if re.search(r"\b(traffic light|traffic signal|signal light|light[s]? (?:red|green|amber))", question):
        values = set(re.findall(r"\b(red|green|amber|yellow)\b", text))
        if values:
            result["signal_color"] = {"amber" if x == "yellow" else x for x in values}
    if re.search(r"\b(?:what|which)\b.*\b(?:vehicle|car|following)\b", question):
        values = set(re.findall(r"\b(car|truck|lorry|bus|van|motorcycle|motorbike|bicycle)\b", text))
        aliases = {"lorry": "truck", "motorbike": "motorcycle"}
        if values:
            result["vehicle_type"] = {aliases.get(x, x) for x in values}
    # Bind motion to the ego subject; do not infer ego motion from other vehicles.
    motion = re.findall(r"\b(?:i am|i'm|we are|we're|the ego vehicle is)\s+(?:currently\s+)?"
                        r"(stopped|stationary|stopping|decelerating|slowing down|accelerating|speeding up|starting|moving off)\b", text)
    aliases = {"stationary": "stopped", "slowing down": "decelerating",
               "speeding up": "accelerating", "moving off": "starting"}
    if motion:
        result["ego_motion"] = {aliases.get(x, x) for x in motion}
    # Negated or hypothetical assertions are left for explicit review.
    if re.search(r"\b(?:not|might|maybe|perhaps|could|would|if)\b|n't\b", text):
        return {}
    return result


def factual_contrast(chosen, rejected, references, question):
    c, r = fact_values(chosen, question), fact_values(rejected, question)
    refs = [fact_values(ref, question) for ref in references]
    evidence = {}
    for key in c.keys() & r.keys():
        if len(c[key]) == len(r[key]) == 1 and c[key] != r[key] and all(x.get(key) == c[key] for x in refs):
            evidence[key] = {"reference_and_chosen": next(iter(c[key])), "rejected": next(iter(r[key]))}
    return evidence


def balanced_pick(rows, maximum, source_counts, seed):
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in rows:
        pools[row["source_subset"]].append(row)
    total = sum(source_counts.values())
    selected = []
    quotas = {k: int(maximum * v / total) for k, v in source_counts.items()}
    remainder = maximum - sum(quotas.values())
    for key in sorted(quotas, key=lambda k: maximum * source_counts[k] / total - quotas[k], reverse=True)[:remainder]:
        quotas[key] += 1
    for key in sorted(pools):
        rng.shuffle(pools[key])
        selected.extend(pools[key][:quotas.get(key, 0)])
    # Do not silently fill a missing source quota with the other source.
    return selected


def select_pairs(originals, predictions, held_out, margin=1.0, max_pairs=1000,
                 replay_fraction=0.25, seed=42, review=None):
    if margin < 0 or not 0 <= replay_fraction < 1:
        raise ValueError("invalid confidence margin or replay fraction")
    original = {r["question_id"]: r for r in originals}
    if len(original) != len(originals) or not original:
        raise ValueError("duplicate or empty original questions")
    if {r["segment_id"] for r in originals} & {r["segment_id"] for r in held_out}:
        raise ValueError("training/held-out segment overlap")
    if {str(Path(p).resolve()) for r in originals for p in r["frames"]} & {str(Path(p).resolve()) for r in held_out for p in r["frames"]}:
        raise ValueError("training/held-out frame overlap")
    groups = defaultdict(dict)
    for pred in predictions:
        qid = pred["question_id"]
        if qid not in original:
            raise ValueError("prediction outside training probe")
        row = original[qid]
        if (pred["segment_id"] != row["segment_id"] or pred["question"] != row["question"] or
                pred["references"] != row["answers"] or pred.get("teacher_context") != "view" or
                pred.get("view") != "full"):
            raise ValueError("student probe identity or no-hint context differs")
        if pred["sample_id"] in groups[qid]:
            raise ValueError("duplicate sample")
        groups[qid][pred["sample_id"]] = pred
    if set(groups) != set(original) or any(set(g) != {0, 1, 2, 3} for g in groups.values()):
        raise ValueError("expected complete four-draw student groups")
    review = review or {}
    pairs, replay, candidates = [], [], []
    counts = Counter()
    for qid, draws in sorted(groups.items()):
        row = original[qid]
        if len(row["frames"]) != 5 or not row["answers"]:
            raise ValueError("expected five frames and original references")
        good = [p for p in draws.values() if p.get("finished") and p["lingo_judge_correct"] and
                p["lingo_judge_logit"] >= margin and str(p["prediction"]).strip() and "<|" not in p["prediction"]]
        bad = [p for p in draws.values() if p.get("finished") and not p["lingo_judge_correct"] and
               p["lingo_judge_logit"] <= -margin and str(p["prediction"]).strip() and "<|" not in p["prediction"]]
        correct = sum(bool(p["lingo_judge_correct"]) for p in draws.values())
        counts["correct_%d_of_4" % correct] += 1
        base = dict(row, preference_protocol=PREFERENCE_PROTOCOL, training_source="train")
        if len(good) == 4:
            entry = dict(base, kind="replay", chosen=row["answers"][0], rejected=None,
                         example_id="replay:" + qid, label_source="original_reference_student_4of4")
            entry["row_sha256"] = row_identity(entry)
            replay.append(entry)
        options = []
        for c, r in itertools.product(good, bad):
            if normalize_answer(c["prediction"]) == normalize_answer(r["prediction"]):
                continue
            lengths = [max(1, len(p["prediction"].split())) for p in (c, r)]
            if max(lengths) / min(lengths) > 2.5:
                continue
            evidence = factual_contrast(c["prediction"], r["prediction"], row["answers"], row["question"])
            entry = dict(base, kind="preference", chosen=c["prediction"], rejected=r["prediction"],
                         chosen_sample_id=c["sample_id"], rejected_sample_id=r["sample_id"],
                         chosen_judge_logit=c["lingo_judge_logit"], rejected_judge_logit=r["lingo_judge_logit"],
                         fact_contrast=evidence, visual_review_status="not_reviewed",
                         label_source="reference_fact_rules_and_judge" if evidence else "judge_only_candidate")
            entry["example_id"] = stable_key(qid, entry["chosen"], entry["rejected"])
            decision = review.get(entry["example_id"])
            if decision is not None:
                entry["review"] = decision
                if decision.get("visual_verified"):
                    entry["visual_review_status"] = "review_file_verified"
                if decision["approved"]:
                    entry["label_source"] = "explicit_review"
            entry["row_sha256"] = row_identity(entry)
            entry["eligible"] = bool(decision["approved"] if decision is not None else evidence)
            options.append(entry)
        if options:
            options.sort(key=lambda x: (x["eligible"], x["chosen_judge_logit"] - x["rejected_judge_logit"]), reverse=True)
            candidates.extend(options)
            eligible = [x for x in options if x["eligible"]]
            if eligible:
                pairs.append(eligible[0])
    source_counts = Counter(r["source_subset"] for r in originals)
    selected_pairs = balanced_pick(pairs, max_pairs, source_counts, seed)
    replay_limit = round(len(selected_pairs) * replay_fraction / (1 - replay_fraction))
    pair_counts = Counter(r["source_subset"] for r in selected_pairs)
    overall_target = len(selected_pairs) + replay_limit
    deficits = {src: max(0., overall_target * count / sum(source_counts.values()) - pair_counts[src])
                for src, count in source_counts.items()}
    selected_replay = balanced_pick(replay, replay_limit, deficits, seed + 1) if replay_limit else []
    rows = selected_pairs + selected_replay
    random.Random(seed).shuffle(rows)
    return rows, candidates, {
        "probe_questions": len(original), "sampling_counts": dict(counts),
        "eligible_pairs_before_cap": len(pairs), "eligible_replay": len(replay),
        "pairs": len(selected_pairs), "replay": len(selected_replay), "records": len(rows),
        "source_counts": dict(Counter(r["source_subset"] for r in rows)),
        "pair_source_counts": dict(Counter(r["source_subset"] for r in selected_pairs)),
        "pair_types": dict(Counter(k for r in selected_pairs for k in r["fact_contrast"])),
        "visually_reviewed_pairs": sum(r["visual_review_status"] != "not_reviewed" for r in selected_pairs),
        "note": "Automatic rules check textual facts against references, not visual ground truth. Judge-only candidates are excluded unless explicitly approved.",
    }


def review_html(rows, destination):
    from PIL import Image
    parts = ['<!doctype html><meta charset="utf-8"><title>Preference review</title>',
             '<style>body{font:16px sans-serif;max-width:1200px;margin:auto}article{border-top:1px solid #888;padding:20px 0}img{max-width:19%;vertical-align:top}.chosen{color:#076}.rejected{color:#a22}code{word-break:break-all}</style>',
             '<h1>偏好训练对检查</h1><p>自动筛选仅核对文本事实与参考答案；未声明已经人工看图验证。</p>']
    for row in rows:
        if row["kind"] != "preference":
            continue
        parts.append('<article><code>%s</code><h3>%s</h3>' % (row["example_id"], html.escape(row["question"])))
        for path in row["frames"]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((240, 160))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=75)
            parts.append('<img src="data:image/jpeg;base64,%s">' % base64.b64encode(buffer.getvalue()).decode("ascii"))
        for label, text in (("reference", " / ".join(row["answers"])), ("chosen", row["chosen"]), ("rejected", row["rejected"])):
            parts.append('<p class="%s"><b>%s:</b> %s</p>' % (label, label, html.escape(text)))
        parts.append('<p>%s</p></article>' % html.escape(json.dumps(row["fact_contrast"], ensure_ascii=False)))
    destination.write_text("\n".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--student-scored", required=True)
    p.add_argument("--source-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-pairs", type=int, default=1000)
    p.add_argument("--judge-margin", type=float, default=1.0)
    p.add_argument("--replay-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review-jsonl", default="")
    args = p.parse_args()
    if (args.output_dir / "manifest.json").exists():
        previous = json.loads((args.output_dir / "manifest.json").read_text())
        expected = {"source_train_sha256": file_sha256(args.source_dir / "train.jsonl"),
                    "probe_sha256": file_sha256(args.probe), "student_scored_sha256": file_sha256(args.student_scored),
                    "judge_margin": args.judge_margin, "replay_fraction": args.replay_fraction,
                    "max_pairs": args.max_pairs, "seed": args.seed,
                    "review_sha256": file_sha256(args.review_jsonl) if args.review_jsonl else None,
                    "held_out_sha256": {k: file_sha256(args.source_dir / (k + ".jsonl")) for k in ("dev", "eval")}}
        if (any(previous.get(k) != v for k, v in expected.items()) or
                previous["data_sha256"] != file_sha256(args.output_dir / "train.jsonl")):
            raise ValueError("prepared dataset/settings differ; choose a new output directory")
        print(json.dumps(dict(previous, existing_dataset_validated=True), indent=2))
        return
    original = load_jsonl(args.probe)
    wanted = {r["question_id"] for r in original}
    train = {}
    with (args.source_dir / "train.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["question_id"] in wanted:
                    if row["question_id"] in train:
                        raise ValueError("duplicate selected question in full training source")
                    train[row["question_id"]] = row
    for row in original:
        if row["question_id"] not in train or any(row[k] != train[row["question_id"]][k] for k in ("segment_id", "frames", "question", "answers", "source_subset")):
            raise ValueError("probe is not unchanged original training QA")
    scored = Path(args.student_scored)
    score_summary = json.loads((scored.parent / "summary.json").read_text())
    generation = json.loads((scored.parent.parent / "run_manifest.json").read_text())
    if (score_summary.get("judge_protocol") != JUDGE_PROTOCOL or generation.get("data_sha256") != file_sha256(args.probe) or
            generation.get("task_instruction_version") != "legacy_v1" or generation.get("feature_stage") != "qformer" or
            generation.get("teacher_context") != "view" or generation.get("num_samples") != 4):
        raise ValueError("expected completed official four-draw, no-hint joint student probe")
    review_rows = load_jsonl(args.review_jsonl) if args.review_jsonl else []
    review = {r["example_id"]: r for r in review_rows}
    if len(review) != len(review_rows) or any(type(r.get("approved")) is not bool for r in review_rows):
        raise ValueError("review entries require unique example_id and boolean approved")
    held_out = load_jsonl(args.source_dir / "dev.jsonl") + load_jsonl(args.source_dir / "eval.jsonl")
    rows, candidates, stats = select_pairs(original, load_jsonl(scored), held_out, args.judge_margin,
                                         args.max_pairs, args.replay_fraction, args.seed, review)
    if not set(review).issubset({r["example_id"] for r in candidates}):
        raise ValueError("review refers to unknown candidate pairs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "train.jsonl", rows)
    dump_jsonl(args.output_dir / "review_candidates.jsonl", candidates)
    for split in ("dev_action", "dev_scenery", "eval"):
        dump_jsonl(args.output_dir / (split + ".jsonl"), load_jsonl(args.source_dir / (split + ".jsonl")))
    review_html(rows, args.output_dir / "review_selected.html")
    manifest = dict(stats, protocol=PREFERENCE_PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
                    data_sha256=file_sha256(args.output_dir / "train.jsonl"),
                    source_train_sha256=file_sha256(args.source_dir / "train.jsonl"),
                    probe_sha256=file_sha256(args.probe), student_scored_sha256=file_sha256(scored),
                    generation_identity=generation, task_instruction_version="legacy_v1",
                    judge_protocol=JUDGE_PROTOCOL, judge_margin=args.judge_margin,
                    replay_fraction=args.replay_fraction, seed=args.seed, max_pairs=args.max_pairs,
                    review_sha256=file_sha256(args.review_jsonl) if args.review_jsonl else None,
                    held_out_sha256={k: file_sha256(args.source_dir / (k + ".jsonl")) for k in ("dev", "eval")})
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
