#!/usr/bin/env python3
"""Review held-out failures and run controlled observations with fixed SFT weights."""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256, build_user_prompt
from scripts.generate_answers import write_json_atomic, directory_fingerprint

MODELS = ("baseline", "sft_control_final", "feedback_opd_final")
CONDITIONS = ("full_replay", "last_repeat", "reverse", "paired_duplicate", "paired_roi")
PROTOCOL = "dev_failure_observation_validation_v1"


def question_bucket(question):
    """Question-only sampling strata; these are not diagnosed causes or true labels."""
    q = question.lower()
    if re.search(r"how many|number of|count the", q):
        return "counting"
    if re.search(r"colou?r|traffic (lights?|signals?)|what (kind|type|brand|make)", q):
        return "attributes"
    if re.search(r"current.*(action|doing)|currently (doing|taking|starting)|starting, stopping|are you (accelerating|decelerating|stopped)|current speed", q):
        return "current_action"
    if re.search(r"safe|should|can you|do you need|have to|adapt your|next action", q):
        return "decision"
    if re.search(r"why|reason|justification", q):
        return "reason"
    if re.search(r"where|which side|on (the|your) (left|right)|lane position|positioned", q):
        return "spatial"
    if re.search(r"attention|hazard|obstacle|traffic ahead|watch|impacting", q):
        return "attention"
    return "scene"


def select_stratified(pool, count, rng, used_segments):
    buckets = defaultdict(list)
    for row in pool:
        buckets[question_bucket(row["question"])].append(row)
    for values in buckets.values():
        values.sort(key=lambda r: r["question_id"])
        rng.shuffle(values)
    result = []
    keys = sorted(buckets)
    while len(result) < count:
        added = False
        for key in keys:
            while buckets[key] and buckets[key][-1]["segment_id"] in used_segments:
                buckets[key].pop()
            if buckets[key] and len(result) < count:
                row = buckets[key].pop()
                result.append(row)
                used_segments.add(row["segment_id"])
                added = True
        if not added:
            raise ValueError("not enough distinct segments for requested diagnostic sample")
    return result


def prepare(args):
    dev_report = json.loads((args.reference_dir / "report.json").read_text())
    if dev_report["selected_checkpoints"] != {"sft_control": "sft_control_final", "feedback_opd": "feedback_opd_final"}:
        raise ValueError("this diagnostic fixes the development-selected final models")
    pools = {source: {"failure": [], "control": []} for source in ("action", "scenery")}
    hashes, all_rows = {}, []
    for source in pools:
        path = args.data_dir / ("dev_%s.jsonl" % source)
        original = {r["question_id"]: r for r in load_jsonl(path)}
        hashes[str(path.resolve())] = file_sha256(path)
        predictions = {}
        for model in MODELS:
            path = args.reference_dir / (model + "_" + source) / "scored/scored_predictions.jsonl"
            records = load_jsonl(path)
            predictions[model] = {r["question_id"]: r for r in records}
            assert len(records) == len(original) == len(predictions[model])
            assert predictions[model].keys() == original.keys()
            hashes[str(path.resolve())] = file_sha256(path)
        for qid, row in original.items():
            for m in MODELS:
                p = predictions[m][qid]
                assert (p["question"], p["segment_id"], p["references"]) == (row["question"], row["segment_id"], row["answers"])
            item = dict(row, model_predictions={m: predictions[m][qid] for m in MODELS}, question_bucket=question_bucket(row["question"]))
            all_rows.append(item)
            correct = [predictions[m][qid]["lingo_judge_correct"] for m in MODELS]
            if not any(correct):
                pools[source]["failure"].append(item)
            elif all(correct):
                pools[source]["control"].append(item)
    rng, used = random.Random(args.seed), set()
    selected = []
    for group, requests in (("failure", {"action": 60, "scenery": 20}), ("control", {"action": 22, "scenery": 8})):
        for source, count in requests.items():
            chosen = select_stratified(pools[source][group], count, rng, used)
            for row in chosen:
                selected.append(dict(row, diagnostic_group=group, case_number=len(selected) + 1))
    train_segments = {r["segment_id"] for r in load_jsonl(args.data_dir / "train.jsonl")}
    assert not train_segments & used
    assert all(len(row["frames"]) == 5 and all(Path(f).is_file() for f in row["frames"]) for row in selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "cases.jsonl"
    if destination.exists() and load_jsonl(destination) != selected:
        raise ValueError("existing selection differs; use a new diagnostic directory")
    dump_jsonl(destination, selected)
    baseline = [dict(row["model_predictions"]["sft_control_final"], diagnostic={"case_number": row["case_number"], "group": row["diagnostic_group"]}) for row in selected]
    dump_jsonl(args.output_dir / "baseline_predictions.jsonl", baseline)
    manifest = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "scope": "Held-out development only; no official cases or weight updates. Stratified segment sample, not an unbiased prevalence estimate. Different segment IDs may share original recording scenes.",
                "pool": {source: {group: len(values) for group, values in groups.items()} for source, groups in pools.items()},
                "common_failure_question_buckets": dict(Counter(r["question_bucket"] for groups in pools.values() for r in groups["failure"])),
                "sample": dict(Counter(r["diagnostic_group"] for r in selected)),
                "sample_sources": dict(Counter(r["source_subset"] for r in selected)),
                "sample_buckets": dict(Counter(r["question_bucket"] for r in selected)),
                "segments": len(used), "train_segment_overlap": 0, "source_sha256": hashes,
                "cases_sha256": file_sha256(destination), "baseline_report_sha256": file_sha256(args.reference_dir / "report.json"),
                "model": dev_report["models"]["sft_control_final"],
                "conditions": {"full_replay": "Original five-frame input, check reproduction.",
                    "last_repeat": "Repeat the final-frame features five times: same visual-token count.",
                    "reverse": "Reverse the five-frame feature order: same frames/tokens, original question and references.",
                    "paired_duplicate": "At each time, full frame plus identical full frame, using the same paired-view prompt as ROI.",
                    "paired_roi": "At each time, full frame plus reviewer-localized region; original question/references unchanged. Oracle localization diagnostic only."}}
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def assets(args):
    from PIL import Image, ImageDraw
    cases = load_jsonl(args.output_dir / "cases.jsonl")
    folder = args.output_dir / "review_assets"
    folder.mkdir(exist_ok=True)
    for offset in range(0, len(cases), 5):
        selected = cases[offset:offset + 5]
        sheet = Image.new("RGB", (2200, 310 * len(selected)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(selected):
            y = 310 * j
            draw.text((8, y + 3), "Case %03d | %s | %s | %s" % (row["case_number"], row["diagnostic_group"], row["source_subset"], row["question_bucket"]), fill="black")
            draw.text((8, y + 19), row["question"][:160], fill="black")
            for i, path in enumerate(row["frames"]):
                with Image.open(path) as original:
                    frame = original.convert("RGB")
                    frame.thumbnail((438, 274))
                    sheet.paste(frame, (440 * i, y + 35))
        sheet.save(folder / ("sheet_%02d.jpg" % (offset // 5 + 1)), quality=95)
    render_review(args, cases)
    print(json.dumps({"cases": len(cases), "sheets": (len(cases) + 4) // 5, "assets": str(folder)}), flush=True)


def read_audit(args):
    path = args.output_dir / "review_audit.json"
    if not path.exists():
        return {}
    return {r["case_number"]: r for r in json.loads(path.read_text())}


def roi_assets(args):
    """Preview reviewed, frame-specific crops before any model comparison."""
    from PIL import Image, ImageDraw
    audit = read_audit(args)
    cases = [r for r in load_jsonl(args.output_dir / "cases.jsonl")
             if audit.get(r["case_number"], {}).get("roi")]
    folder = args.output_dir / "roi_review_assets"
    folder.mkdir(exist_ok=True)
    for offset in range(0, len(cases), 4):
        chosen = cases[offset:offset + 4]
        sheet = Image.new("RGB", (2200, 310 * len(chosen)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(chosen):
            roi = audit[row["case_number"]]["roi"]
            y = 310 * j
            draw.text((8, y + 3), "Case %03d | %s | %s" %
                      (row["case_number"], row["diagnostic_group"], roi["target"]), fill="black")
            for i, (path, box) in enumerate(zip(row["frames"], roi["frame_boxes"])):
                with Image.open(path) as raw:
                    im = raw.convert("RGB")
                    w, h = im.size
                    crop = im.crop((int(box[0] * w), int(box[1] * h),
                                    int(box[2] * w), int(box[3] * h)))
                    crop.thumbnail((438, 274))
                    sheet.paste(crop, (440 * i, y + 25))
        sheet.save(folder / ("roi_sheet_%02d.jpg" % (offset // 4 + 1)), quality=95)
    print(json.dumps({"roi_cases": len(cases), "sheets": (len(cases) + 3) // 4,
                      "folder": str(folder)}), flush=True)


def paired_prompt(question, frames=5):
    instruction = "The images are chronological observations from a driving video. Each time has two views: the full frame followed by an additional view of that same frame. Keep the original scene context when interpreting details. Answer the question using only visible evidence. Give a concise, factual answer."
    return build_user_prompt(question, frames * 2, instruction)


def transform_features(features, condition, crop_features=None):
    import torch
    if condition == "last_repeat":
        return features[-1:].expand_as(features).clone()
    if condition == "reverse":
        return features.flip(0)
    if condition == "paired_duplicate":
        return torch.stack((features, features), dim=1).flatten(0, 1)
    if condition == "paired_roi":
        if crop_features is None or crop_features.shape != features.shape:
            raise ValueError("ROI features must match all five original frames")
        return torch.stack((features, crop_features), dim=1).flatten(0, 1)
    if condition == "full_replay":
        return features
    raise ValueError(condition)


def crop_cache(args, cases, audit):
    import torch
    from PIL import Image
    from mpp_video_qa.model import build_visual_cache_model, add_upstream
    add_upstream(args.upstream_root)
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    folder = args.output_dir / "roi_cache"
    folder.mkdir(exist_ok=True)
    pending = []
    for row in cases:
        roi = audit[row["case_number"]]["roi"]
        assert len(roi["frame_boxes"]) == 5
        identity = {"question_id": row["question_id"], "frames": row["frames"], "roi": roi,
                    "bridge_init_sha256": file_sha256(args.bridge_init), "image_size": 224, "dtype": args.dtype}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        row["roi_cache_path"] = str(folder / (key + ".pt"))
        if not Path(row["roi_cache_path"]).is_file():
            pending.append(row)
    if not pending:
        return
    print(json.dumps({"stage": "loading_eva", "roi_cases": len(pending)}), flush=True)
    encoder = build_visual_cache_model(args).cuda().eval()
    processor = BlipImageEvalProcessor(image_size=224)
    dtype = next(encoder.visual_encoder.parameters()).dtype
    for index, row in enumerate(pending, 1):
        images = []
        for path, box in zip(row["frames"], audit[row["case_number"]]["roi"]["frame_boxes"]):
            assert len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
            with Image.open(path) as raw:
                image = raw.convert("RGB")
                w, h = image.size
                crop = image.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
                images.append(processor(crop))
        with torch.inference_mode():
            features = encoder.encode_image(torch.stack(images).cuda().to(dtype), feature_stage="eva").to(torch.bfloat16).cpu()
        torch.save(features, row["roi_cache_path"])
        print(json.dumps({"stage": "roi_cache", "completed": index, "total": len(pending)}), flush=True)
    del encoder
    torch.cuda.empty_cache()


def infer(args):
    import torch
    from mpp_video_qa.data import CachedQADataset
    from mpp_video_qa.model import CachedVideoQA, build_text_model
    from mpp_video_qa.core import TASK_INSTRUCTIONS
    cases = load_jsonl(args.output_dir / "cases.jsonl")
    audit = read_audit(args)
    roi_cases = [dict(r) for r in cases if audit.get(r["case_number"], {}).get("roi")]
    if any(c.startswith("paired_") for c in args.conditions):
        if not roi_cases:
            raise ValueError("no reviewed ROI cases; write review_audit.json first")
        if "paired_roi" in args.conditions:
            crop_cache(args, roi_cases, audit)
    dataset = CachedQADataset(args.output_dir / "cases.jsonl", args.cache_dir, "eva", validate_full_frames=True)
    indices = {r["question_id"]: i for i, r in enumerate(dataset.rows)}
    identity = {"protocol": PROTOCOL, "cases_sha256": file_sha256(args.output_dir / "cases.jsonl"),
                "adapter_identity": directory_fingerprint(args.adapter), "projector_sha256": file_sha256(args.projector),
                "qformer_sha256": file_sha256(args.qformer_adapter), "cache_dir": str(args.cache_dir.resolve()),
                "max_new_tokens": args.max_new_tokens, "max_txt_len": args.max_txt_len,
                "precision_protocol": args.precision_protocol, "dtype": args.dtype}
    model = None
    for condition in args.conditions:
        chosen = roi_cases if condition.startswith("paired_") else cases
        output = args.output_dir / condition
        output.mkdir(exist_ok=True)
        config = dict(identity, condition=condition)
        if condition.startswith("paired_"):
            config["roi_identity"] = {r["question_id"]: audit[r["case_number"]]["roi"] for r in chosen}
        manifest = output / "run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError("diagnostic identity changed: %s" % output)
        write_json_atomic(manifest, config)
        destination = output / "predictions.jsonl"
        done = {r["question_id"] for r in load_jsonl(destination)} if destination.exists() else set()
        if not done <= {r["question_id"] for r in chosen}:
            raise ValueError("prediction outside selected diagnostic pool")
        if len(done) == len(chosen):
            continue
        if model is None:
            print(json.dumps({"stage": "loading_sft_model", "cases": len(cases)}), flush=True)
            model = CachedVideoQA(build_text_model(args, train=False, adapter=args.adapter, projector=args.projector),
                                  task_instruction=TASK_INSTRUCTIONS["legacy_v1"], feature_stage="eva").cuda().eval()
        with destination.open("a") as stream:
            for row in chosen:
                if row["question_id"] in done:
                    continue
                features = dataset[indices[row["question_id"]]]["features"]
                crops = torch.load(row["roi_cache_path"], map_location="cpu") if condition == "paired_roi" else None
                inputs = transform_features(features, condition, crops).unsqueeze(0).cuda()
                prompts = [paired_prompt(row["question"])] if condition.startswith("paired_") else None
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = model.generate(inputs, [row["question"]], args.max_new_tokens, prompts=prompts)[0]
                torch.cuda.synchronize()
                result = {"question_id": row["question_id"], "segment_id": row["segment_id"],
                          "source_subset": row["source_subset"], "question": row["question"], "references": row["answers"],
                          "prediction": prediction, "latency_ms": 1000 * (time.perf_counter() - started),
                          "diagnostic": {"case_number": row["case_number"], "condition": condition, "group": row["diagnostic_group"]}}
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                done.add(row["question_id"])
                print(json.dumps({"condition": condition, "completed": len(done), "total": len(chosen)}), flush=True)
        write_json_atomic(output / "generation_summary.json", {"completed": len(done), "total": len(chosen),
                          "condition": condition, "max_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30})


def metrics(rows):
    return {"records": len(rows), "correct": sum(r["lingo_judge_correct"] for r in rows),
            "accuracy": statistics.mean(r["lingo_judge_correct"] for r in rows) if rows else None}


def comparison(a, b, cases):
    ids = set(b)
    assert ids <= set(a)
    return {"records": len(ids), "fixed": sum(not a[k]["lingo_judge_correct"] and b[k]["lingo_judge_correct"] for k in ids),
            "regressed": sum(a[k]["lingo_judge_correct"] and not b[k]["lingo_judge_correct"] for k in ids),
            "answer_changed": sum(a[k]["prediction"].lower().strip() != b[k]["prediction"].lower().strip() for k in ids),
            "fixed_cases": [cases[k]["case_number"] for k in sorted(ids) if not a[k]["lingo_judge_correct"] and b[k]["lingo_judge_correct"]],
            "regressed_cases": [cases[k]["case_number"] for k in sorted(ids) if a[k]["lingo_judge_correct"] and not b[k]["lingo_judge_correct"]]}


def report(args):
    cases = load_jsonl(args.output_dir / "cases.jsonl")
    by_id = {r["question_id"]: r for r in cases}
    baseline = {r["question_id"]: r for r in load_jsonl(args.output_dir / "baseline_predictions.jsonl")}
    result = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection": json.loads((args.output_dir / "manifest.json").read_text()), "conditions": {},
              "scope": "Selected development diagnostic; references unchanged; no training. Scores are official Judge proxies and may misjudge semantic errors. ROI localization is reviewer-assisted, not an automatic deployment result."}
    audit = read_audit(args)
    roi_ids = {r["question_id"] for r in cases if audit.get(r["case_number"], {}).get("roi")}
    predictions = {"baseline": baseline}
    for condition in CONDITIONS:
        path = args.output_dir / condition / "scored/scored_predictions.jsonl"
        if not path.exists():
            continue
        values = load_jsonl(path)
        keyed = {r["question_id"]: r for r in values}
        expected = roi_ids if condition.startswith("paired_") else set(by_id)
        assert len(values) == len(keyed) and set(keyed) == expected
        for row in values:
            original = by_id[row["question_id"]]
            assert (row["question"], row["references"], row["segment_id"]) == (original["question"], original["answers"], original["segment_id"])
        predictions[condition] = keyed
        result["conditions"][condition] = {"all": metrics(values), "vs_original": comparison(baseline, keyed, by_id),
            "groups": {g: metrics([r for r in values if by_id[r["question_id"]]["diagnostic_group"] == g]) for g in ("failure", "control")},
            "sources": {s: metrics([r for r in values if by_id[r["question_id"]]["source_subset"] == s]) for s in ("action", "scenery")},
            "review_categories": {label: metrics([r for r in values if audit.get(by_id[r["question_id"]]["case_number"], {}).get("primary_failure") == label])
                                  for label in sorted({a.get("primary_failure") for a in audit.values()})},
            "latency_ms_median": statistics.median(r["latency_ms"] for r in values)}
    if "full_replay" in predictions:
        replay = predictions["full_replay"]
        result["reproduction"] = dict(comparison(baseline, replay, by_id),
            exact_answer_matches=sum(baseline[k]["prediction"] == replay[k]["prediction"] for k in replay),
            original=metrics(list(baseline.values())), replay=metrics(list(replay.values())))
        for condition, value in result["conditions"].items():
            if condition != "full_replay":
                value["vs_full_replay"] = comparison(replay, predictions[condition], by_id)
    if "paired_roi" in predictions and "paired_duplicate" in predictions:
        assert predictions["paired_roi"].keys() == predictions["paired_duplicate"].keys()
        result["roi_vs_same_token_control"] = comparison(predictions["paired_duplicate"], predictions["paired_roi"], by_id)
        result["roi_vs_same_token_control"]["groups"] = {group: comparison(
            {k: p for k, p in predictions["paired_duplicate"].items() if by_id[k]["diagnostic_group"] == group},
            {k: p for k, p in predictions["paired_roi"].items() if by_id[k]["diagnostic_group"] == group}, by_id)
            for group in ("failure", "control")}
    result["review"] = {"records": len(audit), "labels": {key: dict(Counter(a.get(key, "missing") for a in audit.values()))
                        for key in ("reference_support", "sft_reference_consistency", "primary_failure", "confidence")},
                        "groups": {group: {key: dict(Counter(audit.get(r["case_number"], {}).get(key, "missing") for r in cases if r["diagnostic_group"] == group))
                                   for key in ("reference_support", "sft_reference_consistency", "primary_failure", "confidence")}
                                   for group in ("failure", "control")},
                        "reviewer": "Codex visual and text review; uncertain cases retained, not human gold labels."}
    if len(audit) != len(cases):
        result["review"]["missing_cases"] = [r["case_number"] for r in cases if r["case_number"] not in audit]
    semantic_path = args.output_dir / "intervention_review.json"
    semantic = json.loads(semantic_path.read_text()) if semantic_path.exists() else []
    result["semantic_intervention_review"] = {"records": len(semantic), "source": str(semantic_path),
            "labels": dict(Counter(r["verdict"] for r in semantic)),
            "sha256": file_sha256(semantic_path) if semantic_path.exists() else None}
    result["audit_sha256"] = file_sha256(args.output_dir / "review_audit.json")
    result["completed_conditions"] = list(result["conditions"])
    result["all_five_conditions_complete"] = set(result["conditions"]) == set(CONDITIONS)
    per_case = [dict(r, observation_predictions={k: v.get(r["question_id"]) for k, v in predictions.items()},
                     review=audit.get(r["case_number"]), intervention_review=[v for v in semantic if v["case_number"] == r["case_number"]]) for r in cases]
    dump_jsonl(args.output_dir / "case_results.jsonl", per_case)
    write_json_atomic(args.output_dir / "report.json", result)
    lines = ["# 开发集共同失败：观察验证", "", result["scope"], "",
             "本表是80道已知共同失败＋30道Judge正确对照的定向诊断，不能作为开发集总体准确率。paired两组为其中20失败＋8对照。", "",
             "| 条件 | 总题数 | Judge正确 | 原共同失败中Judge通过 | 原正确对照Judge失败 |", "|---|---:|---:|---:|---:|"]
    for key, value in result["conditions"].items():
        lines.append("| %s | %d | %d | %d | %d |" % (key, value["all"]["records"], value["all"]["correct"],
                    value["groups"]["failure"]["correct"], value["groups"]["control"]["records"] - value["groups"]["control"]["correct"]))
    if "roi_vs_same_token_control" in result:
        p = result["roi_vs_same_token_control"]
        lines += ["", "局部观察相对同token数量／同prompt对照：修复%d、退化%d。" % (p["fixed"], p["regressed"])]
    if "reproduction" in result:
        p = result["reproduction"]
        lines += ["", "原五帧重放：逐字一致%d/%d，Judge修复%d、退化%d。其余观察条件以新重放为基线，避免将复现差异误计为改进。" %
                  (p["exact_answer_matches"], p["records"], p["fixed"], p["regressed"]), "",
                  "| 条件 vs 新原五帧重放 | 修复 | 退化 | 回答变化 |", "|---|---:|---:|---:|"]
        for key, value in result["conditions"].items():
            if "vs_full_replay" in value:
                p = value["vs_full_replay"]
                lines.append("| %s | %d | %d | %d |" % (key, p["fixed"], p["regressed"], p["answer_changed"]))
    lines += ["", "视觉／文本复核（主因是暂定人工判断，不是已验证因果；不同segment可能来自相邻路段）：", "",
              "```json", json.dumps(result["review"], indent=2, ensure_ascii=False), "```",
              "", "逐项语义复核见intervention_review.json和内嵌图像HTML；完整结论见docs/dev_failure_validation_20261007.md。"]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    render_review(args, per_case)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def render_review(args, cases):
    from PIL import Image
    for group in ("failure", "control"):
        parts = ['<!doctype html><meta charset="utf-8"><title>开发集观察验证</title><style>body{font:16px system-ui;background:#f4f6f8;margin:20px}article{background:white;margin:16px 0;padding:16px}.frames{display:flex;overflow:auto;gap:4px}.frames img{width:384px}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}header{position:sticky;top:0;background:#f4f6f8;padding:8px}</style>',
                 '<header><h1>开发集：%s</h1><p>五帧按时间排序，图像内嵌，可离线打开；点击图片放大预览。复核与目标框仅用于诊断。</p><input placeholder="搜索问题、编号或复核" oninput="document.querySelectorAll(\'article\').forEach(x=>x.hidden=!x.innerText.toLowerCase().includes(this.value.toLowerCase()))"></header>' % group,
                 '<dialog id="viewer"><button onclick="document.getElementById(\'viewer\').close()">关闭</button><img style="max-width:90vw;max-height:85vh;display:block"></dialog>']
        for row in cases:
            if row["diagnostic_group"] != group:
                continue
            parts += ['<article><h2>Case %03d · %s · %s</h2><p>%s</p><p>参考：%s</p><div class="frames">' % (row["case_number"], row["source_subset"], row["question_bucket"], html.escape(row["question"]), html.escape(" / ".join(row["answers"])))]
            for frame_index, path in enumerate(row["frames"]):
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((768, 480))
                    buffer = io.BytesIO()
                    im.save(buffer, format="JPEG", quality=82)
                    uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
                    parts.append('<figure style="margin:0"><figcaption>Frame %d · full</figcaption><img src="%s" style="cursor:zoom-in" onclick="const d=document.getElementById(\'viewer\'); d.lastElementChild.src=this.src; d.showModal()"></figure>' % (frame_index, uri))
            parts.append('</div>')
            roi = (row.get("review") or {}).get("roi")
            if roi:
                parts.append('<p>推理前冻结的逐帧ROI（目标离开视野时回退全图）：</p><div class="frames">')
                for frame_index, (path, box) in enumerate(zip(row["frames"], roi["frame_boxes"])):
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        crop = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
                        crop.thumbnail((768, 480))
                        buffer = io.BytesIO()
                        crop.save(buffer, format="JPEG", quality=86)
                        uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
                        parts.append('<figure style="margin:0"><figcaption>Frame %d · %s</figcaption><img src="%s" style="cursor:zoom-in" onclick="const d=document.getElementById(\'viewer\'); d.lastElementChild.src=this.src; d.showModal()"></figure>' %
                                     (frame_index, "full fallback" if list(box) == [0, 0, 1, 1] else "ROI", uri))
                parts.append('</div>')
            parts += ['<table><tr><th>模型／条件</th><th>回答</th><th>Judge</th></tr>']
            for name, p in {**row["model_predictions"], **row.get("observation_predictions", {})}.items():
                if p:
                    parts.append('<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % (name, html.escape(p["prediction"]), p["lingo_judge_correct"]))
            parts.append('</table><pre>%s</pre><pre>%s</pre></article>' %
                         (html.escape(json.dumps(row.get("review"), indent=2, ensure_ascii=False)),
                          html.escape(json.dumps(row.get("intervention_review", []), indent=2, ensure_ascii=False))))
        (args.output_dir / ("review_%s.html" % group)).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    from mpp_video_qa.visual_bridge import add_bridge_arguments
    add_bridge_arguments(parser)
    parser.add_argument("action", choices=("prepare", "assets", "roi_assets", "infer", "report"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dev_failure_validation_v1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs/feedback_opd_eval_v2")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1"))
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["full_replay", "last_repeat", "reverse"])
    parser.add_argument("--upstream-root", default=str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"))
    parser.add_argument("--checkpoint-root", default="/tmp/zjr_mpp_fast/checkpoints")
    parser.add_argument("--adapter", default=str(ROOT / "outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/adapter"))
    parser.add_argument("--projector", default=str(ROOT / "outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/projector.pt"))
    parser.add_argument("--feature-stage", default="eva", choices=["eva"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--max-txt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["c_attn"])
    parser.set_defaults(bridge_init=str(ROOT / "checkpoints/visual_bridge_init.pt"),
                        visual_checkpoint_root=str(ROOT.parent / "mpp_llava_video_workspace/checkpoints"),
                        qformer_adapter=str(ROOT / "outputs/mpp_feedback_sft_control_40k_2epoch_v2/final/qformer_adapter.pt"),
                        qformer_mode="auto", precision_protocol="fp32_trainables_v1")
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
