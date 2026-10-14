#!/usr/bin/env python3
"""Fixed 48-video dev diagnostics. No training or official-eval samples are used."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256, normalize_answer
from scripts.diagnose_sft import action_prefix

MODELS = ("base", "old", "new")
CONDITIONS = ("A_original", "B_paraphrase", "C_format")
PARAPHRASE = "What are you doing at this moment, and what is the reason?"
FORMAT = ' Answer in the form "action, justification".'
OUT = ROOT / "outputs/full_v2_action_diagnosis_v1"
EVAL = ROOT / "outputs/full_v2_eval"
DATA = ROOT / "datasets/processed/lingoqa_full_v2"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def keyed(rows):
    result = {r["question_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate question IDs")
    return result


def cases_at(output):
    return load_jsonl(output / "cases.jsonl")


def prepare(args):
    dev = load_jsonl(DATA / "dev_action.jsonl")
    cases = sorted([r for r in dev if r["question"] == "What is your current action and why?"],
                   key=lambda r: r["question_id"])
    assert len(cases) == len({r["segment_id"] for r in cases}) == 48
    # Full-v2 preserves the original video-held-out Action dev set.
    legacy = keyed(load_jsonl(ROOT / "outputs/sft_diagnosis_v1/dev_action_audit.jsonl"))
    assert {r["question_id"] for r in cases} == set(legacy)
    for i, row in enumerate(cases, 1):
        assert all(row[k] == legacy[row["question_id"]][k] for k in ("frames", "answers", "segment_id", "question"))
        row["case_number"] = i
    rows = []
    for case in cases:
        for condition, question in ((CONDITIONS[1], PARAPHRASE), (CONDITIONS[2], PARAPHRASE + FORMAT)):
            row = dict(case, question=question, question_id=case["question_id"] + "::" + condition)
            row["diagnostic"] = {"condition": condition, "original_question_id": case["question_id"],
                                 "case_number": case["case_number"]}
            rows.append(row)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("cases.jsonl", cases), ("diagnostic.jsonl", rows)):
        path = output / filename
        if path.exists() and load_jsonl(path) != payload:
            raise RuntimeError("refusing to change existing diagnostic inputs: %s" % path)
        dump_jsonl(path, payload)
    write_json(output / "manifest.json", {
        "protocol": "full_v2_action_diagnosis_v1", "cases": 48, "models": list(MODELS),
        "conditions": {CONDITIONS[0]: cases[0]["question"], CONDITIONS[1]: PARAPHRASE,
                       CONDITIONS[2]: PARAPHRASE + FORMAT},
        "new_generations_per_model": 96, "reused_original_predictions_per_model": 48,
        "dev_sha256": file_sha256(DATA / "dev_action.jsonl"),
        "diagnostic_sha256": file_sha256(output / "diagnostic.jsonl"),
        "scope": "Fixed Action dev videos only. Diagnostic annotations must not be used as training targets.",
        "visual_control": "Reuse identical-question predictions; compare distinct videos, never interpret a permutation's overall accuracy as visual dependence.",
    })
    print(json.dumps({"output": str(output), "cases": 48, "new_generations_per_model": 96}))


def load_results(output):
    cases = cases_at(output)
    expected = keyed(load_jsonl(output / "diagnostic.jsonl"))
    result = {}
    for model in MODELS:
        original_dir = EVAL / (model + "_action")
        orig_manifest = json.loads((original_dir / "run_manifest.json").read_text())
        new_manifest = json.loads((output / model / "run_manifest.json").read_text())
        for field in ("adapter_identity", "cache_dir", "task_instruction_version", "max_new_tokens", "batch_size"):
            assert orig_manifest[field] == new_manifest[field], (model, field)
        assert orig_manifest["data_sha256"] == file_sha256(DATA / "dev_action.jsonl")
        assert new_manifest["data_sha256"] == file_sha256(output / "diagnostic.jsonl")
        for path in (original_dir, output / model):
            score_summary = json.loads((path / "scored/summary.json").read_text())
            assert score_summary["judge_protocol"] == "official_lower_strip_maxref_logit_gt0_maxlength128_v1"
            generation = json.loads((path / "generation_summary.json").read_text())
            assert generation["completed"] == generation["total"] == score_summary["records"]
        original = keyed(load_jsonl(original_dir / "scored/scored_predictions.jsonl"))
        added = keyed(load_jsonl(output / model / "scored/scored_predictions.jsonl"))
        assert set(added) == set(expected)
        result[model] = {c: {} for c in CONDITIONS}
        for case in cases:
            qid = case["question_id"]
            a = original[qid]
            assert a["segment_id"] == case["segment_id"] and a["references"] == case["answers"]
            result[model][CONDITIONS[0]][qid] = a
            for condition in CONDITIONS[1:]:
                row = added[qid + "::" + condition]
                assert row["question"] == expected[row["question_id"]]["question"]
                assert row["segment_id"] == case["segment_id"] and row["references"] == case["answers"]
                result[model][condition][qid] = row
    return result


def metrics(rows):
    rows = list(rows.values())
    return {"records": len(rows), "correct": sum(r["lingo_judge_correct"] for r in rows),
            "judge_accuracy": statistics.mean(r["lingo_judge_correct"] for r in rows),
            "token_f1": statistics.mean(r["token_f1"] for r in rows),
            "prefixes_heuristic_only": dict(Counter(action_prefix(r["prediction"]) for r in rows)),
            "unique_answers": len({normalize_answer(r["prediction"]) for r in rows})}


def paired(a, b):
    assert a.keys() == b.keys()
    return {"records": len(a),
            "fixed": sum(not a[k]["lingo_judge_correct"] and b[k]["lingo_judge_correct"] for k in a),
            "regressed": sum(a[k]["lingo_judge_correct"] and not b[k]["lingo_judge_correct"] for k in a),
            "answer_changed": sum(normalize_answer(a[k]["prediction"]) != normalize_answer(b[k]["prediction"]) for k in a),
            "prefix_changed_heuristic_only": sum(action_prefix(a[k]["prediction"]) != action_prefix(b[k]["prediction"]) for k in a)}


def report(args):
    results = load_results(args.output_dir)
    cases = cases_at(args.output_dir)
    payload = {"cases": 48, "models": {}, "visual_pairs": {}}
    for model, groups in results.items():
        payload["models"][model] = {"conditions": {c: metrics(rows) for c, rows in groups.items()},
            "A_to_B_wording": paired(groups[CONDITIONS[0]], groups[CONDITIONS[1]]),
            "B_to_C_format": paired(groups[CONDITIONS[1]], groups[CONDITIONS[2]])}
    # Every distinct-video pair, not a cherry-picked list. These are descriptive text
    # sensitivity counts; correctness is measured separately and semantic review is required.
    for model in MODELS:
        payload["visual_pairs"][model] = {}
        for c in CONDITIONS:
            rows = results[model][c]
            pairs = [(a, b) for i, a in enumerate(cases) for b in cases[i+1:]]
            differing_ref = [(a,b) for a,b in pairs if action_prefix(a["answers"][0]) != action_prefix(b["answers"][0])
                             and "other_wording" not in (action_prefix(a["answers"][0]), action_prefix(b["answers"][0]))]
            payload["visual_pairs"][model][c] = {
                "all_pairs": len(pairs),
                "different_answer_pairs": sum(normalize_answer(rows[a["question_id"]]["prediction"]) != normalize_answer(rows[b["question_id"]]["prediction"]) for a,b in pairs),
                "reference_prefix_different_pairs": len(differing_ref),
                "different_predicted_prefix_pairs": sum(action_prefix(rows[a["question_id"]]["prediction"]) != action_prefix(rows[b["question_id"]]["prediction"]) for a,b in differing_ref),
                "limitations": "Pairs share videos and are not independent samples. Prefixes are lexical heuristics, not action ground truth or temporal understanding."
            }
    audit_path = args.output_dir / "visual_audit.json"
    if audit_path.exists():
        audits = json.loads(audit_path.read_text())
        assert {a["question_id"] for a in audits} == {c["question_id"] for c in cases}
        assert len(audits) == len(cases)
        payload["visual_audit"] = {"records": len(audits), "reference_action_support": dict(Counter(a["reference_action_support"] for a in audits)),
                                   "new_A_action_vs_reference": dict(Counter(a["new_A_action_vs_reference"] for a in audits)),
                                   "new_A_reason_vs_reference": dict(Counter(a["new_A_reason_vs_reference"] for a in audits)),
                                   "reviewer": "Codex visual/text review; not human ground truth"}
        semantic = {}
        for letter, condition in zip("ABC", CONDITIONS):
            action_key = "new_%s_action_vs_reference" % letter
            reason_key = "new_%s_reason_vs_reference" % letter
            if not all(action_key in a and reason_key in a for a in audits):
                continue
            both = [a for a in audits if a[action_key] == a[reason_key] == "correct"]
            semantic[condition] = {
                "action": dict(Counter(a[action_key] for a in audits)),
                "reason": dict(Counter(a[reason_key] for a in audits)),
                "fully_reference_consistent": len(both),
                "judge_positive_but_not_fully_reference_consistent": [a["case_number"] for a in audits
                    if results["new"][condition][a["question_id"]]["lingo_judge_correct"] and a not in both],
                "fully_reference_consistent_but_judge_negative": [a["case_number"] for a in both
                    if not results["new"][condition][a["question_id"]]["lingo_judge_correct"]],
            }
        payload["visual_audit"]["new_model_semantic_review"] = semantic
        payload["visual_audit"]["coverage"] = "48 five-frame videos visually reviewed; new model A/B/C (144 answers) reviewed against reference semantics. Base/old use automatic scores; their full outputs are displayed for inspection."
        payload["visual_audit"]["label_definitions"] = {
            "correct": "Matches the reference's relevant semantic content; not necessarily visually verified.",
            "partial": "Compatible but incomplete or different granularity (e.g. slowing versus stopping).",
            "conflict": "Contradictory action/state/direction or a different causal explanation from reference; does not alone prove visual falsehood.",
            "not_addressed": "Misses the requested/reference dimension, e.g. answers speed instead of lane position.",
            "reference_action_support": "supported=visible main action; partial=only some components or context visible; uncertain=insufficient motion/intent evidence; questionable=possible annotation inconsistency, not adjudicated."
        }
    payload["limitations"] = ["48 fixed dev videos; no official-eval input or retraining.",
        "Judge correctness is a text proxy; semantic contradictions require review.",
        "A-to-B changes wording, B-to-C adds format; only one paraphrase was tested.",
        "Different videos do not isolate temporal information from static frame content."]
    write_json(args.output_dir / "report.json", payload)
    dump_jsonl(args.output_dir / "case_predictions.jsonl", [dict(case, predictions={m:{c:results[m][c][case["question_id"]] for c in CONDITIONS} for m in MODELS}) for case in cases])
    lines = ["# 全量 SFT：48 段开发视频动作诊断", "", "| 模型 | A 原问 | B 改写 | C 加格式 | A→B 修复/退化 | B→C 修复/退化 |", "|---|---:|---:|---:|---:|---:|"]
    for m, v in payload["models"].items():
        scores = ["%.2f%% (%s/48)" % (100*v["conditions"][c]["judge_accuracy"],v["conditions"][c]["correct"]) for c in CONDITIONS]
        p,q=v["A_to_B_wording"],v["B_to_C_format"]
        lines.append("| %s | %s | %s | %s | %s/%s | %s/%s |" % (m,*scores,p["fixed"],p["regressed"],q["fixed"],q["regressed"]))
    lines += ["", "## 解释边界", ""] + ["- "+x for x in payload["limitations"]]
    (args.output_dir / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,ensure_ascii=False))


def assets(args):
    from PIL import Image, ImageDraw
    cases=cases_at(args.output_dir)
    folder=args.output_dir / "review_assets"; folder.mkdir(exist_ok=True)
    for offset in range(0,len(cases),4):
        sheet=Image.new("RGB",(2400,4*300),"white"); draw=ImageDraw.Draw(sheet)
        for j,case in enumerate(cases[offset:offset+4]):
            draw.text((5,j*300+3),"Case %02d | %s" % (case["case_number"],case["question_id"]),fill="black")
            for i,path in enumerate(case["frames"]):
                with Image.open(path) as im:
                    im=im.convert("RGB"); im.thumbnail((480,270)); sheet.paste(im,(480*i,j*300+23))
        sheet.save(folder/("sheet_%02d.jpg"%(offset//4+1)),quality=92)
    print(json.dumps({"sheets":12,"directory":str(folder)}))


def review(args):
    from PIL import Image
    cases=cases_at(args.output_dir)
    try: results=load_results(args.output_dir)
    except FileNotFoundError: results=None
    audit_path=args.output_dir/"visual_audit.json"
    audits={a["case_number"]:a for a in json.loads(audit_path.read_text())} if audit_path.exists() else {}
    parts=['<!doctype html><meta charset="utf-8"><title>48 段动作诊断复核</title>',
      '<style>body{font-family:system-ui;margin:24px;background:#f4f6f8;color:#17202a}article{background:white;padding:16px;margin:20px 0;border:1px solid #ddd} .frames{display:flex;gap:4px;overflow:auto}.frames img{width:360px;height:auto;cursor:zoom-in}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap}summary{cursor:pointer;font-weight:bold}header{position:sticky;top:0;background:#f4f6f8;padding:10px}input{padding:8px}</style>',
      '<header><h1>48 段开发视频：动作与原因诊断</h1><p>帧从左到右按时间排序；点击图像查看大图。离线 HTML，图片已内嵌，无需服务器。</p><p>复核由 Codex 完成，不是人工真值；动作/原因与参考答案的一致性，与视觉可确认程度分别记录。</p><input id="filter" placeholder="搜索编号、问题或复核内容" oninput="document.querySelectorAll(\'article\').forEach(x=>x.hidden=!x.innerText.toLowerCase().includes(this.value.toLowerCase()))"></header>']
    for case in cases:
        parts.append('<article><h2>Case %02d · %s</h2><div class="frames">'%(case["case_number"],html.escape(case["question_id"])))
        for i,path in enumerate(case["frames"]):
            with Image.open(path) as im:
                im=im.convert("RGB"); im.thumbnail((1280,800)); buf=io.BytesIO(); im.save(buf,format="JPEG",quality=87)
            uri="data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
            parts.append('<div>Frame %d<br><img src="%s" onclick="document.getElementById(\'large\').src=this.src;document.getElementById(\'viewer\').showModal()"></div>'%(i,uri))
        parts.append('</div><details><summary>参考答案与复核</summary><p>%s</p><pre>%s</pre></details>'%(html.escape(case["answers"][0]),html.escape(json.dumps(audits.get(case["case_number"],{"status":"pending"}),ensure_ascii=False,indent=2))))
        if results:
            parts.append('<details><summary>三模型 × 三问法回答（Judge 仅作自动参考）</summary><table><tr><th>模型</th><th>条件</th><th>回答</th><th>Judge</th></tr>')
            for model in MODELS:
                for c in CONDITIONS:
                    r=results[model][c][case["question_id"]]
                    parts.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'%(model,c,html.escape(r["prediction"]),r["lingo_judge_correct"]))
            parts.append('</table></details>')
        parts.append('</article>')
    parts.append('<dialog id="viewer" style="max-width:95vw" onclick="this.close()"><img id="large" style="max-width:90vw"><p>点击关闭</p></dialog>')
    path=args.output_dir/"review.html"; path.write_text("\n".join(parts),encoding="utf-8")
    print(str(path))


def status(args):
    out={"models":{},"report_ready":(args.output_dir/"report.json").exists(),"process_liveness_verified":False}
    for model in MODELS:
        p=args.output_dir/model/"predictions.jsonl"
        count=sum(bool(l.strip()) for l in p.open()) if p.exists() else 0
        summary=args.output_dir/model/"scored/summary.json"
        out["models"][model]={"generated":count,"total":96,"scored":summary.exists()}
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","report","assets","review","status"))
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    globals()[args.action](args)
