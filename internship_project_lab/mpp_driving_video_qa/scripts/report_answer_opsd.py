#!/usr/bin/env python3
"""Progress and strictly paired dev comparisons for answer-conditioned OPD."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256

JUDGE_PROTOCOL = "official_lower_strip_maxref_logit_gt0_maxlength128_v1"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def line_count(path):
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def paired_metrics(base, candidate):
    if len({r["question_id"] for r in base}) != len(base) or len({r["question_id"] for r in candidate}) != len(candidate):
        raise ValueError("duplicate evaluation question IDs")
    b, c = ({r["question_id"]: r for r in rows} for rows in (base, candidate))
    if not b or set(b) != set(c):
        raise ValueError("evaluation question sets differ or are empty")
    for qid in b:
        if any(b[qid][key] != c[qid][key] for key in ("question", "references", "segment_id")):
            raise ValueError("evaluation question/reference/segment identity mismatch")
    result = {"records": len(b)}
    for metric in ("lingo_judge_correct", "token_f1", "normalized_exact"):
        delta = [float(c[q][metric]) - float(b[q][metric]) for q in b]
        result[metric] = {"fixed": sum(d > 0 for d in delta), "regressed": sum(d < 0 for d in delta),
                          "net_change_mean": sum(delta) / len(delta)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=("status", "report"), default="status")
    for name in ("data-dir", "probe-dir", "train-dir", "control-dir", "eval-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=4)
    args = p.parse_args()
    if args.action == "status":
        result = {"note": "File-based progress; process liveness is not verified.", "probe": {}, "training": {}, "eval": {}}
        for role in ("student", "teacher"):
            folder = args.probe_dir / role
            ident = read_json(folder / "run_manifest.json") or {}
            result["probe"][role] = {
                "generated": line_count(folder / "predictions.jsonl"),
                "total": line_count(args.data_dir / "probe.jsonl") * ident.get("num_samples", args.num_samples),
                "scored": (folder / "scored/summary.json").exists(),
                "scoring_progress": read_json(folder / "scored/score_progress.json"),
            }
        result["selection"] = read_json(args.data_dir / "train.manifest.json")
        if result["selection"]:
            result["selection"] = {k: result["selection"][k] for k in ("records", "selection_counts", "source_counts")}
        for name, folder in (("opsd", args.train_dir), ("sft_control", args.control_dir)):
            failures = sorted(folder.glob("failure_rank_*.json"))
            result["training"][name] = (read_json(failures[0]) if failures else
                read_json(folder / "summary.json") or read_json(folder / "progress.json") or {"status": "not_started"})
        for name in ("joint", "sft_control", "opsd"):
            for split in ("action", "scenery", "eval"):
                folder = args.eval_dir / (name + "_" + split)
                result["eval"][name + "_" + split] = {
                    "generated": line_count(folder / "predictions.jsonl"),
                    "scored": (folder / "scored/summary.json").exists()}
        print(json.dumps(result, indent=2))
        return
    result = {"judge_protocol": JUDGE_PROTOCOL, "splits": {},
              "interpretation": "Conditional teacher probe success is not held-out visual accuracy. Compare no-hint student generations against both joint and matched SFT."}
    for split in ("action", "scenery", "eval"):
        paths = {name: args.eval_dir / (name + "_" + split) for name in ("joint", "sft_control", "opsd")}
        if split == "eval" and not all((f / "scored/summary.json").exists() for f in paths.values()):
            continue
        data_path = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        expected = load_jsonl(data_path)
        expected_ids = {r["question_id"] for r in expected}
        rows, summaries = {}, {}
        for name, folder in paths.items():
            summaries[name] = read_json(folder / "scored/summary.json")
            if not summaries[name] or summaries[name].get("judge_protocol") != JUDGE_PROTOCOL:
                raise ValueError("complete official scoring required: %s" % folder)
            manifest = read_json(folder / "run_manifest.json") or {}
            if manifest.get("data_sha256") != file_sha256(data_path) or manifest.get("task_instruction_version") != "legacy_v1":
                raise ValueError("evaluation data or task instruction differs: %s" % folder)
            rows[name] = load_jsonl(folder / "scored/scored_predictions.jsonl")
            if {r["question_id"] for r in rows[name]} != expected_ids or len(rows[name]) != len(expected):
                raise ValueError("incomplete scored evaluation")
        result["splits"][split] = {
            "models": summaries,
            "joint_vs_sft_control": paired_metrics(rows["joint"], rows["sft_control"]),
            "joint_vs_opsd": paired_metrics(rows["joint"], rows["opsd"]),
            "sft_control_vs_opsd": paired_metrics(rows["sft_control"], rows["opsd"]),
        }
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    (args.eval_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Answer-conditioned OPD comparison", "", "No-reference, full-frame student evaluation.", "",
             "| Split | Model | N | LingoJudge | Token F1 |", "|---|---|---:|---:|---:|"]
    for split, report in result["splits"].items():
        for name, metrics in report["models"].items():
            lines.append("| %s | %s | %d | %.2f%% | %.4f |" %
                         (split, name, metrics["records"], 100 * metrics["lingo_judge_accuracy"], metrics["token_f1"]))
        for comparison in ("joint_vs_sft_control", "joint_vs_opsd", "sft_control_vs_opsd"):
            m = report[comparison]["lingo_judge_correct"]
            lines.append("\n%s / %s: fixed %d, regressed %d, net %+.2f pp." %
                         (split, comparison, m["fixed"], m["regressed"], m["net_change_mean"] * 100))
    lines.extend(["", result["interpretation"]])
    (args.eval_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
