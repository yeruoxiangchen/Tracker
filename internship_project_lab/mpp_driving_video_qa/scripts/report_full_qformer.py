#!/usr/bin/env python3
"""Compare same-precision joint and epoch checkpoints; select using dev only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256
from scripts.report_answer_opsd import JUDGE_PROTOCOL, paired_metrics
from scripts.report_preference import cluster_interval, summarize
from scripts.generate_answers import write_json_atomic, directory_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--joint-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["joint", "epoch-01", "epoch-02"])
    args = p.parse_args()
    if args.models[0] != "joint":
        p.error("first model must be joint")
    rows = {model: [] for model in args.models}
    result = {"judge_protocol": JUDGE_PROTOCOL, "selection_split": "dev", "splits": {}}
    for split in ("action", "scenery"):
        data = args.data_dir / ("dev_%s.jsonl" % split)
        expected = {r["question_id"]: r for r in load_jsonl(data)}
        current, protocols = {}, []
        for model in args.models:
            directory = args.eval_dir / (model + "_" + split)
            run = json.loads((directory / "run_manifest.json").read_text())
            summary = json.loads((directory / "scored/summary.json").read_text())
            weights = args.joint_dir if model == "joint" else args.train_dir / model
            adapter = weights / ("final_adapter" if model == "joint" else "adapter")
            projector = weights / ("final_projector.pt" if model == "joint" else "projector.pt")
            bridge_hash = None if model == "joint" else file_sha256(weights / "qformer_adapter.pt")
            if (run["data_sha256"] != file_sha256(data) or
                run["adapter_identity"] != directory_fingerprint(adapter) or
                run["projector_identity"] != file_sha256(projector) or
                run.get("qformer_adapter_sha256") != bridge_hash or
                summary.get("judge_protocol") != JUDGE_PROTOCOL):
                raise ValueError("mismatched scoring/data/weights: %s" % directory)
            if run.get("precision_protocol") != "fp32_trainables_v1" or run["feature_stage"] != "eva":
                raise ValueError("evaluation must use EVA and the common precision protocol")
            protocols.append(tuple(run.get(k) for k in ("protocol", "cache_dir", "task_instruction_version",
                                                        "precision_protocol", "dtype", "max_new_tokens", "batch_size",
                                                        "bridge_init_sha256")))
            current[model] = load_jsonl(directory / "scored/scored_predictions.jsonl")
            if len(current[model]) != len(expected) or {r["question_id"] for r in current[model]} != set(expected):
                raise ValueError("incomplete or duplicate scored split")
            for row in current[model]:
                original = expected[row["question_id"]]
                if (row["question"], row["references"], row["segment_id"]) != (
                        original["question"], original["answers"], original["segment_id"]):
                    raise ValueError("scored row differs from the fixed development dataset")
            rows[model].extend(current[model])
        if len(set(protocols)) != 1:
            raise ValueError("generation protocols differ")
        result["splits"][split] = {m: summarize(r) for m, r in current.items()}
    result["dev_combined"] = {m: summarize(r) for m, r in rows.items()}
    result["paired"] = {m: dict(paired_metrics(rows["joint"], rows[m]),
                                 **cluster_interval(rows["joint"], rows[m])) for m in args.models[1:]}
    selected = max(args.models, key=lambda m: result["dev_combined"][m]["lingo_judge_accuracy"])
    result["selected_model"] = selected
    result["selection_note"] = "Dev point estimate; ties retain earlier model. Read video-bootstrap intervals before claiming an improvement."
    write_json_atomic(args.eval_dir / "report.json", result)
    lines = ["# Full Q-Former SFT development comparison", "",
             "Same EVA cache, original five frames and FP32 trainable-weight precision; official eval is not used for selection.", "",
             "| Model | Action | Scenery | Dev combined |", "|---|---:|---:|---:|"]
    for m in args.models:
        lines.append("| %s | %.2f%% | %.2f%% | %.2f%% |" % (m,
            100 * result["splits"]["action"][m]["lingo_judge_accuracy"],
            100 * result["splits"]["scenery"][m]["lingo_judge_accuracy"],
            100 * result["dev_combined"][m]["lingo_judge_accuracy"]))
    for m, value in result["paired"].items():
        paired = value["lingo_judge_correct"]
        interval = value["net_change_95pct_interval"]
        lines.append("\n%s: fixed %d, regressed %d, net %+.2f pp; video-bootstrap 95%% CI [%+.2f, %+.2f] pp." %
                     (m, paired["fixed"], paired["regressed"], paired["net_change_mean"] * 100,
                      interval[0] * 100, interval[1] * 100))
    lines.extend(["", "Selected by dev point estimate: `%s`." % selected, "", result["selection_note"]])
    (args.eval_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
