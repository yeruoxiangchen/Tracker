#!/usr/bin/env python3
"""External visual teacher -> train-only offline distillation -> matched SFT.

The external teacher never receives reference answers. A positive paired dev
teacher interval is required before producing training supervision. All GPU
commands run as subprocesses so the teacher, Judge and student release memory
between stages. The user controls long jobs through the shell entrypoint.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256, TASK_INSTRUCTIONS
from mpp_video_qa.distillation import assert_disjoint, validate_predictions, select_training_rows
from scripts.generate_vlm_teacher import write_json, model_identity as teacher_identity
from scripts.prepare_visual_adaptation import stratified_subset, write_rows


def read(path):
    return json.loads(path.read_text())


def run(args, *command):
    command = [str(v) for v in command]
    print("$ " + " ".join(shlex.quote(v) for v in command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(ROOT) + ":" + str(args.upstream) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def immutable_json(path, value):
    if path.exists() and read(path) != value:
        raise ValueError("existing experiment identity differs: %s" % path)
    write_json(path, value)


def weights(args, model):
    root = args.parent if model == "baseline" else args.output_dir / ("train_" + model)
    return ["--adapter", root / "final_adapter", "--projector", root / "final_projector.pt",
            "--qformer-adapter", root / "final_qformer_adapter.pt", "--qformer-mode", "auto"]


def common(args):
    return ["--upstream-root", args.upstream, "--checkpoint-root", args.checkpoints,
            "--visual-checkpoint-root", args.visual_checkpoints, "--bridge-init", args.bridge,
            "--feature-stage", "eva", "--precision-protocol", "fp32_trainables_v1",
            "--task-instruction-version", "legacy_v1"]


def score(args, folder):
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    summary = folder / "scored/summary.json"
    stamp = folder / "scored/input_identity.json"
    identity = {"predictions_sha256": file_sha256(folder / "predictions.jsonl"),
                "judge_protocol": JUDGE_PROTOCOL}
    if summary.exists() and stamp.exists() and read(stamp) == identity:
        checked_scored(folder, load_jsonl(args.current_data))
        return
    run(args, args.python, "scripts/evaluate_answers.py", "--predictions", folder / "predictions.jsonl",
        "--output-dir", folder / "scored", "--lingo-judge")
    if not args.dry_run:
        write_json(stamp, identity)


def checked_scored(folder, originals):
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    from scripts.report_preference import summarize
    summary = read(folder / "scored/summary.json")
    if summary.get("judge_protocol") != JUDGE_PROTOCOL:
        raise ValueError("official Judge scoring is required: %s" % folder)
    generated = validate_predictions(originals, load_jsonl(folder / "predictions.jsonl"))
    rows = load_jsonl(folder / "scored/scored_predictions.jsonl")
    validate_predictions(originals, rows)
    if any(row["prediction"] != generated[row["question_id"]]["prediction"] for row in rows):
        raise ValueError("stale scoring belongs to different generated answers")
    if any(row.get("finished") != generated[row["question_id"]].get("finished") for row in rows):
        raise ValueError("scored completion flags differ from generated answers")
    if any(abs(summary.get(k, -1) - v) > 1e-9 for k, v in summarize(rows).items()):
        raise ValueError("scoring summary differs from its rows")
    return rows


def prepare_probes(args):
    train = load_jsonl(args.source_dir / "train.jsonl")
    dev = load_jsonl(args.source_dir / "dev.jsonl")
    evaluation = load_jsonl(args.source_dir / "eval.jsonl")
    assert_disjoint(train, dev, evaluation)
    train_probe = stratified_subset(train, args.train_probe_size, args.seed, 2)
    dev_probe = stratified_subset(dev, args.dev_probe_size, args.seed, 2)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_probe", train_probe), ("dev_probe", dev_probe)):
        write_rows(args.data_dir / (name + ".jsonl"), rows)
    immutable_json(args.data_dir / "probe_manifest.json", {
        "source_dir": str(args.source_dir.resolve()), "source_train_sha256": file_sha256(args.source_dir / "train.jsonl"),
        "seed": args.seed, "max_qa_per_segment": 2,
        "sampling": "source proportional, round robin over videos, independent of correctness",
        "splits": {name: {"records": len(rows), "segments": len({r["segment_id"] for r in rows}),
                            "source_counts": dict(Counter(r["source_subset"] for r in rows)),
                            "sha256": file_sha256(args.data_dir / (name + ".jsonl"))}
                   for name, rows in (("train_probe", train_probe), ("dev_probe", dev_probe))},
        "held_out_sha256": {n: file_sha256(args.source_dir / (n + ".jsonl"))
                            for n in ("dev", "dev_action", "dev_scenery", "eval")}})


def ensure_probes(args):
    if not (args.data_dir / "probe_manifest.json").exists():
        prepare_probes(args)
    manifest = read(args.data_dir / "probe_manifest.json")
    if (manifest["seed"] != args.seed or manifest["splits"]["train_probe"]["records"] != args.train_probe_size
            or manifest["splits"]["dev_probe"]["records"] != args.dev_probe_size
            or manifest["source_dir"] != str(args.source_dir.resolve())):
        raise ValueError("probe sampling settings differ; use a new output directory")
    for name, spec in manifest["splits"].items():
        if file_sha256(args.data_dir / (name + ".jsonl")) != spec["sha256"]:
            raise ValueError("probe data changed")
    for name, sha in manifest["held_out_sha256"].items():
        if file_sha256(args.source_dir / (name + ".jsonl")) != sha:
            raise ValueError("held-out data changed")


def baseline(args, split):
    from scripts.report_full_qformer_continue import parent_evaluation_identity
    from scripts.report_full_qformer_scale import load_result
    if not hasattr(args, "baseline_identity"):
        args.baseline_identity = parent_evaluation_identity(args.parent)
    data = args.source_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
    return load_result(args.parent_eval / ("epoch-01_" + split), data, args.baseline_identity)


def reuse_student_dev(args):
    from scripts.report_preference import summarize
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    from scripts.report_full_qformer_continue import identity
    rows = baseline(args, "action")[0] + baseline(args, "scenery")[0]
    lookup = {r["question_id"]: r for r in rows}
    selected = [lookup[r["question_id"]] for r in load_jsonl(args.data_dir / "dev_probe.jsonl")]
    folder = args.output_dir / "probe_dev/student"
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / "run_manifest.json", dict(identity(args.parent, final=True),
        data_sha256=file_sha256(args.data_dir / "dev_probe.jsonl"), feature_stage="eva",
        precision_protocol="fp32_trainables_v1", task_instruction_version="legacy_v1",
        reused_from=str(args.parent_eval), max_new_tokens=96))
    write_rows(folder / "predictions.jsonl", selected)
    (folder / "scored").mkdir(exist_ok=True)
    write_rows(folder / "scored/scored_predictions.jsonl", selected)
    write_json(folder / "scored/summary.json", dict(summarize(selected), judge_protocol=JUDGE_PROTOCOL))


def generate_teacher(args, name):
    data = args.data_dir / (name + "_probe.jsonl")
    folder = args.output_dir / ("probe_" + name) / "teacher"
    run(args, args.teacher_python, "scripts/generate_vlm_teacher.py", "--data", data,
        "--model", args.teacher_model, "--output-dir", folder, "--max-pixels", args.teacher_max_pixels,
        "--max-new-tokens", 96, "--task-instruction-version", getattr(args, "teacher_task_instruction_version", "legacy_v1"),
        "--input-mode", getattr(args, "teacher_input_mode", "multi_image"),
        "--frame-interval-seconds", getattr(args, "teacher_frame_interval_seconds", 1.0))
    args.current_data = data
    if not args.dry_run:
        score(args, folder)


def check_teacher_settings(args, manifest):
    if (manifest["task_instruction_version"] != getattr(args, "teacher_task_instruction_version", "legacy_v1")
            or manifest.get("input_mode", "multi_image") != getattr(args, "teacher_input_mode", "multi_image")):
        raise ValueError("teacher input mode or task instruction differs from this experiment")
    if (manifest.get("input_mode") == "video" and
            manifest.get("frame_interval_seconds") != getattr(args, "teacher_frame_interval_seconds", 1.0)):
        raise ValueError("teacher video timing differs from this experiment")


def teacher_report(args):
    from scripts.report_preference import summarize, cluster_interval
    from scripts.report_answer_opsd import paired_metrics
    from scripts.report_full_qformer_continue import identity
    data = args.data_dir / "dev_probe.jsonl"
    originals = load_jsonl(data)
    folders = {role: args.output_dir / "probe_dev" / role for role in ("student", "teacher")}
    rows = {role: checked_scored(folder, originals) for role, folder in folders.items()}
    t = read(folders["teacher"] / "run_manifest.json")
    s = read(folders["student"] / "run_manifest.json")
    check_teacher_settings(args, t)
    if (t.get("references_in_prompt") is not False or t["data_sha256"] != file_sha256(data)
            or t["model"] != teacher_identity(args.teacher_model)
            or t["max_pixels"] != args.teacher_max_pixels
            or s["data_sha256"] != file_sha256(data)
            or any(s.get(k) != v for k, v in identity(args.parent, final=True).items())):
        raise ValueError("teacher/student probe settings do not match this experiment")
    paired = dict(paired_metrics(rows["student"], rows["teacher"]), **cluster_interval(rows["student"], rows["teacher"]))
    usable = [dict(r, lingo_judge_correct=bool(r["lingo_judge_correct"] and r.get("finished") is True))
              for r in rows["teacher"]]
    usable_paired = dict(paired_metrics(rows["student"], usable), **cluster_interval(rows["student"], usable))
    report = {"models": {r: summarize(v) for r, v in rows.items()}, "paired": paired,
              "usable_teacher_paired": usable_paired,
              "teacher_complete_rate": sum(r.get("finished") is True for r in rows["teacher"]) / len(originals),
              "teacher_qualified": usable_paired["net_change_95pct_interval"][0] > 0,
              "qualification": "positive lower bound of paired video-bootstrap 95% interval; unfinished teacher outputs count as incorrect",
              "note": "Judge-based evidence; inspect review.html for visual correctness. This is development data.",
              "groups": {}, "teacher_identity": t,
              "student_identity": s}
    for source in sorted({r["source_subset"] for r in originals}):
        ids = {r["question_id"] for r in originals if r["source_subset"] == source}
        sub = {k: [r for r in v if r["question_id"] in ids] for k, v in rows.items()}
        report["groups"][source] = {"models": {r: summarize(v) for r, v in sub.items()},
                                   "paired": paired_metrics(sub["student"], sub["teacher"])}
    write_json(args.output_dir / "teacher_report.json", report)
    run(args, args.python, "scripts/compare_teachers.py", "report", "--data", data,
        "--output-dir", args.output_dir / "teacher_review", "--baseline", "student", "--scored",
        "student=" + str(folders["student"] / "scored/scored_predictions.jsonl"),
        "teacher=" + str(folders["teacher"] / "scored/scored_predictions.jsonl"))
    print(json.dumps(report, indent=2), flush=True)
    return report


def require_teacher(args):
    from scripts.report_full_qformer_continue import identity
    report = read(args.output_dir / "teacher_report.json")
    if not report["teacher_qualified"]:
        raise ValueError("teacher has not demonstrated a positive paired dev advantage; see teacher_report.json")
    check_teacher_settings(args, report["teacher_identity"])
    if report["teacher_identity"]["model"] != teacher_identity(args.teacher_model):
        raise ValueError("teacher differs from the qualified model")
    if (report["teacher_identity"]["max_pixels"] != args.teacher_max_pixels
            or report["teacher_identity"]["data_sha256"] != file_sha256(args.data_dir / "dev_probe.jsonl")
            or any(report["student_identity"].get(k) != v for k, v in identity(args.parent, final=True).items())):
        raise ValueError("qualified teacher settings/student/probe changed")
    return report


def prepare_training(args):
    gate = require_teacher(args)
    data = args.data_dir / "train_probe.jsonl"
    folder = args.output_dir / "probe_train/student"
    run(args, args.python, "scripts/generate_answers.py", "--data", data,
        "--cache-dir", args.cache, "--output-dir", folder, *common(args), *weights(args, "baseline"),
        "--max-new-tokens", 96)
    if args.dry_run:
        generate_teacher(args, "train")
        return
    args.current_data = data
    score(args, folder)
    generate_teacher(args, "train")
    originals = load_jsonl(data)
    student = checked_scored(folder, originals)
    teacher_folder = args.output_dir / "probe_train/teacher"
    teacher = checked_scored(teacher_folder, originals)
    t = read(teacher_folder / "run_manifest.json")
    if any(t[k] != gate["teacher_identity"][k] for k in t if k != "data_sha256"):
        raise ValueError("training teacher differs from the development probe teacher")
    selected, manifest = select_training_rows(originals, student, teacher, args.replay_fraction, args.seed)
    write_rows(args.data_dir / "train.jsonl", selected)
    manifest.update(train_sha256=file_sha256(args.data_dir / "train.jsonl"),
        probe_sha256=file_sha256(data), teacher_report_sha256=file_sha256(args.output_dir / "teacher_report.json"),
        student_identity=read(folder / "run_manifest.json"), teacher_identity=t,
        scored_sha256={r: file_sha256(args.output_dir / "probe_train" / r / "scored/scored_predictions.jsonl")
                       for r in ("student", "teacher")})
    immutable_json(args.data_dir / "train_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def train(args, mode):
    from scripts.report_full_qformer_continue import identity
    require_teacher(args)
    manifest = read(args.data_dir / "train_manifest.json")
    if manifest["train_sha256"] != file_sha256(args.data_dir / "train.jsonl"):
        raise ValueError("training data changed")
    initial = identity(args.parent, final=True)
    if any(manifest["student_identity"].get(k) != v for k, v in initial.items()):
        raise ValueError("student initialization differs from the training probe")
    if manifest["teacher_report_sha256"] != file_sha256(args.output_dir / "teacher_report.json"):
        raise ValueError("teacher qualification report changed")
    initial_args = ["--init-adapter", args.parent / "final_adapter", "--projector", args.parent / "final_projector.pt",
                    "--qformer-adapter", args.parent / "final_qformer_adapter.pt"]
    if args.resume_from:
        initial_args = ["--resume-from", args.resume_from]
    command = [args.python, "-u", "-m", "torch.distributed.run", "--nproc_per_node", args.nproc,
               "--master_port", args.master_port, "scripts/train_sft.py", *common(args), *initial_args,
               "--data", args.data_dir / "train.jsonl", "--dev-data", args.source_dir / "dev.jsonl",
               "--cache-dir", args.cache, "--output-dir", args.output_dir / ("train_" + mode),
               "--target-mode", "reference" if mode == "control" else "distillation",
               "--qformer-mode", "full", "--train-qformer", "--train-projector", "--qformer-gradient-checkpointing",
               "--gradient-checkpointing", "--epochs", args.epochs, "--batch-size", 1,
               "--grad-accumulation", 4, "--lr", "2.5e-6", "--projector-lr", "2.5e-6",
               "--qformer-lr", "5e-6", "--seed", args.seed, "--save-every", 200, "--max-dev-records", 64]
    run(args, *command)


def validate_matched(args):
    from scripts.report_full_qformer_continue import identity
    configs = [read(args.output_dir / ("train_" + m) / "run_config.json") for m in ("control", "distill")]
    keys = ("data_sha256", "epochs", "total_steps", "seed", "batch_size", "world_size", "grad_accumulation",
            "lr", "projector_lr", "qformer_lr", "weight_decay", "warmup_ratio", "qformer_mode",
            "precision_protocol", "feature_stage", "task_instruction_version", "initial_adapter_sha256",
            "initial_projector_sha256", "initial_qformer_sha256", "bridge_init_sha256")
    if any(configs[0].get(k) != configs[1].get(k) for k in keys):
        raise ValueError("control/distillation are not matched in data, initialization and training budget")
    if [c.get("target_mode", "reference") for c in configs] != ["reference", "distillation"]:
        raise ValueError("control/distillation target modes differ from the experiment")
    expected = identity(args.parent, final=True)
    mapping = {"initial_adapter_sha256": "adapter_identity", "initial_projector_sha256": "projector_identity",
               "initial_qformer_sha256": "qformer_adapter_sha256"}
    if any(configs[0].get(k) != expected[v] for k, v in mapping.items()):
        raise ValueError("training did not start from the full-data parent")
    if configs[0]["data_sha256"] != file_sha256(args.data_dir / "train.jsonl"):
        raise ValueError("training file differs from completed training")
    for mode, config in zip(("control", "distill"), configs):
        summary = read(args.output_dir / ("train_" + mode) / "summary.json")
        if not summary.get("completed") or summary["optimizer_steps"] != config["total_steps"]:
            raise ValueError("matched training has not completed")


def evaluate(args, model, split):
    data = args.source_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
    folder = args.output_dir / "eval" / (model + "_" + split)
    run(args, args.python, "scripts/generate_answers.py", "--data", data,
        "--cache-dir", args.cache, "--output-dir", folder, *common(args), *weights(args, model),
        "--max-new-tokens", 96)
    if not args.dry_run:
        args.current_data = data
        score(args, folder)


def build_report(args, official=False):
    from scripts.report_full_qformer_continue import identity
    from scripts.report_full_qformer_scale import load_result
    from scripts.report_preference import summarize, cluster_interval
    from scripts.report_answer_opsd import paired_metrics
    validate_matched(args)
    if official:
        choice = read(args.output_dir / "report.json")["selected_model"]
        models = ["baseline"] + ([choice] if choice != "baseline" else [])
    else:
        models = ["baseline", "control", "distill"]
    result = {"selection_split": "dev", "splits": {}, "paired": {},
              "experiment": "offline answer distillation with matched original-answer SFT"}
    pooled = {m: [] for m in models}
    for split in (("eval",) if official else ("action", "scenery")):
        data = args.source_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        rows = {}
        rows["baseline"], base_protocol = baseline(args, split)
        for model in models[1:]:
            rows[model], protocol = load_result(args.output_dir / "eval" / (model + "_" + split), data,
                identity(args.output_dir / ("train_" + model), final=True))
            if protocol != base_protocol:
                raise ValueError("baseline and candidate evaluation protocols differ")
        result["splits"][split] = {m: summarize(v) for m, v in rows.items()}
        for m in models:
            pooled[m].extend(rows[m])
    result["combined"] = {m: summarize(v) for m, v in pooled.items()}
    for a, b in (("baseline", "control"), ("baseline", "distill"), ("control", "distill")):
        if a in models and b in models:
            result["paired"][a + "_vs_" + b] = dict(paired_metrics(pooled[a], pooled[b]),
                                                      **cluster_interval(pooled[a], pooled[b]))
    result["selected_model"] = choice if official else max(models, key=lambda m: result["combined"][m]["lingo_judge_accuracy"])
    result["note"] = "Dev point estimate selects final weights; ties retain baseline/control. Bootstrap is exploratory after selection. Official data has been examined in earlier experiments."
    name = "official_report" if official else "report"
    write_json(args.output_dir / (name + ".json"), result)
    lines = ["# Teacher distillation comparison", "", "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    for split, metrics in dict(result["splits"], combined=result["combined"]).items():
        for model, metric in metrics.items():
            lines.append("| %s | %s | %d | %.2f%% | %.4f |" % (split, model, metric["records"],
                         metric["lingo_judge_accuracy"] * 100, metric["token_f1"]))
    for pair, value in result["paired"].items():
        metric, ci = value["lingo_judge_correct"], value["net_change_95pct_interval"]
        lines.append("\n%s: fixed %d, regressed %d, net %+.2f pp, video-bootstrap 95%% CI [%+.2f, %+.2f] pp." %
            (pair, metric["fixed"], metric["regressed"], metric["net_change_mean"] * 100, ci[0] * 100, ci[1] * 100))
    lines.extend(["", "Selected: `%s`." % result["selected_model"], "", result["note"]])
    (args.output_dir / (name + ".md")).write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def status(args):
    from scripts.report_answer_opsd import line_count
    result = {"process_liveness_verified": False, "output_dir": str(args.output_dir), "jobs": {}}
    workflow = args.output_dir / "workflow_progress.json"
    result["workflow"] = read(workflow) if workflow.exists() else {"status": "not_started"}
    for folder in sorted(args.output_dir.glob("**/progress.json")):
        value = read(folder)
        stamp = value.get("updated_unix_seconds", folder.stat().st_mtime)
        result["jobs"][str(folder.parent.relative_to(args.output_dir))] = dict(value, seconds_since_update=round(time.time()-stamp, 1))
    result["generation"] = {}
    for prefix, split in (("probe_dev", "dev_probe"), ("probe_train", "train_probe")):
        for role in ("student", "teacher"):
            folder = args.output_dir / prefix / role
            result["generation"][prefix + "/" + role] = {
                "generated": line_count(folder / "predictions.jsonl"),
                "total": line_count(args.data_dir / (split + ".jsonl")),
                "scored": (folder / "scored/summary.json").exists(),
                "scoring": read(folder / "scored/score_progress.json") if (folder / "scored/score_progress.json").exists() else None}
    result["training"] = {}
    for mode in ("control", "distill"):
        folder = args.output_dir / ("train_" + mode)
        failures = sorted(folder.glob("failure_rank_*.json"))
        newest_failure = max(failures, key=lambda p: p.stat().st_mtime) if failures else None
        progress = folder / "progress.json"
        summary = folder / "summary.json"
        path = summary if summary.exists() else (
            newest_failure if newest_failure and (not progress.exists() or newest_failure.stat().st_mtime >= progress.stat().st_mtime)
            else progress)
        result["training"][mode] = read(path) if path.exists() else {"status": "not_completed"}
    result["evaluation"] = {f.name: {"generated": line_count(f / "predictions.jsonl"),
                                    "scored": (f / "scored/summary.json").exists()}
                            for f in sorted((args.output_dir / "eval").glob("*")) if f.is_dir()}
    result["teacher_qualified"] = read(args.output_dir / "teacher_report.json")["teacher_qualified"] if (args.output_dir / "teacher_report.json").exists() else None
    result["report_ready"] = (args.output_dir / "report.json").exists()
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("probes", "teacher_compare", "teacher_report", "prepare", "train_control",
                                      "train_distill", "eval_dev", "report", "eval_selected", "status"))
    p.add_argument("--source-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_qformer_all_v1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/teacher_distillation_v1")
    p.add_argument("--parent", type=Path, default=ROOT / "outputs/mpp_lingoqa_qformer_all_1epoch_v1")
    p.add_argument("--parent-eval", type=Path, default=ROOT / "outputs/full_qformer_scale_eval_v1")
    p.add_argument("--teacher-model", type=Path, default=os.environ.get("TEACHER_MODEL", str(ROOT.parent / "chart_verification_vlm/checkpoints/Qwen3-VL-4B-Instruct")))
    p.add_argument("--teacher-python", default=os.environ.get("TEACHER_PYTHON", "/home/zjr/anaconda3/envs/chartvlm/bin/python"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--teacher-max-pixels", type=int, default=1003520)
    p.add_argument("--teacher-input-mode", choices=("multi_image", "video"), default="multi_image")
    p.add_argument("--teacher-task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    p.add_argument("--teacher-frame-interval-seconds", type=float, default=1.0)
    p.add_argument("--train-probe-size", type=int, default=10000)
    p.add_argument("--dev-probe-size", type=int, default=600)
    p.add_argument("--replay-fraction", type=float, default=.3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC_PER_NODE", "4")))
    p.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29689")))
    p.add_argument("--resume-from", default="")
    persistent = ROOT.parent / "mpp_llava_video_workspace/checkpoints"
    fast = Path("/tmp/zjr_mpp_fast/checkpoints")
    p.add_argument("--checkpoints", type=Path, default=os.environ.get("MPP_CHECKPOINT_ROOT", str(fast if (fast / "ckpt/Qwen7B-chat/config.json").exists() else persistent)))
    p.add_argument("--visual-checkpoints", type=Path, default=os.environ.get("MPP_VISUAL_CHECKPOINT_ROOT", str(persistent)))
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA")
    p.add_argument("--bridge", type=Path, default=ROOT / "checkpoints/visual_bridge_init.pt")
    p.add_argument("--cache", type=Path, default=os.environ.get("MPP_EVA_CACHE_DIR", "/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1"))
    p.add_argument("--dry-run", action="store_true", help="print GPU commands; data/report operations still execute")
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.output_dir / "data"
    if args.action == "status":
        status(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / "workflow_progress.json"
    write_json(progress, {"status": "running", "action": args.action, "pid": os.getpid(), "updated_unix_seconds": time.time()})
    try:
        ensure_probes(args)
        if args.action == "probes":
            print(json.dumps(read(args.data_dir / "probe_manifest.json"), indent=2), flush=True)
        elif args.action == "teacher_compare":
            reuse_student_dev(args)
            generate_teacher(args, "dev")
            if not args.dry_run:
                teacher_report(args)
        elif args.action == "teacher_report":
            teacher_report(args)
        elif args.action == "prepare":
            prepare_training(args)
        elif args.action.startswith("train_"):
            train(args, args.action.split("_", 1)[1])
        elif args.action == "eval_dev":
            validate_matched(args)
            for model in ("control", "distill"):
                for split in ("action", "scenery"):
                    evaluate(args, model, split)
            if not args.dry_run:
                build_report(args)
        elif args.action == "report":
            build_report(args)
        elif args.action == "eval_selected":
            report = build_report(args)
            if report["selected_model"] != "baseline":
                evaluate(args, report["selected_model"], "eval")
            if not args.dry_run:
                build_report(args, official=True)
    except Exception as error:
        write_json(progress, {"status": "failed", "action": args.action, "error": str(error), "updated_unix_seconds": time.time()})
        raise
    write_json(progress, {"status": "dry_run_completed" if args.dry_run else "completed",
                          "action": args.action, "updated_unix_seconds": time.time()})


if __name__ == "__main__":
    main()
