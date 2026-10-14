#!/usr/bin/env python3
"""Replay fixed OPD trajectories/batches without any optimizer or weight writes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import base64
import copy
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import best_reference_scores, file_sha256, load_jsonl
from mpp_video_qa.opd_diagnostics import (
    PROTOCOL, ARMS, digest, assert_split_isolation, select_trajectories,
    select_effective_batches, parameter_group, capture_gradients, add_gradients,
    gradient_geometry, tensor_fingerprint, select_review, validate_annotations,
    review_category)
from scripts.generate_answers import directory_fingerprint, write_json_atomic
from scripts.report_answer_opsd import JUDGE_PROTOCOL, paired_metrics
from scripts.report_preference import summarize
from scripts.report_full_qformer_scale import load_result
from scripts.feedback_opd_common import (
    add_runtime_arguments, configure_model, runtime_identity, training_dataset,
    build_student, Judge)
from mpp_video_qa.opd import SharedFrozenTeacher, sample_completions, completion_logits
from mpp_video_qa.feedback_opd import feedback_prompt, decode_completion, token_diagnostics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def append_row(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def progress(folder, stage, completed=0, total=0, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    value = dict(status=stage, pid=os.getpid(), completed=completed, total=total,
                 updated_unix_seconds=time.time(), **extra)
    write_json_atomic(folder / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def weight_spec(folder, parent=False):
    folder = Path(folder)
    paths = {"adapter": folder / ("final_adapter" if parent else "adapter"),
             "projector": folder / ("final_projector.pt" if parent else "projector.pt"),
             "qformer": folder / ("final_qformer_adapter.pt" if parent else "qformer_adapter.pt")}
    return {"paths": {k: str(v.resolve()) for k, v in paths.items()},
            "identity": {"adapter": directory_fingerprint(paths["adapter"]),
                         "projector": file_sha256(paths["projector"]),
                         "qformer": file_sha256(paths["qformer"])}}


def check_weights(spec):
    p = spec["paths"]
    actual = {"adapter": directory_fingerprint(p["adapter"]),
              "projector": file_sha256(p["projector"]), "qformer": file_sha256(p["qformer"])}
    if actual != spec["identity"]:
        raise ValueError("diagnostic weight bundle changed; use an independent output directory")


def training_log_summary(folder):
    by_step, raw = {}, 0
    for row in load_jsonl(Path(folder) / "training_log.jsonl"):
        raw += 1
        by_step[row["optimizer_step"]] = row
    result = {"raw_rows": raw, "unique_optimizer_steps": len(by_step),
              "duplicate_resume_rows_removed": raw - len(by_step), "epochs": {}}
    for epoch, bounds in ((1, (1, 2500)), (2, (2501, 5000))):
        selected = [r["step_metrics"] for s, r in by_step.items() if bounds[0] <= s <= bounds[1]]
        counts = Counter()
        for metrics in selected:
            counts.update(metrics)
        result["epochs"][str(epoch)] = {
            "steps": len(selected), "ce_mean": counts["ce_sum"] / max(1, counts["examples"]),
            "jsd_mean_eligible": counts["jsd_sum"] / max(1, counts["eligible"]),
            "eligible_rate": counts["eligible"] / max(1, counts["rollouts"]),
            "student_correct_rate": counts["student_correct"] / max(1, counts["rollouts"]),
            "teacher_correct_per_attempt": counts["teacher_correct"] / max(1, counts["teacher_attempts"])}
    result["caution"] = "Changing on-policy trajectories/gates: these epoch JSD means are not fixed-sample alignment."
    return result


def dev_pairs(args, models):
    rows, input_hashes = [], {}
    for split in ("action", "scenery"):
        path = Path(args.data_dir) / ("dev_%s.jsonl" % split)
        originals = {r["question_id"]: r for r in load_jsonl(path)}
        loaded = {}
        protocols = []
        for arm, key in (("parent", "baseline"), ("ce", "sft_control_final"),
                         ("opd", "feedback_opd_final")):
            identity = models[arm]["identity"]
            folder = Path(args.eval_dir) / (key + "_" + split)
            scored, protocol = load_result(folder, path, {
                "adapter_identity": identity["adapter"], "projector_identity": identity["projector"],
                "qformer_adapter_sha256": identity["qformer"]})
            loaded[arm] = {r["question_id"]: r for r in scored}
            protocols.append(protocol)
            for name in ("predictions.jsonl", "run_manifest.json", "scored/scored_predictions.jsonl",
                         "scored/summary.json"):
                input_hashes[str(folder / name)] = file_sha256(folder / name)
        if len(set(protocols)) != 1:
            raise ValueError("historical development generation protocols differ")
        for qid, original in originals.items():
            ce, opd = loaded["ce"][qid], loaded["opd"][qid]
            a, b = ce["lingo_judge_correct"], opd["lingo_judge_correct"]
            outcome = "fixed" if not a and b else "regressed" if a and not b else (
                "both_correct" if a else "both_wrong")
            if outcome in ("fixed", "regressed"):
                rows.append(dict(original, split=split, outcome=outcome,
                                 models={arm: {k: loaded[arm][qid][k] for k in
                                         ("prediction", "lingo_judge_logit", "lingo_judge_correct", "token_f1")}
                                         for arm in ARMS}))
    return rows, input_hashes


def prepare(args):
    output = Path(args.output_dir)
    if (output / "manifest.json").exists():
        manifest = require_manifest(args)
        for name in ("controls", "error_cases", "seed", "gradient_steps"):
            if manifest[name] != getattr(args, name):
                raise ValueError("prepared selection differs: %s; use a new output directory" % name)
        print(json.dumps({"status": "already_prepared", "cases": manifest["cases"],
                          "gradient_draws": manifest["gradient_draws"]}, indent=2))
        return
    progress(output, "preparing_inputs")
    configure_model(args, train=False)
    identity = runtime_identity(args)
    config = read(Path(args.train_dir) / "run_config.json")
    control = read(Path(args.control_dir) / "run_config.json")
    for name, folder, saved in (("opd", args.train_dir, config), ("ce", args.control_dir, control)):
        summary = read(Path(folder) / "summary.json")
        metadata = read(Path(folder) / "final/metadata.json")
        if (not summary.get("completed") or summary["optimizer_steps"] != saved["total_steps"] or
                metadata["identity"] != saved or metadata["optimizer_steps"] != saved["total_steps"]):
            raise ValueError("incomplete or unrelated %s final checkpoint" % name)
        expected_mode = "feedback_opd" if name == "opd" else "sft_control"
        if saved["training_mode"] != expected_mode:
            raise ValueError("wrong branch assigned to %s" % name)
        for key in ("initialization", "data_sha256", "base_checkpoint_sha256", "bridge_init_sha256",
                    "task_instruction_version", "max_txt_len", "max_new_tokens", "dtype",
                    "precision_protocol", "qformer_mode", "teacher_mode"):
            if saved.get(key) != identity.get(key):
                raise ValueError("runtime differs from original %s training: %s" % (name, key))
    for key in ("world_size", "grad_accumulation", "epochs", "total_steps", "lr", "projector_lr",
                "qformer_lr", "ce_weight", "weight_decay", "warmup_ratio", "records"):
        if config[key] != control[key]:
            raise ValueError("CE/OPD budgets differ: %s" % key)
    if (config["world_size"], config["grad_accumulation"], config["records"], config["total_steps"]) != (4, 4, 40000, 5000):
        raise ValueError("this replay is the completed four-GPU 40k/5000-step experiment")
    compare = Path(args.comparison_dir)
    source = Path(args.source_probe_dir)
    comparison_config = read(compare / "run_config.json")
    if not read(compare / "probe_report.json").get("completed"):
        raise ValueError("paired teacher comparison is incomplete")
    previous = comparison_config["identity_by_mode"]["reference_only"]
    for key in ("initialization", "data_sha256", "base_checkpoint_sha256", "bridge_init_sha256",
                "task_instruction_version", "max_txt_len", "max_new_tokens", "teacher_mode"):
        if previous.get(key) != identity.get(key):
            raise ValueError("cached teacher comparison differs: %s" % key)
    if file_sha256(source / "predictions.jsonl") != comparison_config["source_probe_sha256"]:
        raise ValueError("correct-control probe is not the source of the paired teacher comparison")
    original_rows = load_jsonl(args.data)
    originals = {r["question_id"]: r for r in original_rows}
    if len(originals) != len(original_rows):
        raise ValueError("duplicate training question IDs")
    heldout = {s: load_jsonl(Path(args.data_dir) / ("dev_%s.jsonl" % s)) for s in ("action", "scenery")}
    heldout["official_reused"] = load_jsonl(Path(args.data_dir) / "eval.jsonl")
    isolation = assert_split_isolation(original_rows, heldout)
    comparison = load_jsonl(compare / "predictions.jsonl")
    if len(comparison) != 200:
        raise ValueError("expected the completed 200-error teacher comparison")
    if args.error_cases:
        comparison = comparison[:args.error_cases]
    cases = select_trajectories(comparison, load_jsonl(source / "predictions.jsonl"),
                                originals, args.controls, args.seed)
    logs = {}
    for rank in range(config["world_size"]):
        logs[rank] = []
        with (Path(args.train_dir) / ("rollouts_rank_%d.jsonl" % rank)).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["micro_step"] // config["grad_accumulation"] + 1 in args.gradient_steps:
                    logs[rank].append(row)
    draws, duplicates = select_effective_batches(logs, originals, args.gradient_steps,
                                                 config["grad_accumulation"], config["world_size"])
    models = {"parent": weight_spec(args.initial_dir, True),
              "ce": weight_spec(Path(args.control_dir) / "final"),
              "opd": weight_spec(Path(args.train_dir) / "final")}
    if models["parent"]["identity"] != config["initialization"]:
        raise ValueError("teacher must stay at the original full SFT snapshot")
    changed, dev_hashes = dev_pairs(args, models)
    review_ids = select_review(cases, args.review_cases)
    annotations = [{"case_id": k, "reviewed": False, "reviewer": "",
                    "reference_consistent": None, "spans": [], "notes": ""} for k in review_ids]
    write_rows(output / "cases.jsonl", cases)
    write_rows(output / "gradient_draws.jsonl", draws)
    write_rows(output / "dev_changes.jsonl", changed)
    write_json_atomic(output / "review_annotations.template.json", annotations)
    source_paths = [Path(args.data), compare / "predictions.jsonl", compare / "run_config.json",
                    compare / "probe_report.json", source / "predictions.jsonl",
                    Path(args.train_dir) / "run_config.json", Path(args.control_dir) / "run_config.json",
                    Path(args.train_dir) / "training_log.jsonl"]
    source_paths += [Path(args.train_dir) / ("rollouts_rank_%d.jsonl" % r) for r in range(4)]
    source_paths += [Path(args.data_dir) / n for n in ("dev_action.jsonl", "dev_scenery.jsonl", "eval.jsonl")]
    manifest = {"protocol": PROTOCOL, "runtime_identity": identity, "models": models,
                "teacher": copy.deepcopy(models["parent"]), "teacher_mode": "reference_only",
                "training_config": config, "cases": len(cases), "controls": args.controls,
                "error_cases": args.error_cases, "seed": args.seed,
                "case_cohorts": dict(Counter(r["cohort"] for r in cases)),
                "case_sources": dict(Counter(r["source_subset"] for r in cases)),
                "case_teacher_eligible": sum(r["eligible"] for r in cases),
                "gradient_steps": args.gradient_steps, "gradient_draws": len(draws),
                "gradient_eligible_draws": sum(r["eligible"] for r in draws),
                "gradient_resume_duplicates_removed": duplicates,
                "split_isolation": isolation, "review_case_ids": review_ids,
                "dev_outcomes": dict(Counter(r["outcome"] for r in changed)),
                "data_path": str(Path(args.data).resolve()), "optimizer_updates": 0,
                "source_hashes": dict({str(p): file_sha256(p) for p in source_paths}, **dev_hashes),
                "artifact_hashes": {n: file_sha256(output / n) for n in
                    ("cases.jsonl", "gradient_draws.jsonl", "dev_changes.jsonl", "review_annotations.template.json")},
                "worker_code_hashes": {str(p): file_sha256(p) for p in
                    (Path(__file__), ROOT / "mpp_video_qa/opd_diagnostics.py")},
                "training_log": training_log_summary(args.train_dir)}
    write_json_atomic(output / "manifest.json", manifest)
    write_review_html(output / "review.html", cases, {}, review_ids, output)
    dev_ids = select_review([dict(r, case_id=r["question_id"], cohort=r["outcome"]) for r in changed], 40)
    write_dev_html(output / "dev_review.html", changed, dev_ids)
    progress(output, "prepared", len(cases), len(cases), gradient_draws=len(draws),
             dev_outcomes=manifest["dev_outcomes"])


def require_manifest(args):
    output = Path(args.output_dir)
    m = read(output / "manifest.json")
    if m["protocol"] != PROTOCOL or m["teacher"] != m["models"]["parent"] or m["teacher_mode"] != "reference_only":
        raise ValueError("teacher snapshot/condition changed")
    if m["data_path"] != str(Path(args.data).resolve()):
        raise ValueError("different diagnostic dataset")
    for name, value in m["artifact_hashes"].items():
        if file_sha256(output / name) != value:
            raise ValueError("prepared diagnostic sample changed: %s" % name)
    for path, value in m["worker_code_hashes"].items():
        if file_sha256(path) != value:
            raise ValueError("worker implementation changed; use a new diagnostic output directory")
    return m


def expected_rows(path, ids, key):
    rows = load_jsonl(path) if path.exists() else []
    seen = [r[key] for r in rows]
    if len(seen) != len(set(seen)) or not set(seen) <= set(ids):
        raise ValueError("duplicate or foreign resumed rows: %s" % path)
    return rows


def run_trajectories(args, model, teacher, dataset, indices, cases, folder):
    path = folder / "trajectories.jsonl"
    rows = expected_rows(path, [r["case_id"] for r in cases], "case_id")
    done = {r["case_id"] for r in rows}
    started, initial = time.time(), len(rows)
    tokenizer = model.backbone.llm_tokenizer
    model.eval()
    for row in cases:
        if row["case_id"] in done:
            continue
        progress(folder, "fixed_trajectories", len(rows), len(cases), current_case=row["case_id"])
        record = dataset[indices[row["question_id"]]]
        features = record["features"].unsqueeze(0).to(next(model.parameters()).device)
        ids = torch.tensor(row["student_token_ids"], device=features.device, dtype=torch.long)
        if decode_completion(model, ids) != row["student_answer"] or ids[-1].item() != tokenizer.im_end_id:
            raise ValueError("cached trajectory tokenizer/EOS mismatch")
        prompt = feedback_prompt(row, row["student_answer"], model.task_instruction, "reference_only")
        with torch.no_grad():
            with teacher.active() as frozen:
                plain = completion_logits(frozen, features, row["question"], ids, answer_only=True).cpu()
                targets = completion_logits(frozen, features, row["question"], ids, prompt, answer_only=True).cpu()
            logits = completion_logits(model, features, row["question"], ids, answer_only=True).cpu()
            tokens = token_diagnostics(logits, plain, targets, ids.cpu(), tokenizer)
            ce = float(model.answer_only_ce(features, [row["question"]], [record["answer"]]))
            generated_ids, finished = sample_completions(model, features, row["question"],
                                                         args.max_new_tokens, do_sample=False)
        answer = decode_completion(model, generated_ids)
        diagnostic = {"case_id": row["case_id"], "question_id": row["question_id"],
                      "segment_id": row["segment_id"], "source_subset": row["source_subset"],
                      "question": row["question"], "references": row["answers"], "cohort": row["cohort"],
                      "cached_token_ids_sha256": digest(row["student_token_ids"]),
                      "cached_teacher_eligible": row["eligible"], "teacher_prompt_sha256": digest(prompt),
                      "reference_ce": ce, "fixed_trajectory_jsd": statistics.mean(t["jsd"] for t in tokens),
                      "sampled_logp_mean": statistics.mean(t["student_log_prob"] for t in tokens),
                      "tokens": tokens, "prediction": answer, "generation_finished": finished,
                      "generated_token_ids": generated_ids.tolist()}
        append_row(path, diagnostic)
        rows.append(diagnostic)
        elapsed = time.time() - started
        progress(folder, "fixed_trajectories", len(rows), len(cases),
                 estimated_remaining_minutes=(len(cases) - len(rows)) * elapsed / max(1, len(rows) - initial) / 60)
        del logits, plain, targets, features
    return rows


def run_gradients(args, model, teacher, dataset, indices, draws, folder, manifest):
    from mpp_video_qa.opd import distillation_loss
    path = folder / "gradients.jsonl"
    steps = manifest["gradient_steps"]
    rows = expected_rows(path, steps, "effective_step")
    done = {r["effective_step"] for r in rows}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    shapes = {n: p.numel() for n, p in named}
    config = manifest["training_config"]
    for step in steps:
        if step in done:
            continue
        batch = [r for r in draws if r["effective_step"] == step]
        divisor = config["world_size"] * config["grad_accumulation"]
        if len(batch) != divisor or any(r["divisor"] != divisor for r in batch):
            raise ValueError("incomplete effective gradient batch")
        ce_total, jsd_total, details = {}, {}, []
        for row in batch:
            progress(folder, "gradient_replay", len(rows), len(steps),
                     effective_step=step, batch_completed=len(details), batch_total=len(batch))
            record = dataset[indices[row["question_id"]]]
            features = record["features"].unsqueeze(0).to(next(model.parameters()).device)
            ids = torch.tensor(row["student_token_ids"], device=features.device, dtype=torch.long)
            if decode_completion(model, ids) != row["student_answer"]:
                raise ValueError("training trajectory decode differs")
            # Match original execution: finish teacher first, JSD backward, then CE.
            # No adapter switching may occur between a checkpointed forward/backward.
            jsd_grads, divergence = {}, None
            if row["eligible"]:
                prompt = feedback_prompt(row, row["student_answer"], model.task_instruction, "reference_only")
                import hashlib
                if hashlib.sha256(prompt.encode()).hexdigest() != row.get("teacher_prompt_sha256"):
                    raise ValueError("replayed teacher condition differs from training")
                with teacher.active() as frozen:
                    targets = completion_logits(frozen, features, row["question"], ids, prompt, answer_only=True).detach()
                model.train()
                model.zero_grad(set_to_none=True)
                logits = completion_logits(model, features, row["question"], ids, answer_only=True)
                loss = distillation_loss(logits, targets, "jsd", config["chunk_tokens"])
                divergence = float(loss.detach())
                (config["opd_weight"] * loss / divisor).backward()
                jsd_grads = capture_gradients(named)
                del loss, logits, targets
            model.train()
            model.zero_grad(set_to_none=True)
            ce = model.answer_only_ce(features, [row["question"]], [record["answer"]])
            ce_value = float(ce.detach())
            (config["ce_weight"] * ce / divisor).backward()
            ce_grads = capture_gradients(named)
            model.zero_grad(set_to_none=True)
            geometry = gradient_geometry(ce_grads, jsd_grads, shapes)
            add_gradients(ce_total, ce_grads)
            add_gradients(jsd_total, jsd_grads)
            details.append({"case_id": row["case_id"], "question_id": row["question_id"],
                            "rank": row["rank"], "micro_step": row["micro_step"], "eligible": row["eligible"],
                            "gate_reason": row["gate_reason"], "reference_ce": ce_value, "jsd": divergence,
                            "gradients": geometry})
            del ce, ce_grads, jsd_grads, features
        value = {"effective_step": step, "draws": len(batch), "eligible_draws": sum(r["eligible"] for r in batch),
                 "divisor": divisor, "loss_weights": {"ce": config["ce_weight"], "jsd": config["opd_weight"]},
                 "gradients": gradient_geometry(ce_total, jsd_total, shapes), "examples": details,
                 "note": "Fixed cached gates, no new on-policy rollout, no optimizer step; geometry is not an AdamW update."}
        append_row(path, value)
        rows.append(value)
        del ce_total, jsd_total
    return rows


def infer(args):
    manifest = require_manifest(args)
    output = Path(args.output_dir)
    folder = output / args.arm
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": file_sha256(output / "manifest.json"), "arm": args.arm,
                "student": manifest["models"][args.arm], "teacher": manifest["teacher"],
                "teacher_mode": "reference_only", "answer_only_logits": True,
                "optimizer_updates": 0, "generation": "greedy_original_prompt_96_no_reference"}
    config_path = folder / "run_config.json"
    if config_path.exists() and read(config_path) != identity:
        raise ValueError("diagnostic resume model/sample identity changed")
    write_json_atomic(config_path, identity)
    cases = load_jsonl(output / "cases.jsonl")
    draws = load_jsonl(output / "gradient_draws.jsonl")
    trajectories = expected_rows(folder / "trajectories.jsonl", [r["case_id"] for r in cases], "case_id")
    gradients = expected_rows(folder / "gradients.jsonl", manifest["gradient_steps"], "effective_step")
    want_t = args.part in ("all", "trajectories")
    want_g = args.part in ("all", "gradients")
    summary_path = folder / "worker_summary.json"
    if ((not want_t or len(trajectories) == len(cases)) and
            (not want_g or len(gradients) == len(manifest["gradient_steps"])) and summary_path.exists()):
        print(json.dumps({"status": "already_complete", "arm": args.arm, "part": args.part}), flush=True)
        return
    check_weights(manifest["models"][args.arm])
    check_weights(manifest["teacher"])
    configure_model(args, train=True)
    current = runtime_identity(args)
    if current != manifest["runtime_identity"]:
        raise ValueError("fixed diagnostic runtime changed from preparation")
    spec = manifest["models"][args.arm]["paths"]
    # configure_model sets the frozen teacher to parent; override ONLY student paths.
    args.init_adapter, args.projector, args.qformer_adapter = spec["adapter"], spec["projector"], spec["qformer"]
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    dataset = training_dataset(args)
    indices = {r["question_id"]: i for i, r in enumerate(dataset.rows)}
    progress(folder, "loading_model", arm=args.arm)
    model = build_student(args, torch.device("cuda", 0), train=True)
    teacher = SharedFrozenTeacher(model, args)
    if args.arm == "parent":
        teacher.validate_initial_copy()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    counts = Counter()
    for name, p in named:
        counts[parameter_group(name)] += p.numel()
    if any(not counts[g] for g in ("language", "projector", "qformer")):
        raise ValueError("missing full trainable group for gradient diagnostics")
    teacher_named = [(n, p) for n, p in teacher.model.named_parameters() if
                     ".opd_teacher." in n or ("llm_model." not in n)]
    before, teacher_before = tensor_fingerprint(named), tensor_fingerprint(teacher_named)
    started = time.time()
    try:
        if want_t and len(trajectories) < len(cases):
            trajectories = run_trajectories(args, model, teacher, dataset, indices, cases, folder)
        if want_g and len(gradients) < len(manifest["gradient_steps"]):
            gradients = run_gradients(args, model, teacher, dataset, indices, draws, folder, manifest)
        if (tensor_fingerprint(named) != before or tensor_fingerprint(teacher_named) != teacher_before):
            raise RuntimeError("diagnostic unexpectedly modified student or teacher weights")
        complete = len(trajectories) == len(cases) and len(gradients) == len(manifest["gradient_steps"])
        value = {"completed": complete, "arm": args.arm, "trajectories": len(trajectories),
                 "gradient_batches": len(gradients), "optimizer_updates": 0,
                 "student_unchanged": True, "teacher_unchanged": True,
                 "student_tensor_sha256": before, "teacher_tensor_sha256": teacher_before,
                 "trainable_parameters": dict(counts), "elapsed_seconds_this_run": time.time() - started,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
        write_json_atomic(summary_path, value)
        progress(folder, "completed" if complete else "part_completed", len(trajectories), len(cases),
                 gradient_batches=len(gradients))
    except BaseException:
        progress(folder, "failed", traceback=traceback.format_exc())
        raise


def score(args):
    manifest = require_manifest(args)
    output = Path(args.output_dir)
    arms = [args.arm] if args.arm else list(ARMS)
    judge = None
    for arm in arms:
        folder = output / arm
        rows = expected_rows(folder / "trajectories.jsonl",
                             [r["case_id"] for r in load_jsonl(output / "cases.jsonl")], "case_id")
        if len(rows) != manifest["cases"]:
            raise ValueError("generate all fixed cases before scoring %s" % arm)
        score_folder = folder / "scored"
        score_folder.mkdir(parents=True, exist_ok=True)
        identity = {"predictions_sha256": file_sha256(folder / "trajectories.jsonl"),
                    "judge_protocol": JUDGE_PROTOCOL, "judge_model": args.judge_model,
                    "judge_device": "cpu", "scorer_sha256": file_sha256(ROOT / "scripts/evaluate_answers.py")}
        config_path = score_folder / "run_config.json"
        if config_path.exists() and read(config_path) != identity:
            raise ValueError("scoring input/protocol changed")
        write_json_atomic(config_path, identity)
        path = score_folder / "scored_predictions.jsonl"
        details = expected_rows(path, [r["case_id"] for r in rows], "case_id")
        done = {r["case_id"] for r in details}
        if len(details) < len(rows):
            if judge is None:
                torch.set_num_threads(args.cpu_threads)
                args.judge_device = "cpu"
                judge = Judge(args)
            for row in rows:
                if row["case_id"] in done:
                    continue
                exact, f1 = best_reference_scores(row["prediction"], row["references"])
                verdict = judge.score(dict(row, answers=row["references"]), row["prediction"])
                # Keep raw token records separately; scored rows are compact and joined by case_id.
                detail = {k: v for k, v in row.items() if k != "tokens"}
                detail.update(normalized_exact=exact, token_f1=f1,
                              lingo_judge_correct=verdict["correct"], lingo_judge_logit=verdict["logit"])
                append_row(path, detail)
                details.append(detail)
                if len(details) % 20 == 0:
                    progress(score_folder, "scoring", len(details), len(rows))
        summary = dict(summarize(details), judge_protocol=JUDGE_PROTOCOL)
        write_json_atomic(score_folder / "summary.json", summary)
        progress(score_folder, "completed", len(details), len(rows))


def images_markup(frames):
    chunks = []
    for i, path in enumerate(frames):
        if Path(path).is_file():
            from PIL import Image
            from io import BytesIO
            im = Image.open(path).convert("RGB")
            im.thumbnail((480, 300))
            buffer = BytesIO()
            im.save(buffer, format="JPEG", quality=75)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            chunks.append('<figure><img src="data:image/jpeg;base64,%s"><figcaption>frame %d</figcaption></figure>' % (b64, i))
        else:
            chunks.append('<span>Missing image: %s</span>' % html.escape(str(path)))
    return '<div class="frames">' + ''.join(chunks) + '</div>'


def html_document(body):
    return '<!doctype html><html><meta charset="utf-8"><title>OPD diagnostic review</title><style>' + \
        'body{max-width:1450px;margin:24px auto;font:15px system-ui;background:#fafafa}' + \
        'section{background:white;padding:20px;margin:20px 0;border:1px solid #ddd}.frames{display:flex;gap:6px}' + \
        'figure{margin:0;flex:1;min-width:0}img{width:100%}table{border-collapse:collapse;width:100%}' + \
        'td,th{border:1px solid #ddd;padding:5px;vertical-align:top}pre{white-space:pre-wrap}' + \
        '</style><body>' + body + '</body></html>'


def write_review_html(path, cases, arms, ids, output):
    body = '<h1>固定训练轨迹：教师方向、事实位置与自由回答</h1><p>这些是诊断样本，不是盲测。' + \
           '教师新回答正确不等于原始错误前缀上的分布正确；log-prob/JSD 不是事实标签。' + \
           '事实位置需明确审阅后填写 review_annotations.template.json 的副本。</p>'
    for row in cases:
        if row["case_id"] not in ids:
            continue
        body += '<section><h2>%s · %s</h2>' % (html.escape(row["case_id"]), review_category(row["question"]))
        body += images_markup(row["frames"])
        for label, value in (("问题", row["question"]), ("原参考", row["answers"]),
                             ("固定学生草稿", row["student_answer"]),
                             ("原 reference_only 教师纠正（仅门控）", row["teacher_correction"]),
                             ("cohort / gate", [row["cohort"], row["gate_reason"]])):
            body += '<p><strong>%s</strong>：%s</p>' % (label, html.escape(str(value)))
        for arm in ARMS:
            value = arms.get(arm, {}).get(row["case_id"])
            if value:
                body += '<p><strong>%s 自由回答</strong>：%s · JSD %.5f · CE %.5f</p>' % (
                    arm, html.escape(value["prediction"]), value["fixed_trajectory_jsd"], value["reference_ce"])
        if any(arms.get(a, {}).get(row["case_id"]) for a in ARMS):
            body += '<details><summary>逐 token 概率与候选词（位置从 0 开始）</summary><table><tr>' + \
                    '<th>位置/原 token</th><th>Teacher ref-only</th><th>parent</th><th>CE</th><th>OPD</th></tr>'
            values = next(arms[a][row["case_id"]] for a in ARMS if row["case_id"] in arms.get(a, {}))
            for i, token in enumerate(values["tokens"]):
                body += '<tr><td>%d · %s</td><td>logp %.4f<br>%s</td>' % (
                    i, html.escape(token["token"]), token["feedback_teacher_log_prob"],
                    html.escape(str(token["feedback_teacher_top"])))
                for arm in ARMS:
                    item = arms.get(arm, {}).get(row["case_id"])
                    if item:
                        t = item["tokens"][i]
                        body += '<td>logp %.4f · JSD %.5f<br>%s</td>' % (
                            t["student_log_prob"], t["jsd"], html.escape(str(t["student_top"])))
                    else:
                        body += '<td>pending</td>'
                body += '</tr>'
            body += '</table></details>'
        body += '</section>'
    path.write_text(html_document(body), encoding="utf-8")


def write_dev_html(path, rows, ids):
    body = '<h1>CE → OPD 开发集修复/退化</h1><p>Judge 翻转不等于事实翻转。下面为按问题类型分层选取的40例；' + \
           '全部变化保存在 dev_changes.jsonl。没有改写答案或重新训练。</p>'
    for row in rows:
        if row["question_id"] not in ids:
            continue
        body += '<section><h2>%s · %s</h2>' % (row["outcome"], html.escape(row["question_id"]))
        body += images_markup(row["frames"])
        body += '<p>%s</p><p>参考：%s</p>' % (html.escape(row["question"]), html.escape(str(row["answers"])))
        for arm, value in row["models"].items():
            body += '<p>%s · Judge %s（%.4f）：%s</p>' % (
                arm, value["lingo_judge_correct"], value["lingo_judge_logit"], html.escape(value["prediction"]))
        body += '</section>'
    path.write_text(html_document(body), encoding="utf-8")


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def report(args):
    manifest = require_manifest(args)
    output = Path(args.output_dir)
    cases = load_jsonl(output / "cases.jsonl")
    arms, scored, gradients, summaries = {}, {}, {}, {}
    complete = True
    for arm in ARMS:
        folder = output / arm
        trajectories = expected_rows(folder / "trajectories.jsonl", [r["case_id"] for r in cases], "case_id")
        arms[arm] = {r["case_id"]: r for r in trajectories}
        scored[arm] = expected_rows(folder / "scored/scored_predictions.jsonl", [r["case_id"] for r in cases], "case_id")
        gradients[arm] = expected_rows(folder / "gradients.jsonl", manifest["gradient_steps"], "effective_step")
        summaries[arm] = read(folder / "worker_summary.json") if (folder / "worker_summary.json").exists() else None
        complete &= (len(trajectories) == len(cases) and len(scored[arm]) == len(cases) and
                     len(gradients[arm]) == len(manifest["gradient_steps"]) and
                     bool(summaries[arm] and summaries[arm].get("completed")))
        if trajectories:
            config = read(folder / "run_config.json")
            if (config["manifest_sha256"] != file_sha256(output / "manifest.json") or
                    config["student"] != manifest["models"][arm] or config["teacher"] != manifest["teacher"]):
                raise ValueError("report combines unrelated student/teacher identities")
        if scored[arm]:
            config = read(folder / "scored/run_config.json")
            if (config["predictions_sha256"] != file_sha256(folder / "trajectories.jsonl") or
                    config["judge_protocol"] != JUDGE_PROTOCOL):
                raise ValueError("report scores belong to different predictions")
    facts, factual_status = {}, "not_reviewed"
    if args.annotations:
        annotations = validate_annotations(read(args.annotations), cases)
        reviewed = [a for a in annotations if a.get("reviewed")]
        factual_status = "reviewed_%d_cases" % len(reviewed)
        for arm in ARMS:
            buckets = defaultdict(list)
            for annotation in reviewed:
                if annotation["reference_consistent"] is not True:
                    continue
                value = arms[arm].get(annotation["case_id"])
                if not value:
                    continue
                for span in annotation.get("spans", []):
                    for pos in span["positions"]:
                        buckets[span["label"]].append(value["tokens"][pos])
            facts[arm] = {label: {"tokens": len(ts), "jsd_mean": mean(t["jsd"] for t in ts),
                "student_sampled_logp_mean": mean(t["student_log_prob"] for t in ts),
                "teacher_sampled_logp_mean": mean(t["feedback_teacher_log_prob"] for t in ts),
                "teacher_minus_unconditioned_logp_mean": mean(t["feedback_minus_plain"] for t in ts),
                "teacher_increased_sampled_token_count": sum(t["feedback_minus_plain"] > 0 for t in ts)}
                for label, ts in buckets.items()}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
              "completed": bool(complete), "optimizer_updates": 0, "teacher_mode": "reference_only",
              "teacher_identity": manifest["teacher"]["identity"], "case_cohorts": manifest["case_cohorts"],
              "training_log": manifest["training_log"], "models": {}, "fixed_jsd_comparisons": {},
              "free_generation_comparisons": {}, "gradient_summary": {}, "worker_summaries": summaries,
              "dev_outcomes": manifest["dev_outcomes"], "factual_review_status": factual_status,
              "reviewed_factual_tokens": facts,
              "cautions": ["Teacher correction was a gate; the target is its distribution on the cached student prefix.",
                  "Selected errors + correct controls are biased TRAIN diagnostics, not new benchmark accuracy.",
                  "Lower whole-answer JSD or higher sampled-token probability does not prove corrected facts.",
                  "Gradient cosine/norm are indicators, not AdamW updates or proof of the cause of dev regression.",
                  "Cached gates stay fixed across snapshots; this is not new on-policy training.",
                  "Dev fixes/regressions reuse historical scores; the official 500 is not a new blind test."]}
    teacher_hashes = {s["teacher_tensor_sha256"] for s in summaries.values() if s}
    if len(teacher_hashes) > 1:
        raise ValueError("actual teacher tensor identity differs across arms")
    for arm in ARMS:
        result["models"][arm] = {}
        for cohort in manifest["case_cohorts"]:
            values = [r for r in arms[arm].values() if r["cohort"] == cohort]
            scores = [r for r in scored[arm] if r["cohort"] == cohort]
            result["models"][arm][cohort] = {"records": len(values),
                "reference_ce_mean": mean(r["reference_ce"] for r in values),
                "fixed_trajectory_jsd_mean": mean(r["fixed_trajectory_jsd"] for r in values),
                "plain_teacher_jsd_mean": mean(mean(t["plain_teacher_jsd"] for t in r["tokens"]) for r in values),
                "free_generation": summarize(scores) if scores else None}
        result["gradient_summary"][arm] = {group: {
            "batches": len(gradients[arm]),
            "ce_norm_mean": mean(r["gradients"][group]["ce_l2"] for r in gradients[arm]),
            "weighted_jsd_norm_mean": mean(r["gradients"][group]["weighted_jsd_l2"] for r in gradients[arm]),
            "cosine_mean": mean(r["gradients"][group]["cosine"] for r in gradients[arm]
                                if r["gradients"][group]["cosine"] is not None),
            "negative_cosine_batches": sum(r["gradients"][group]["cosine"] is not None and
                                            r["gradients"][group]["cosine"] < 0 for r in gradients[arm]),
            "jsd_to_ce_norm_ratio_mean": mean(r["gradients"][group]["jsd_to_ce_norm_ratio"] for r in gradients[arm]
                                      if r["gradients"][group]["jsd_to_ce_norm_ratio"] is not None)}
            for group in ("language", "projector", "qformer", "all")}
    for a, b in (("parent", "ce"), ("parent", "opd"), ("ce", "opd")):
        common = set(arms[a]) & set(arms[b])
        values = {k: arms[b][k]["fixed_trajectory_jsd"] - arms[a][k]["fixed_trajectory_jsd"] for k in common}
        result["fixed_jsd_comparisons"][a + "_to_" + b] = {
            "records": len(common), "mean_delta": mean(values.values()),
            "decreased_cases": sum(v < 0 for v in values.values()),
            "eligible_error_mean_delta": mean(values[r["case_id"]] for r in cases
                if r["eligible"] and r["case_id"] in common)}
        if len(scored[a]) == len(cases) and len(scored[b]) == len(cases):
            ia, ib = {r["case_id"]: r for r in scored[a]}, {r["case_id"]: r for r in scored[b]}
            result["free_generation_comparisons"][a + "_to_" + b] = paired_metrics(
                [ia[r["case_id"]] for r in cases], [ib[r["case_id"]] for r in cases])
    joined = []
    for row in cases:
        joined.append({"case_id": row["case_id"], "cohort": row["cohort"],
                       "teacher_gate_correct": bool(row["teacher_judge"] and row["teacher_judge"]["correct"]),
                       "models": {arm: {"fixed_jsd": arms[arm][row["case_id"]]["fixed_trajectory_jsd"],
                           "free_judge_correct": next((r["lingo_judge_correct"] for r in scored[arm]
                                                      if r["case_id"] == row["case_id"]), None)}
                                  for arm in ARMS if row["case_id"] in arms[arm]}})
    write_rows(output / "joined_cases.jsonl", joined)
    write_json_atomic(output / "report.json", result)
    write_review_html(output / "review.html", cases, arms, manifest["review_case_ids"], output)
    text = "# 固定教师 OPD 诊断\n\n" + ("全部计算完成。" if complete else "计算尚未全部完成，以下为中间报告。")
    text += "事实 token 标注状态：`%s`。未审阅时不能宣称教师纠错方向已验证。\n\n" % factual_status
    text += "原始输入、原始参考、原始学生 token、固定教师及实际门控均已锁定；没有 optimizer.step。\n\n"
    text += "## 固定轨迹与同题自由回答\n\n"
    text += "| 模型 | 固定组 | N | 参考 CE | 固定轨迹 JSD | 无提示自由回答 Judge |\n|---|---|---:|---:|---:|---:|\n"
    for arm, cohorts in result["models"].items():
        for cohort, value in cohorts.items():
            if value["records"]:
                judge = value["free_generation"]
                text += "| %s | %s | %d | %.5f | %.5f | %s |\n" % (
                    arm, cohort, value["records"], value["reference_ce_mean"], value["fixed_trajectory_jsd_mean"],
                    "%.2f%%" % (100 * judge["lingo_judge_accuracy"]) if judge else "待评分")
    text += "\n这些是按原回答正误选取的训练侧样本，不能作为新的准确率 benchmark。\n\n"
    text += "## 按有效 batch 累积的梯度\n\n"
    text += "| 模型 | 参数组 | batch 数 | CE 梯度范数均值 | 加权 JSD 范数均值 | JSD/CE 范数比均值 | 余弦均值 | 负余弦 batch 数 |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
    for arm, groups in result["gradient_summary"].items():
        for group, value in groups.items():
            if value["batches"]:
                display = lambda k: "—" if value[k] is None else "%.5f" % value[k]
                text += "| %s | %s | %d | %s | %s | %s | %s | %d |\n" % (
                    arm, group, value["batches"], display("ce_norm_mean"), display("weighted_jsd_norm_mean"),
                    display("jsd_to_ce_norm_ratio_mean"), display("cosine_mean"), value["negative_cosine_batches"])
    text += "\n这是在起始／终点权重上回放历史 batch 的梯度，不能当作各历史 step 当时的梯度或 AdamW 实际更新。\n\n"
    text += "## 读取方法\n\n"
    text += "1. 先比较固定错误轨迹中 OPD 与 CE 的 JSD；动态训练均值不能替代它。\n"
    text += "2. 再看三个参数组的有效批次梯度比例与夹角，区分信号弱、冲突和门控范围。\n"
    text += "3. 对照同题自由回答及 `review.html` 的事实位置；使用 `dev_review.html` 检查历史修复/退化。\n\n"
    text += "教师错误→检查监督；教师正确但未对齐→检查优化；分布已对齐但事实未改→检查蒸馏位置；"
    text += "训练事实改善而开发集未改善→检查泛化。以上是排查线索，不是自动归因。\n\n"
    text += "```json\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n```\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    progress(output, "completed" if complete else "partial_report", int(complete), 1,
             factual_review_status=factual_status)
    print(json.dumps({"completed": bool(complete), "json": str(output / "report.json"),
                      "markdown": str(output / "report.md"), "factual_review": factual_status}, indent=2))


def status(args):
    output = Path(args.output_dir)
    if not (output / "manifest.json").exists():
        value = read(output / "progress.json") if (output / "progress.json").exists() else {"status": "not_prepared"}
        value["output_dir"] = str(output)
        if value.get("updated_unix_seconds"):
            value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
        print(json.dumps(value, indent=2))
        return
    manifest = read(output / "manifest.json")
    result = {"output_dir": str(output), "cases_per_model": manifest["cases"],
              "gradient_batches_per_model": len(manifest["gradient_steps"]), "models": {}}
    for arm in ARMS:
        folder = output / arm
        p = read(folder / "progress.json") if (folder / "progress.json").exists() else {"status": "not_started"}
        if p.get("pid"):
            try:
                os.kill(p["pid"], 0)
                p["process_alive"] = True
            except (ProcessLookupError, PermissionError):
                p["process_alive"] = False
            p["seconds_since_update"] = round(time.time() - p["updated_unix_seconds"], 1)
        if p["status"] in ("completed", "part_completed"):
            p["process_alive"] = False
        p.update(trajectories=sum(1 for _ in (folder / "trajectories.jsonl").open()) if
                 (folder / "trajectories.jsonl").exists() else 0,
                 gradient_batches=sum(1 for _ in (folder / "gradients.jsonl").open()) if
                 (folder / "gradients.jsonl").exists() else 0,
                 scored=sum(1 for _ in (folder / "scored/scored_predictions.jsonl").open()) if
                 (folder / "scored/scored_predictions.jsonl").exists() else 0)
        score_progress = folder / "scored/progress.json"
        if score_progress.exists():
            sp = read(score_progress)
            p["scoring"] = {k: sp[k] for k in ("status", "completed", "total")}
        if p.get("status") not in ("failed", "completed", "part_completed", "not_started") and p.get("process_alive") is False:
            p["last_stage"] = p["status"]
            p["status"] = "interrupted"
        result["models"][arm] = p
    result["report_ready"] = (output / "report.json").exists()
    result["completed"] = bool(read(output / "report.json")["completed"]) if result["report_ready"] else False
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(p)
    p.set_defaults(teacher_mode="reference_only")
    p.add_argument("--action", choices=("prepare", "infer", "score", "report", "status"), required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--part", choices=("all", "trajectories", "gradients"), default="all")
    p.add_argument("--data-dir", default=str(ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2"))
    p.add_argument("--source-probe-dir", default=str(ROOT / "outputs/feedback_opd_probe_v1"))
    p.add_argument("--comparison-dir", default=str(ROOT / "outputs/feedback_teacher_compare_v2"))
    p.add_argument("--train-dir", default=str(ROOT / "outputs/mpp_feedback_opd_40k_2epoch_v2"))
    p.add_argument("--control-dir", default=str(ROOT / "outputs/mpp_feedback_sft_control_40k_2epoch_v2"))
    p.add_argument("--eval-dir", default=str(ROOT / "outputs/feedback_opd_eval_v2"))
    p.add_argument("--controls", type=int, default=100)
    p.add_argument("--error-cases", type=int, default=0, help="0 retains all 200; a smaller value is only for independent smoke outputs")
    p.add_argument("--gradient-steps", type=int, nargs="+", default=[1, 500, 1000, 2000, 2501, 3000, 4000, 5000])
    p.add_argument("--review-cases", type=int, default=40)
    p.add_argument("--annotations", default="", help="optional reviewed copy of the template; never a training input")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.teacher_mode != "reference_only" or args.language_lora_scope != "qkv":
        raise ValueError("this diagnostic holds the original QKV student/teacher protocol fixed")
    if args.action == "infer" and not args.arm:
        raise ValueError("infer requires --arm parent|ce|opd")
    if not 0 <= args.error_cases <= 200 or args.controls < 0 or args.review_cases < 0:
        raise ValueError("invalid diagnostic sample count")
    try:
        {"prepare": prepare, "infer": infer, "score": score, "report": report, "status": status}[args.action](args)
    except BaseException:
        if args.action != "status":
            folder = Path(args.output_dir) / (args.arm or "")
            progress(folder, "failed", traceback=traceback.format_exc())
        raise
