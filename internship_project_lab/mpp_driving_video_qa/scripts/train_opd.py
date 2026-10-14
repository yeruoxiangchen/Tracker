#!/usr/bin/env python3
"""Same-tokenizer OPD with fixed teacher, supervised replay and resumable state."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.utils.data import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import TASK_INSTRUCTIONS, file_sha256
from mpp_video_qa.data import CachedQADataset, cache_file
from mpp_video_qa.model import CachedVideoQA, build_text_model, save_projector
from mpp_video_qa.observations import teacher_prompt, ANSWER_HINT_PROTOCOL
from mpp_video_qa.opd import (SharedFrozenTeacher, sample_completions, completion_logits,
                              distillation_loss, average_gradients, disable_dropout)
from mpp_video_qa.visual_bridge import add_bridge_arguments, save_bridge
from scripts.generate_answers import write_json_atomic, directory_fingerprint
from scripts.train_sft import seed_all, rng_state, restore_rng_state, gpu_memory


def parse_args():
    p = argparse.ArgumentParser()
    add_bridge_arguments(p)
    for name in ("data", "cache-dir", "teacher-cache-dir", "output-dir", "upstream-root",
                 "checkpoint-root", "init-adapter", "projector", "teacher-adapter", "teacher-projector"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--teacher-qformer-adapter", default="")
    p.add_argument("--teacher-view", choices=("full", "full_crop"), default="full_crop")
    p.add_argument("--teacher-context", choices=("view", "answer_hint"), default="view")
    p.add_argument("--training-mode", choices=("opd", "sft_control"), default="opd")
    p.add_argument("--feature-stage", choices=("eva", "qformer"), default="eva")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--projector-lr", type=float, default=2e-6)
    p.add_argument("--sft-weight", type=float, default=0.2)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--objective", choices=("jsd", "forward_kl"), default="jsd")
    p.add_argument("--chunk-tokens", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-txt-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", default="")
    p.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    p.set_defaults(dtype="bf16", train_projector=True, gradient_checkpointing=True,
                   lora_r=16, lora_alpha=32, lora_dropout=0.0, lora_target_modules=["c_attn"])
    return p.parse_args()


def save(output, model, optimizer, state, rank, world, final=False):
    rngs = [None] * world
    if world > 1:
        torch.distributed.all_gather_object(rngs, rng_state())
    else:
        rngs[0] = rng_state()
    if rank == 0:
        destination = output / ("final" if final else "checkpoint-%06d" % state["optimizer_steps"])
        temporary = destination.with_name(destination.name + ".incomplete")
        temporary.mkdir(parents=True, exist_ok=False)
        model.backbone.llm_model.save_pretrained(temporary / "adapter", selected_adapters=["default"])
        save_projector(model.backbone, temporary / "projector.pt")
        save_bridge(model.backbone, temporary / "qformer_adapter.pt")
        torch.save(dict(state, optimizer=optimizer.state_dict(), rng_states=rngs), temporary / "training_state.pt")
        write_json_atomic(temporary / "metadata.json", state)
        temporary.replace(destination)
    if world > 1:
        torch.distributed.barrier()


def main():
    args = parse_args()
    if args.task_instruction_version != "legacy_v1":
        raise ValueError("teacher selection currently uses legacy_v1; keep the OPD task instruction identical")
    if args.grad_accumulation < 1 or args.sft_weight <= 0 or args.distill_weight <= 0:
        raise ValueError("positive accumulation, SFT replay and distillation weights are required")
    if args.teacher_context == "answer_hint" and args.teacher_view != "full":
        raise ValueError("answer-conditioned teacher requires the original full frames")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=12))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("completed run exists; choose a new output directory")
    if (output / "run_config.json").exists() and not args.resume_from:
        raise ValueError("run exists; use --resume-from or a new directory")
    seed_all(args.seed)
    dataset = CachedQADataset(args.data, args.cache_dir, args.feature_stage)
    same_cache = Path(args.cache_dir).resolve() == Path(args.teacher_cache_dir).resolve()
    teacher_data = dataset if same_cache else CachedQADataset(args.data, args.teacher_cache_dir, args.feature_stage)
    if not dataset.rows or any(row.get("opd_training_source") != "train" for row in dataset.rows):
        raise ValueError("use prepare_opd_data.py to select train-only teacher supervision first")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True)
    total_steps = math.ceil(len(sampler) / args.grad_accumulation) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    identity = {
        "data_sha256": file_sha256(args.data), "world_size": world, "total_steps": total_steps,
        "teacher_adapter_sha256": directory_fingerprint(args.teacher_adapter),
        "teacher_projector_sha256": file_sha256(args.teacher_projector),
        "teacher_qformer_sha256": file_sha256(args.teacher_qformer_adapter) if args.teacher_qformer_adapter else None,
        "bridge_init_sha256": file_sha256(args.bridge_init) if args.bridge_init else None,
        "student_initial_adapter_sha256": directory_fingerprint(args.init_adapter),
        "student_initial_projector_sha256": file_sha256(args.projector),
        "student_initial_qformer_sha256": file_sha256(args.qformer_adapter) if args.qformer_adapter else None,
        "task_instruction_version": args.task_instruction_version,
        "training_mode": args.training_mode, "teacher_context": args.teacher_context,
        "answer_hint_protocol": ANSWER_HINT_PROTOCOL if args.teacher_context == "answer_hint" else None,
        "lr": args.lr, "projector_lr": args.projector_lr, "qformer_lr": args.qformer_lr,
        "seed": args.seed, "max_txt_len": args.max_txt_len, "chunk_tokens": args.chunk_tokens,
        "teacher_view": args.teacher_view, "objective": args.objective, "feature_stage": args.feature_stage,
        "train_qformer": args.train_qformer, "grad_accumulation": args.grad_accumulation,
        "max_new_tokens": args.max_new_tokens, "sft_weight": args.sft_weight,
        "distill_weight": args.distill_weight}
    selection_path = Path(args.data).with_suffix(".manifest.json")
    selection = json.loads(selection_path.read_text())
    identity["selection_manifest_sha256"] = file_sha256(selection_path)
    if selection["selected_sha256"] != identity["data_sha256"]:
        raise ValueError("OPD selected data differs from its selection manifest")
    scored_teacher = selection["generation_identities"]["teacher"]
    expected_teacher = {
        "adapter": identity["teacher_adapter_sha256"],
        "projector": identity["teacher_projector_sha256"],
        "qformer": identity["teacher_qformer_sha256"],
        "view": args.teacher_view, "feature_stage": args.feature_stage,
        "bridge_init": identity["bridge_init_sha256"]}
    if any(scored_teacher.get(key) != value for key, value in expected_teacher.items()):
        raise ValueError("OPD teacher differs from the teacher used for training-data selection")
    if scored_teacher.get("backend") != "mpp":
        raise ValueError("token OPD requires the same MPP tokenizer, not an external VLM teacher")
    expected_student = {
        "adapter": identity["student_initial_adapter_sha256"],
        "projector": identity["student_initial_projector_sha256"],
        "qformer": identity["student_initial_qformer_sha256"],
        "view": "full", "feature_stage": args.feature_stage,
        "bridge_init": identity["bridge_init_sha256"]}
    scored_student = selection["generation_identities"]["student"]
    if any(scored_student.get(key) != value for key, value in expected_student.items()):
        raise ValueError("OPD student initialization differs from training-data selection")
    if args.teacher_context == "answer_hint":
        if (scored_teacher.get("answer_hint_protocol") != ANSWER_HINT_PROTOCOL or
                scored_teacher.get("teacher_context") != "answer_hint" or
                scored_student.get("teacher_context") != "view" or selection.get("num_samples", 0) < 2):
            raise ValueError("answer-conditioned OPD requires the multi-sample train-only selection")
        for scored in (scored_teacher, scored_student):
            if scored.get("task_instruction_version") != args.task_instruction_version:
                raise ValueError("selection and training task instructions differ")
    resume = None
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        resume = torch.load(checkpoint / "training_state.pt", map_location="cpu")
        if resume["identity"] != identity:
            raise ValueError("OPD resume configuration/data/teacher differs")
        args.init_adapter, args.projector = str(checkpoint / "adapter"), str(checkpoint / "projector.pt")
        if args.feature_stage == "eva":
            args.qformer_adapter = str(checkpoint / "qformer_adapter.pt")
        if rank == 0:
            for path in output.glob("failure_rank_*.json"):
                path.rename(path.with_suffix(".before_resume_%d" % int(time.time())))
    for loading_rank in range(world):
        if rank == 0:
            write_json_atomic(output / "progress.json", {
                "status": "loading_model", "loading_rank": loading_rank, "world_size": world,
                "updated_unix_seconds": time.time()})
        if rank == loading_rank:
            model = CachedVideoQA(build_text_model(args, train=True, adapter=args.init_adapter,
                                                   projector=args.projector),
                                  task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version],
                                  feature_stage=args.feature_stage).to(device)
            teacher = SharedFrozenTeacher(model, args) if args.training_mode == "opd" else None
            disable_dropout(model)
        if world > 1:
            torch.distributed.barrier()
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    # Broadcast fresh adapter initializations as well as resumed trainable tensors.
    if world > 1:
        for _, parameter in named:
            torch.distributed.broadcast(parameter.data, src=0)
    parameters = [param for _, param in named]
    groups = []
    for group, marker, lr in (("qformer", "visual_bridge.", args.qformer_lr),
                              ("projector", "llm_proj.", args.projector_lr),
                              ("language", "llm_model.", args.lr)):
        selected = [param for name, param in named if marker in name]
        if selected:
            groups.append({"params": selected, "lr": lr, "group_name": group})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    step, micro, start_epoch, start_batch = 0, 0, 0, 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        step, micro = resume["optimizer_steps"], resume["micro_steps"]
        start_epoch, start_batch = resume["next_epoch"], resume["next_batch"]
        restore_rng_state(resume["rng_states"][rank])
    else:
        seed_all(args.seed + rank)
    tracked_name, tracked = next((n, p) for n, p in named if "lora_B" in n)
    initial = tracked.detach().float().cpu().clone()
    if rank == 0:
        write_json_atomic(output / "run_config.json", dict(vars(args), **{
            "records": len(dataset), "world_size": world, "total_steps": total_steps,
            "identity": identity, "shared_frozen_qwen": True,
            "rollout_distribution": "temperature=1, top_p=1, top_k=0, dropout=0"}))
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    counters = torch.zeros(5, device=device, dtype=torch.float64)
    state = None
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        model.train()
        for batch_index in range(start_batch if epoch == start_epoch else 0, len(indices)):
            if step >= total_steps:
                break
            index = indices[batch_index]
            record = dataset[index]
            row = record["row"]
            features = record["features"].unsqueeze(0).to(device)
            student_payload = torch.load(cache_file(args.cache_dir, row["segment_id"]), map_location="cpu")
            if student_payload.get("view", "full") != "full" or student_payload["frame_paths"] != row["frames"]:
                raise ValueError("student cache must contain the original full frames")
            del student_payload
            teacher_features = features
            if not same_cache and teacher is not None:
                payload = torch.load(cache_file(args.teacher_cache_dir, row["segment_id"]), map_location="cpu")
                if payload.get("view", "full") != args.teacher_view or payload["frame_paths"] != row["frames"]:
                    raise ValueError("teacher cache is not the selected view of the same frames")
                del payload
                teacher_features = teacher_data[index]["features"].unsqueeze(0).to(device)
            elif same_cache and args.teacher_view != "full":
                raise ValueError("shared student cache cannot supply a cropped teacher view")
            divisor = min(args.grad_accumulation,
                          len(indices) - (batch_index // args.grad_accumulation) * args.grad_accumulation)
            completion, finished = None, False
            if teacher is not None:
                completion, finished = sample_completions(model, features, row["question"], args.max_new_tokens)
            divergence_value = 0.0
            if finished:
                prompt = teacher_prompt(row["question"], len(row["frames"]), row["answers"],
                                        args.teacher_context, args.teacher_view,
                                        TASK_INSTRUCTIONS[args.task_instruction_version])
                with teacher.active() as frozen:
                    targets = completion_logits(frozen, teacher_features, row["question"],
                                                completion, prompt).detach().clone()
                logits = completion_logits(model, features, row["question"], completion)
                divergence = distillation_loss(logits, targets, args.objective, args.chunk_tokens)
                if not torch.isfinite(divergence):
                    raise RuntimeError("nonfinite OPD divergence")
                divergence_value = float(divergence.detach())
                (args.distill_weight * divergence / divisor).backward()
                del divergence, logits, targets
            # Reference-prefix replay is a separate forward/backward, releasing OPD activations first.
            replay = model(features, [row["question"]], [record["answer"]]).loss
            if not torch.isfinite(replay):
                raise RuntimeError("nonfinite supervised replay loss")
            replay_value = float(replay.detach())
            replay_weight = args.sft_weight if teacher is not None else 1.0
            (replay_weight * replay / divisor).backward()
            del replay
            micro += 1
            counters += counters.new_tensor([divergence_value, replay_value, 1, int(finished),
                                              completion.numel() if completion is not None else 0])
            should_step = (batch_index + 1) % args.grad_accumulation == 0 or batch_index + 1 == len(indices)
            if should_step:
                average_gradients(parameters, world)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if world > 1:
                    torch.distributed.all_reduce(counters)
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch == len(indices):
                    next_epoch, next_batch = epoch + 1, 0
                state = {"optimizer_steps": step, "micro_steps": micro, "next_epoch": next_epoch,
                         "next_batch": next_batch, "identity": identity}
                if rank == 0:
                    progress = {
                        "status": "running", "optimizer_step": step, "optimizer_steps_total": total_steps,
                        "progress_percent": 100 * step / total_steps,
                        "distill_loss_mean_finished": float(counters[0] / counters[3].clamp_min(1)),
                        "sft_replay_loss_mean": float(counters[1] / counters[2]),
                        "training_mode": args.training_mode, "micro_examples_in_step": int(counters[2]),
                        "rollouts_in_step": int(counters[2]) if teacher is not None else 0,
                        "finished_rollouts_in_step": int(counters[3]),
                        "truncated_rollouts_in_step": int(counters[2] - counters[3]) if teacher is not None else 0,
                        "completion_tokens_in_step": int(counters[4]), "gradient_norm": float(norm),
                        "elapsed_seconds": time.time() - started, "updated_unix_seconds": time.time(),
                        "gpu_memory": gpu_memory(device)}
                    write_json_atomic(output / "progress.json", progress)
                    with (output / "training_log.jsonl").open("a") as stream:
                        stream.write(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
                counters.zero_()
                if args.save_every and step % args.save_every == 0:
                    save(output, model, optimizer, state, rank, world)
        start_batch = 0
        if step >= total_steps:
            break
    if state is None:
        raise ValueError("no optimizer step executed; selected checkpoint may already be complete")
    save(output, model, optimizer, state, rank, world, final=True)
    if rank == 0:
        summary = {
            "completed": True, "optimizer_steps": step, "micro_steps": micro,
            "elapsed_seconds": time.time() - started, "objective": args.objective,
            "training_mode": args.training_mode, "teacher_context": args.teacher_context,
            "final_bundle": str((output / "final").resolve()), "tracked_parameter": tracked_name,
            "tracked_parameter_max_abs_delta": float((tracked.detach().float().cpu() - initial).abs().max()),
            "gpu_memory_rank0": gpu_memory(device)}
        write_json_atomic(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args = parse_args()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / ("failure_rank_%s.json" % os.environ.get("RANK", "0")),
                          {"status": "failed", "traceback": traceback.format_exc(),
                           "updated_unix_seconds": time.time(), "pid": os.getpid()})
        raise
