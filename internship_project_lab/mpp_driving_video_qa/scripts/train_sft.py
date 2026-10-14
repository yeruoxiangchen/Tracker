#!/usr/bin/env python3
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import timedelta
from itertools import islice
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import TASK_INSTRUCTIONS, file_sha256
from mpp_video_qa.data import CachedQADataset, qa_collate
from mpp_video_qa.model import (FEATURE_STAGE_PROJECTED, FEATURE_STAGE_QFORMER, FEATURE_STAGE_EVA,
                                FEATURE_STAGES, CachedVideoQA, build_text_model,
                                save_projector)
from mpp_video_qa.visual_bridge import add_bridge_arguments, save_bridge
from mpp_video_qa.region_views import add_region_arguments, region_runtime_identity


def parse_args():
    parser = argparse.ArgumentParser()
    add_bridge_arguments(parser)
    add_region_arguments(parser)
    parser.add_argument("--dev-region-views", default="")
    parser.add_argument("--answer-only-logits", action="store_true",
                        help="same causal CE with LM-head projection only on answer positions")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-mode", choices=("reference", "distillation"), default="reference",
                        help="train on original answers or filtered teacher answers with reference replay")
    parser.add_argument("--dev-data")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-txt-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["c_attn"])
    parser.add_argument("--feature-stage", choices=FEATURE_STAGES,
                        default=FEATURE_STAGE_PROJECTED)
    parser.add_argument("--train-projector", action="store_true",
                        help="jointly train the official MPP 768->4096 projection")
    parser.add_argument("--projector", default="",
                        help="optional projector.pt used as initialization")
    parser.add_argument("--projector-lr", type=float, default=None,
                        help="projector learning rate; defaults to --lr")
    parser.add_argument("--max-dev-records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer updates; 0 runs all epochs.")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a resumable checkpoint every N optimizer steps; 0 disables it.")
    parser.add_argument("--resume-from", default="",
                        help="checkpoint-N directory containing adapter/ and training_state.pt")
    parser.add_argument("--init-adapter", default="",
                        help="initialize trainable LoRA weights; start a fresh optimizer and schedule")
    parser.add_argument("--parent-run", default="",
                        help="completed training stage supplying all three initialization weights")
    parser.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS),
                        default="legacy_v1")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_true",
                        help="Recompute Qwen blocks during backward to reduce activation VRAM.")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()
    if args.region_views and (args.feature_stage != FEATURE_STAGE_EVA or args.batch_size != 1):
        parser.error("variable automatic views require --feature-stage eva --batch-size 1")
    if args.region_views and args.dev_data and not args.dev_region_views:
        parser.error("region training with dev data requires separate --dev-region-views")
    if args.qformer_mode == "full":
        args.train_qformer = True
        if args.precision_protocol != "fp32_trainables_v1":
            parser.error("full Q-Former training requires --precision-protocol fp32_trainables_v1")
    elif args.qformer_mode == "frozen":
        args.train_qformer = False
    return args


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def retain_initialization_provenance(config, resume_from):
    """A resumed weight bundle is not the original experiment initialization."""
    if not resume_from:
        return
    source = Path(resume_from).parent / "run_config.json"
    if source.is_file():
        previous = json.loads(source.read_text(encoding="utf-8"))
        for key in ("initial_adapter_sha256", "initial_projector_sha256", "initial_qformer_sha256"):
            if key in previous:
                config[key] = previous[key]
        config["initialization_provenance_from"] = str(source.resolve())


def continuation_provenance(args):
    if not args.parent_run:
        return {}
    parent = Path(args.parent_run).resolve()
    if parent == Path(args.output_dir).resolve():
        raise ValueError("continuation requires a new output directory")
    summary = json.loads((parent / "summary.json").read_text())
    config = json.loads((parent / "run_config.json").read_text())
    if not summary.get("completed") or summary["optimizer_steps"] != config["total_steps"]:
        raise ValueError("parent training stage is not complete")
    if file_sha256(args.data) != config["data_sha256"]:
        raise ValueError("continuation must retain the parent training dataset")
    if args.resume_from:
        previous = json.loads((Path(args.resume_from).parent / "run_config.json").read_text())
        if previous.get("parent_run") != str(parent):
            raise ValueError("resume checkpoint belongs to a different continuation stage")
        keys = ("seed", "lr", "projector_lr", "qformer_lr", "warmup_ratio", "weight_decay")
        changed = [key for key in keys if hasattr(args, key) and previous.get(key) != getattr(args, key)]
        if changed:
            raise ValueError("continuation resume configuration changed: %s" % changed)
    else:
        from scripts.generate_answers import directory_fingerprint
        if (directory_fingerprint(args.init_adapter) != directory_fingerprint(parent / "final_adapter") or
            file_sha256(args.projector) != file_sha256(parent / "final_projector.pt") or
            file_sha256(args.qformer_adapter) != file_sha256(parent / "final_qformer_adapter.pt")):
            raise ValueError("continuation must load the complete parent weight bundle")
    return {"parent_run": str(parent), "parent_optimizer_steps": summary["optimizer_steps"],
            "continuation_kind": "weights_only_new_optimizer_and_schedule"}


class OffsetSampler(Sampler):
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

    def __iter__(self):
        return islice(iter(self.base), self.offset, None)

    def __len__(self):
        return max(0, len(self.base) - self.offset)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state()}


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(output, raw_model, optimizer, scheduler, state, rank, world, distributed):
    states = [None] * world
    local_rng = rng_state()
    if distributed:
        torch.distributed.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    destination = output / ("checkpoint-%06d" % state["global_step"])
    if rank == 0:
        if destination.exists():
            raise RuntimeError("checkpoint already exists: %s" % destination)
        temporary = output / (destination.name + ".incomplete")
        temporary.mkdir(parents=True, exist_ok=False)
        raw_model.backbone.llm_model.save_pretrained(temporary / "adapter")
        if hasattr(raw_model.backbone, "llm_proj"):
            save_projector(raw_model.backbone, temporary / "projector.pt")
        save_bridge(raw_model.backbone, temporary / "qformer_adapter.pt")
        payload = dict(state)
        payload.update({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng_states": states})
        torch.save(payload, temporary / "training_state.pt")
        (temporary / "metadata.json").write_text(
            json.dumps({key: value for key, value in state.items()
                        if key not in ("optimizer", "scheduler", "rng_states")}, indent=2),
            encoding="utf-8")
        temporary.replace(destination)
        print(json.dumps({"status": "checkpoint_saved", "checkpoint": str(destination),
                          "optimizer_step": state["global_step"]}), flush=True)
    if distributed:
        torch.distributed.barrier()


def gpu_memory(device):
    """Return allocator and device-level memory without changing model state."""
    result = {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    try:
        free, total = torch.cuda.mem_get_info(device)
        result.update({"device_free_gib": free / 2**30,
                       "device_total_gib": total / 2**30})
    except Exception as error:
        result["mem_get_info_error"] = repr(error)
    return result


def record_gpu_memory(output, rank, device, stage, synchronize=True):
    """Persist per-rank stage memory so an asynchronous CUDA failure is attributable."""
    if synchronize:
        torch.cuda.synchronize(device)
    record = {"status": "gpu_memory", "stage": stage, "rank": rank,
              "updated_unix_seconds": time.time(), **gpu_memory(device)}
    with (output / ("memory_rank_%d.jsonl" % rank)).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


@torch.no_grad()
def dev_loss(model, dataset, device, limit, answer_only_logits=False):
    if dataset is None or limit <= 0:
        return None
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=qa_collate)
    values = []
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        features = batch["features"].to(device)
        questions = [row["question"] for row in batch["rows"]]
        values.append(float(model(features, questions, batch["answers"],
                                  prompts=batch.get("prompts"), answer_only_logits=answer_only_logits).loss))
    model.train()
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    if args.resume_from and (args.init_adapter or args.projector or args.qformer_adapter):
        raise ValueError("use either --resume-from or weight initialization, not both")
    if args.init_adapter and args.train_projector and not args.projector:
        raise ValueError("joint continuation requires both --init-adapter and --projector")
    parent_info = continuation_provenance(args)
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0)); distributed = world > 1
    torch.cuda.set_device(local_rank)
    if distributed:
        # Model construction is serialized across ranks to avoid four workers
        # reading the large checkpoint simultaneously.  On a busy shared disk
        # one rank can legitimately take longer than NCCL's default timeout.
        torch.distributed.init_process_group(
            "nccl", timeout=timedelta(hours=12))
    seed_all(args.seed + rank)
    device = torch.device("cuda", local_rank)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        torch.distributed.barrier()
    existing_config = output / "run_config.json"
    if existing_config.exists() and not args.resume_from:
        raise RuntimeError("output directory already contains a run; use --resume-from or a new path")
    if rank == 0 and args.resume_from:
        for failure_path in output.glob("failure_rank_*.json"):
            archived = failure_path.with_suffix(
                failure_path.suffix + ".before_resume_%d" % int(time.time()))
            failure_path.replace(archived)
    if distributed:
        torch.distributed.barrier()

    if args.train_projector and args.feature_stage == FEATURE_STAGE_PROJECTED:
        raise ValueError("--train-projector requires qformer or eva features")
    view_identity = region_runtime_identity(args)
    if args.dev_region_views:
        view_identity["dev_region_views_sha256"] = file_sha256(args.dev_region_views)
    observation_args = dict(region_cache_dir=args.region_cache_dir, region_condition=args.region_condition,
                            task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version])
    train_set = CachedQADataset(args.data, args.cache_dir, args.feature_stage, args.target_mode,
                               region_views=args.region_views, **observation_args)
    dev_set = (CachedQADataset(args.dev_data, args.cache_dir, args.feature_stage,
                              region_views=args.dev_region_views, **observation_args)
               if args.dev_data else None)
    sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=args.seed) \
        if distributed else RandomSampler(train_set)
    batches_per_epoch = math.ceil(len(sampler) / args.batch_size)

    backbone = None
    # Serial CPU loading avoids all ranks reading the 15GB checkpoint together.
    for loading_rank in range(world):
        if rank == 0:
            (output / "progress.json").write_text(json.dumps({
                "status": "loading_model", "loading_rank": loading_rank,
                "world_size": world, "pid": os.getpid(),
                "updated_unix_seconds": time.time()}), encoding="utf-8")
        if loading_rank == rank:
            print(json.dumps({"status": "loading_model", "rank": rank}), flush=True)
            resume_adapter = (str(Path(args.resume_from) / "adapter")
                              if args.resume_from else args.init_adapter)
            resume_projector = ""
            if args.resume_from:
                if args.feature_stage == FEATURE_STAGE_EVA:
                    args.qformer_adapter = str(Path(args.resume_from) / "qformer_adapter.pt")
                candidate = Path(args.resume_from) / "projector.pt"
                if not candidate.is_file():
                    raise RuntimeError("resume checkpoint is missing projector.pt: %s" % candidate)
                resume_projector = str(candidate)
            backbone = build_text_model(
                args, train=True, adapter=resume_adapter,
                projector=resume_projector or args.projector)
            print(json.dumps({"status": "model_loaded", "rank": rank}), flush=True)
        if distributed:
            torch.distributed.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    model = CachedVideoQA(
        backbone, task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version],
        feature_stage=args.feature_stage).to(device)
    record_gpu_memory(output, rank, device, "model_to_device")
    raw_model = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        output_device=local_rank,
                                        find_unused_parameters=False,
                                        gradient_as_bucket_view=True,
                                        bucket_cap_mb=10)
        record_gpu_memory(output, rank, device, "ddp_wrapped")
    named_trainable = [(name, parameter) for name, parameter in raw_model.named_parameters()
                       if parameter.requires_grad]
    trainable = [parameter for _, parameter in named_trainable]
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters")
    projector_parameters = [parameter for name, parameter in named_trainable
                            if "backbone.llm_proj." in name]
    bridge_parameters = [parameter for name, parameter in named_trainable
                         if "backbone.visual_bridge." in name]
    lora_parameters = [parameter for name, parameter in named_trainable
                       if "backbone.llm_proj." not in name and "backbone.visual_bridge." not in name]
    if args.train_projector and not projector_parameters:
        raise RuntimeError("--train-projector was set but llm_proj is not trainable")
    parameter_groups = []
    if lora_parameters:
        parameter_groups.append({"params": lora_parameters, "lr": args.lr,
                                 "group_name": "lora"})
    if projector_parameters:
        parameter_groups.append({"params": projector_parameters,
                                 "lr": (args.projector_lr
                                        if args.projector_lr is not None else args.lr),
                                 "group_name": "projector"})
    if bridge_parameters:
        parameter_groups.append({"params": bridge_parameters, "lr": args.qformer_lr,
                                 "group_name": "qformer"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    groups_per_epoch = math.ceil(batches_per_epoch / args.grad_accumulation)
    total_steps = groups_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, round(total_steps * args.warmup_ratio))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    initial_lora = next((p.detach().float().cpu().clone()
                         for n, p in raw_model.named_parameters()
                         if p.requires_grad and "lora_B" in n), None)
    initial_projector = (raw_model.backbone.llm_proj.weight.detach().float().cpu().clone()
                         if projector_parameters else None)
    initial_queries = (raw_model.backbone.visual_bridge.query_tokens.detach().float().cpu().clone()
                       if bridge_parameters else None)
    tracked_bridge_name, initial_bridge = None, None
    if bridge_parameters:
        tracked_bridge_name, tracked_bridge = next(
            (n, p) for n, p in named_trainable
            if "visual_bridge.Qformer." in n and p.ndim == 2)
        initial_bridge = tracked_bridge.detach().float().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    global_step = micro_step = 0
    start_epoch = start_batch = 0
    data_sha256 = file_sha256(args.data)
    bridge_sha256 = file_sha256(args.bridge_init) if args.feature_stage == FEATURE_STAGE_EVA else None
    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        resume = torch.load(state_path, map_location="cpu")
        if resume["data_sha256"] != data_sha256:
            raise RuntimeError("resume dataset fingerprint does not match current --data")
        if resume["world_size"] != world:
            raise RuntimeError("resume world_size=%s, current world_size=%s" %
                               (resume["world_size"], world))
        if resume["total_steps"] != total_steps:
            raise RuntimeError("resume total_steps=%s, current total_steps=%s" %
                               (resume["total_steps"], total_steps))
        if resume.get("target_mode", "reference") != args.target_mode:
            raise RuntimeError("resume training targets differ")
        if resume.get("feature_stage", args.feature_stage) != args.feature_stage:
            raise RuntimeError("resume feature_stage does not match current run")
        if bool(resume.get("train_projector", args.train_projector)) != args.train_projector:
            raise RuntimeError("resume train_projector does not match current run")
        if bool(resume.get("train_qformer", False)) != args.train_qformer:
            raise RuntimeError("resume train_qformer does not match current run")
        for key, default in (("qformer_mode", "auto"), ("precision_protocol", "legacy")):
            if resume.get(key, default) != getattr(args, key):
                raise RuntimeError("resume %s does not match current run" % key)
        if resume.get("bridge_init_sha256") != bridge_sha256:
            raise RuntimeError("resume visual initialization differs")
        if resume.get("region_identity", {}) != view_identity:
            raise RuntimeError("resume region plans, condition, or frozen encoder differ")
        if resume.get("answer_only_logits", False) != args.answer_only_logits:
            raise RuntimeError("resume answer-only logits configuration changed")
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        global_step = int(resume["global_step"])
        micro_step = int(resume["micro_step"])
        start_epoch = int(resume["next_epoch"])
        start_batch = int(resume["next_batch_index"])
        restore_rng_state(resume["rng_states"][rank])
    first_forward_recorded = first_backward_recorded = first_step_recorded = False
    started = time.time()
    config = vars(args).copy()
    config.update(parent_info)
    config["region_identity"] = view_identity
    if bridge_sha256 is not None:
        config["bridge_init_sha256"] = bridge_sha256
        config["initial_qformer_sha256"] = file_sha256(args.qformer_adapter) if args.qformer_adapter else None
    if args.init_adapter:
        from scripts.generate_answers import directory_fingerprint
        config["initial_adapter_sha256"] = directory_fingerprint(args.init_adapter)
    if args.projector:
        config["initial_projector_sha256"] = file_sha256(args.projector)
    retain_initialization_provenance(config, args.resume_from)
    config.update({"world_size": world, "records": len(train_set),
                   "data_sha256": data_sha256, "total_steps": total_steps,
                   "batches_per_epoch": batches_per_epoch,
                   "resumed_global_step": global_step,
                   "trainable_parameters_by_group": {
                       g["group_name"]: sum(p.numel() for p in g["params"])
                       for g in optimizer.param_groups}})
    if rank == 0:
        (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        epoch_sampler = OffsetSampler(sampler, epoch_start_batch * args.batch_size)
        loader = DataLoader(train_set, batch_size=args.batch_size, sampler=epoch_sampler,
                            collate_fn=qa_collate, num_workers=2, pin_memory=True)
        model.train()
        accumulated_loss = torch.zeros((), device=device)
        accumulated_micro = 0
        for batch_index, batch in enumerate(loader, start=epoch_start_batch):
            final_micro = batch_index + 1 == batches_per_epoch
            group_position = batch_index % args.grad_accumulation
            should_step = group_position == args.grad_accumulation - 1 or final_micro
            remaining = batches_per_epoch - (batch_index - group_position)
            divisor = min(args.grad_accumulation, remaining)
            sync = model.no_sync() if distributed and not should_step else contextlib.nullcontext()
            with sync:
                features = batch["features"].to(device, non_blocking=True)
                questions = [row["question"] for row in batch["rows"]]
                loss = model(features, questions, batch["answers"], prompts=batch.get("prompts"),
                             answer_only_logits=args.answer_only_logits).loss
                accumulated_loss += loss.detach().float()
                accumulated_micro += 1
                if not first_forward_recorded:
                    record_gpu_memory(output, rank, device, "first_forward")
                    first_forward_recorded = True
                (loss / divisor).backward()
                if not first_backward_recorded:
                    record_gpu_memory(output, rank, device, "first_backward")
                    first_backward_recorded = True
            micro_step += 1
            if should_step:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not first_step_recorded:
                    gradients = {g["group_name"]: {
                        "parameters": len(g["params"]),
                        "with_gradient": sum(p.grad is not None for p in g["params"]),
                        "nonzero": sum(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                       for p in g["params"])} for g in optimizer.param_groups}
                    (output / ("gradients_rank_%d.json" % rank)).write_text(
                        json.dumps(gradients, indent=2), encoding="utf-8")
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if not first_step_recorded:
                    record_gpu_memory(output, rank, device, "first_optimizer_step")
                    first_step_recorded = True
                mean_loss = accumulated_loss / accumulated_micro
                if distributed:
                    torch.distributed.all_reduce(mean_loss)
                    mean_loss /= world
                accumulated_loss = torch.zeros((), device=device)
                accumulated_micro = 0
                if rank == 0:
                    progress = {"status": "running", "epoch": epoch + 1,
                                "optimizer_step": global_step,
                                "optimizer_steps_total": total_steps,
                                "progress_percent": 100 * global_step / total_steps,
                                "loss": float(mean_loss),
                                "loss_scope": "all_ranks_accumulation_mean",
                                "gradient_norm": float(gradient_norm),
                                "elapsed_seconds": time.time() - started,
                                "updated_unix_seconds": time.time(),
                                "learning_rates": {
                                    group.get("group_name", str(index)): group["lr"]
                                    for index, group in enumerate(optimizer.param_groups)}}
                    if parent_info:
                        progress.update(parent_info)
                        progress["cumulative_optimizer_steps"] = parent_info["parent_optimizer_steps"] + global_step
                    (output / "progress.json").write_text(json.dumps(progress, indent=2),
                                                          encoding="utf-8")
                    with (output / "training_log.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
                if args.save_every > 0 and global_step % args.save_every == 0:
                    next_epoch = epoch
                    next_batch = batch_index + 1
                    if next_batch >= batches_per_epoch:
                        next_epoch, next_batch = epoch + 1, 0
                    save_checkpoint(
                        output, raw_model, optimizer, scheduler,
                        {"global_step": global_step, "micro_step": micro_step,
                         "next_epoch": next_epoch, "next_batch_index": next_batch,
                         "data_sha256": data_sha256, "world_size": world,
                         "total_steps": total_steps,
                         "feature_stage": args.feature_stage,
                         "bridge_init_sha256": bridge_sha256,
                         "qformer_mode": args.qformer_mode,
                         "precision_protocol": args.precision_protocol,
                         "train_projector": args.train_projector, "train_qformer": args.train_qformer,
                         "region_identity": view_identity, "answer_only_logits": args.answer_only_logits,
                         "target_mode": args.target_mode},
                        rank, world, distributed)
                if global_step >= total_steps:
                    break
        if distributed:
            torch.distributed.barrier()
        if rank == 0:
            epoch_dir = output / ("epoch-%02d" % (epoch + 1))
            raw_model.backbone.llm_model.save_pretrained(epoch_dir / "adapter")
            if hasattr(raw_model.backbone, "llm_proj"):
                save_projector(raw_model.backbone, epoch_dir / "projector.pt")
            save_bridge(raw_model.backbone, epoch_dir / "qformer_adapter.pt")
            loss_value = dev_loss(raw_model, dev_set, device, args.max_dev_records, args.answer_only_logits)
            (epoch_dir / "metrics.json").write_text(
                json.dumps({"epoch": epoch + 1, "dev_teacher_forced_loss": loss_value}, indent=2),
                encoding="utf-8")
        if distributed:
            torch.distributed.barrier()
        if global_step >= total_steps:
            break
    checksums = torch.stack([p.detach().sum(dtype=torch.float64) for p in trainable])
    rank_checksums = [torch.empty_like(checksums) for _ in range(world)]
    if distributed:
        torch.distributed.all_gather(rank_checksums, checksums)
    else:
        rank_checksums[0] = checksums
    checksum_spread = float((torch.stack(rank_checksums).max(0).values -
                             torch.stack(rank_checksums).min(0).values).abs().max())
    if rank == 0:
        raw_model.backbone.llm_model.save_pretrained(output / "final_adapter")
        final_projector = None
        if hasattr(raw_model.backbone, "llm_proj"):
            final_projector = output / "final_projector.pt"
            save_projector(raw_model.backbone, final_projector)
        save_bridge(raw_model.backbone, output / "final_qformer_adapter.pt")
        current_lora = next((p.detach().float().cpu()
                             for n, p in raw_model.named_parameters()
                             if p.requires_grad and "lora_B" in n), None)
        lora_delta = (float((current_lora - initial_lora).abs().max())
                      if initial_lora is not None else None)
        projector_delta = None
        if initial_projector is not None:
            current_projector = raw_model.backbone.llm_proj.weight.detach().float().cpu()
            projector_delta = float((current_projector - initial_projector).abs().max())
        summary = {"completed": True, "optimizer_steps": global_step,
                   "micro_steps": micro_step, "elapsed_seconds": time.time() - started,
                   "tracked_lora_max_abs_delta": lora_delta,
                   "tracked_projector_max_abs_delta": projector_delta,
                   "gradient_checkpointing": args.gradient_checkpointing,
                   "gpu_memory_rank0": gpu_memory(device),
                   "feature_stage": args.feature_stage,
                   "train_projector": args.train_projector,
                   "train_qformer": args.train_qformer,
                   "qformer_mode": args.qformer_mode,
                   "region_identity": view_identity,
                   "precision_protocol": args.precision_protocol,
                   "max_parameter_checksum_spread": checksum_spread,
                   "tracked_bridge_parameter": tracked_bridge_name,
                   "tracked_bridge_max_abs_delta": (
                       float((dict(raw_model.named_parameters())[tracked_bridge_name].detach().float().cpu()
                              - initial_bridge).abs().max()) if initial_bridge is not None else None),
                   "tracked_queries_max_abs_delta": (
                       float((raw_model.backbone.visual_bridge.query_tokens.detach().float().cpu()
                              - initial_queries).abs().max()) if initial_queries is not None else None),
                   "final_qformer_adapter": (str((output / "final_qformer_adapter.pt").resolve())
                                             if hasattr(raw_model.backbone, "visual_bridge") else None),
                   "final_adapter": str((output / "final_adapter").resolve()),
                   "final_projector": (str(final_projector.resolve())
                                       if final_projector is not None else None)}
        if parent_info:
            summary.update(parent_info)
            summary["cumulative_optimizer_steps"] = parent_info["parent_optimizer_steps"] + global_step
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Persist the actual Python failure so the monitor does not keep
        # reporting a run with only run_config.json as possibly running.
        args = parse_args()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", 0))
        failure = {"status": "failed", "rank": rank, "pid": os.getpid(),
                   "updated_unix_seconds": time.time(), "traceback": traceback.format_exc()}
        if torch.cuda.is_available():
            try:
                failure["gpu_memory"] = gpu_memory(torch.cuda.current_device())
            except Exception as memory_error:
                failure["gpu_memory_error"] = repr(memory_error)
        (output / ("failure_rank_%d.json" % rank)).write_text(
            json.dumps(failure, indent=2), encoding="utf-8")
        raise
