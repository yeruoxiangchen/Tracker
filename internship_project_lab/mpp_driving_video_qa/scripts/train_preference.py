#!/usr/bin/env python3
"""DPO + matched CE or CE-only control, using cached fixed joint log-probabilities."""
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
from mpp_video_qa.core import file_sha256
from mpp_video_qa.preference import (PreferenceDataset, cache_inventory, read_reference_bundle,
                                    tokenizer_identity, encode_completion, completion_logprob, dpo_loss)
from mpp_video_qa.opd import average_gradients
from scripts.generate_answers import write_json_atomic
from scripts.preference_common import add_model_arguments, make_model, validate_selection
from scripts.train_sft import seed_all, rng_state, restore_rng_state, gpu_memory
from mpp_video_qa.model import save_projector


def parse_args():
    p = argparse.ArgumentParser()
    add_model_arguments(p)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("dpo", "sft_control"), default="dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--chosen-ce-weight", type=float, default=0.2)
    p.add_argument("--replay-ce-weight", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--projector-lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", default="")
    p.add_argument("--verify-reference", action="store_true", help="at fresh initialization compare first scores to offline reference")
    return p.parse_args()


def save_bundle(output, name, model, optimizer, state, rank, world):
    rngs = [None] * world
    if world > 1:
        torch.distributed.all_gather_object(rngs, rng_state())
    else:
        rngs[0] = rng_state()
    destination = output / name
    if rank == 0:
        if destination.exists():
            raise ValueError("checkpoint exists: %s" % destination)
        temporary = output / (name + ".incomplete")
        temporary.mkdir(parents=True, exist_ok=False)
        model.backbone.llm_model.save_pretrained(temporary / "adapter", selected_adapters=["default"])
        save_projector(model.backbone, temporary / "projector.pt")
        torch.save(dict(state, optimizer=optimizer.state_dict(), rng_states=rngs), temporary / "training_state.pt")
        write_json_atomic(temporary / "metadata.json", state)
        temporary.replace(destination)
    if world > 1:
        torch.distributed.barrier()


def main():
    args = parse_args()
    if min(args.epochs, args.grad_accumulation, args.beta, args.chosen_ce_weight, args.replay_ce_weight,
           args.lr, args.projector_lr) <= 0:
        raise ValueError("positive epochs/accumulation/loss weights/learning rates required")
    if args.resume_from and args.verify_reference:
        raise ValueError("reference equality check applies only to fresh joint initialization")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=12))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    progress = lambda **kw: write_json_atomic(output / "progress.json", dict(updated_unix_seconds=time.time(), **kw)) if rank == 0 else None
    if (output / "summary.json").exists() or ((output / "run_config.json").exists() and not args.resume_from):
        raise ValueError("run already exists; choose a new output or resume an unfinished run")
    progress(status="checking_data_and_reference", rank0_pid=os.getpid())
    dataset = PreferenceDataset(args.data, args.cache_dir)
    selection, init = validate_selection(args, dataset.rows)
    manifest, ref_summary, refs = read_reference_bundle(args.reference_dir, args.data)
    if manifest["initialization"] != init:
        raise ValueError("reference was not computed with this joint initialization")
    inventory_ok = [None]
    if rank == 0:
        inventory_ok[0] = cache_inventory(dataset.rows, args.cache_dir) == manifest["visual_cache_sha256"]
    if world > 1:
        torch.distributed.broadcast_object_list(inventory_ok, src=0)
    if not inventory_ok[0]:
        raise ValueError("visual cache changed after reference scoring")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True)
    per_epoch = math.ceil(len(sampler) / args.grad_accumulation)
    total_steps = min(per_epoch * args.epochs, args.max_steps) if args.max_steps > 0 else per_epoch * args.epochs
    identity = {"data_sha256": file_sha256(args.data), "selection_sha256": file_sha256(Path(args.data).parent / "manifest.json"),
                "reference_manifest_sha256": file_sha256(args.reference_dir / "manifest.json"),
                "reference_sha256": ref_summary["reference_sha256"], "initialization": init,
                "mode": args.mode, "beta": args.beta, "chosen_ce_weight": args.chosen_ce_weight,
                "replay_ce_weight": args.replay_ce_weight, "lr": args.lr, "projector_lr": args.projector_lr,
                "epochs": args.epochs, "world_size": world, "grad_accumulation": args.grad_accumulation,
                "total_steps": total_steps, "seed": args.seed, "dropout": 0.0}
    resume = None
    if args.resume_from:
        resume = torch.load(Path(args.resume_from) / "training_state.pt", map_location="cpu")
        if resume["identity"] != identity:
            raise ValueError("resume model/data/reference/training settings differ")
        if resume["optimizer_steps"] >= total_steps:
            raise ValueError("checkpoint has already completed this training budget")
        if rank == 0:
            for path in output.glob("failure_rank_*.json"):
                path.rename(path.with_suffix(".before_resume_%d" % time.time()))
    seed_all(args.seed)
    for loading_rank in range(world):
        progress(status="loading_model", loading_rank=loading_rank, world_size=world,
                 rank0_pid=os.getpid(), records=len(dataset), optimizer_steps_total=total_steps)
        if rank == loading_rank:
            model = make_model(args, True, device,
                               str(Path(args.resume_from) / "adapter") if resume else None,
                               str(Path(args.resume_from) / "projector.pt") if resume else None)
            print(json.dumps({"status": "model_loaded", "rank": rank}), flush=True)
        if world > 1:
            torch.distributed.barrier()
    tokenizer = model.backbone.llm_tokenizer
    if tokenizer_identity(tokenizer) != manifest["tokenizer_sha256"]:
        raise ValueError("training and reference tokenizers differ")
    for row in dataset.rows:
        ref = refs[row["example_id"]]
        for kind in ("chosen", "rejected") if row["kind"] == "preference" else ("chosen",):
            if encode_completion(tokenizer, row[kind]) != ref[kind + "_ids"]:
                raise ValueError("reference answer tokenization differs")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if world > 1:
        for parameter in parameters:
            torch.distributed.broadcast(parameter.data, src=0)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in named if "llm_model." in n], "lr": args.lr},
        {"params": [p for n, p in named if "llm_proj." in n], "lr": args.projector_lr}], weight_decay=0.01)
    step = micro = start_epoch = start_batch = 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        step, micro = resume["optimizer_steps"], resume["micro_steps"]
        start_epoch, start_batch = resume["next_epoch"], resume["next_batch"]
        restore_rng_state(resume["rng_states"][rank])
    else:
        seed_all(args.seed + rank)
    tracked = {n: p.detach().float().cpu().clone() for n, p in named
               if n.endswith("llm_proj.weight") or ("h.0.attn.c_attn.lora_B" in n)}
    if rank == 0:
        write_json_atomic(output / "run_config.json", dict(identity, records=len(dataset), pairs=selection["pairs"],
                            replay=selection["replay"], optimizer_steps_total=total_steps,
                            reference_dir=str(args.reference_dir.resolve()), trainable_parameters=sum(p.numel() for p in parameters),
                            objective="dpo_plus_matched_ce" if args.mode == "dpo" else "matched_ce_only"))
    optimizer.zero_grad(set_to_none=True)
    counters = torch.zeros(9, dtype=torch.float64, device=device)
    started = time.time()
    if args.verify_reference:
        index = next(i for i, row in enumerate(dataset.rows) if row["kind"] == "preference")
        item = dataset[index]
        row, features = item["row"], item["features"].unsqueeze(0).to(device)
        ref = refs[row["example_id"]]
        differences = {}
        with torch.no_grad():
            for kind in ("chosen", "rejected"):
                value = completion_logprob(model, features, row["question"], ref[kind + "_ids"], args.chunk_tokens)
                delta = abs(float(value) - ref["ref_" + kind + "_logp"])
                differences[kind] = delta
                if delta / len(ref[kind + "_ids"]) > 0.02:
                    raise ValueError("fresh policy/reference per-token score mismatch: %s" % differences)
        print(json.dumps({"reference_parity_abs_logp": differences, "rank": rank}), flush=True)
    state = None
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        for batch_index in range(start_batch if epoch == start_epoch else 0, len(indices)):
            if step >= total_steps:
                break
            item = dataset[indices[batch_index]]
            row = item["row"]
            ref = refs[row["example_id"]]
            features = item["features"].unsqueeze(0).to(device)
            divisor = min(args.grad_accumulation, len(indices) - (batch_index // args.grad_accumulation) * args.grad_accumulation)
            chosen = completion_logprob(model, features, row["question"], ref["chosen_ids"], args.chunk_tokens)
            ce = -chosen / len(ref["chosen_ids"])
            preference_value, margin_value, win = 0.0, 0.0, 0.0
            is_pair = row["kind"] == "preference"
            rejected = None
            if is_pair and args.mode == "dpo":
                rejected = completion_logprob(model, features, row["question"], ref["rejected_ids"], args.chunk_tokens)
                preference, margin = dpo_loss(chosen, rejected, ref["ref_chosen_logp"], ref["ref_rejected_logp"], args.beta)
                loss = preference + args.chosen_ce_weight * ce
                preference_value, margin_value, win = float(preference.detach()), float(margin), float(margin > 0)
            else:
                loss = (args.chosen_ce_weight if is_pair else args.replay_ce_weight) * ce
            if not torch.isfinite(loss):
                raise ValueError("nonfinite preference/CE loss")
            ce_value, loss_value = float(ce.detach()), float(loss.detach())
            (loss / divisor).backward()
            counters += counters.new_tensor([loss_value, preference_value, ce_value if is_pair else 0,
                                              ce_value if not is_pair else 0, int(is_pair), int(not is_pair),
                                              margin_value, win, len(ref["chosen_ids"])])
            del loss, chosen, ce, rejected
            if is_pair and args.mode == "dpo":
                del preference, margin
            micro += 1
            if (batch_index + 1) % args.grad_accumulation != 0 and batch_index + 1 != len(indices):
                continue
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
                pair_n, replay_n = float(counters[4]), float(counters[5])
                record = {"status": "running", "mode": args.mode, "epoch": epoch + 1,
                          "optimizer_steps": step, "optimizer_steps_total": total_steps,
                          "loss_mean": float(counters[0] / (pair_n + replay_n)),
                          "dpo_loss": float(counters[1] / max(1, pair_n)) if args.mode == "dpo" else None,
                          "chosen_ce": float(counters[2] / max(1, pair_n)), "replay_ce": float(counters[3] / max(1, replay_n)),
                          "preference_margin": float(counters[6] / max(1, pair_n)) if args.mode == "dpo" else None,
                          "preference_win_rate": float(counters[7] / max(1, pair_n)) if args.mode == "dpo" else None,
                          "pairs_in_step": int(pair_n), "replay_in_step": int(replay_n), "gradient_norm": float(norm),
                          "elapsed_seconds": time.time() - started, "updated_unix_seconds": time.time(),
                          "gpu_memory": gpu_memory(device)}
                write_json_atomic(output / "progress.json", record)
                with (output / "training_log.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            counters.zero_()
            if args.save_every and step % args.save_every == 0:
                save_bundle(output, "checkpoint-%06d" % step, model, optimizer, state, rank, world)
        start_batch = 0
        if state is not None and state["next_epoch"] == epoch + 1:
            save_bundle(output, "epoch-%02d" % (epoch + 1), model, optimizer, state, rank, world)
        if step >= total_steps:
            break
    if state is None:
        raise ValueError("no optimizer updates")
    save_bundle(output, "final", model, optimizer, state, rank, world)
    checksum = torch.stack([p.detach().double().sum() for _, p in named]).sum()
    all_checksums = [torch.zeros_like(checksum) for _ in range(world)]
    if world > 1:
        torch.distributed.all_gather(all_checksums, checksum)
    else:
        all_checksums[0] = checksum
    if rank == 0:
        changes = {n: float((dict(named)[n].detach().float().cpu() - initial).abs().max()) for n, initial in tracked.items()}
        summary = {"completed": True, "status": "completed", "mode": args.mode, "optimizer_steps": step,
                   "optimizer_steps_total": total_steps, "micro_steps": micro,
                   "elapsed_seconds": time.time() - started, "tracked_parameter_max_abs_delta": changes,
                   "parameter_update_verified": all(v > 0 for v in changes.values()) and bool(changes),
                   "max_parameter_checksum_spread": float(max(all_checksums) - min(all_checksums)),
                   "final_bundle": str((output / "final").resolve()), "gpu_memory": gpu_memory(device)}
        write_json_atomic(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / ("failure_rank_%s.json" % os.environ.get("RANK", "0")),
                          {"status": "failed", "traceback": traceback.format_exc(), "pid": os.getpid(),
                           "updated_unix_seconds": time.time()})
        raise
