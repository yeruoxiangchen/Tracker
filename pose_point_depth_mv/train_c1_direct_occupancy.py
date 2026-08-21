#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch

from pose_point_depth_mv.c1_direct_occupancy import (
    C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION,
    DIRECT_MODELS,
    extract_direct_occupancy_objects,
    fit_normalization,
    initialize_nested_models,
    model_logits,
    nested_base_initialization_equal,
    normalize_features,
    protocol_hash,
)
from pose_point_depth_mv.c1_matched_budget import (
    C1_MATCHED_BUDGET_SUMMARY_VERSION,
    CORRUPTION_POLICY_NAMES,
)
from pose_point_depth_mv.c1_occupancy import (
    C1MapTargetDataset,
    balanced_binary_loss,
    file_sha256,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent C1.1b M0 reliability, M1 per-view "
            "visual/geometry, and M2 correspondence-augmented occupancy probes."
        )
    )
    parser.add_argument("--c1_0b_summary", required=True)
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_mode", choices=("exact", "surface_r1"), default="exact")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_score_diff", type=float, default=1.0e-3)
    return parser.parse_args()


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _validate_admission(
    summary_path: Path, dataset: C1MapTargetDataset, seed: int
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    summary = load_json(summary_path)
    if summary.get("format") != C1_MATCHED_BUDGET_SUMMARY_VERSION:
        raise ValueError("unexpected C1.0b summary format")
    if summary.get("integrity_passed") is not True or summary.get("passed") is not True:
        raise ValueError("C1.1b is blocked because C1.0b did not admit a route")
    route = str(summary["decision"]["route"])
    if route not in {
        "restricted_surface_occupancy_gate_candidate",
        "auxiliary_correspondence_feature_only",
    }:
        raise ValueError(f"C1.0b route does not allow C1.1b: {route}")
    policy = summary["decision"].get("selected_policy")
    if not policy:
        raise ValueError("C1.0b summary has no selected policy")
    n3_path = Path(summary["source_n3_report"]).resolve()
    n3 = load_json(n3_path)
    matches = [row for row in n3["per_seed"] if int(row["seed"]) == int(seed)]
    if len(matches) != 1:
        raise ValueError(f"N3 seed={seed} is not unique")
    n3_row = matches[0]
    expected_report = Path(n3_row["run_dir"]) / "c0_3_train16" / "report.json"
    if dataset.report_path != expected_report.resolve():
        raise ValueError("C1.1b train C0 report is not bound to the N3 seed run")
    if dataset.report.get("split_name") != "train16":
        raise ValueError("C1.1b training requires train16")
    if dataset.report["checkpoint_sha256"] != n3_row["checkpoint_sha256"]:
        raise ValueError("C1.1b checkpoint hash differs from N3")
    return summary, str(policy), n3_row


def _checkpoint_payload(
    *,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    normalization: dict[str, torch.Tensor],
    step: int,
    args: argparse.Namespace,
    dataset: C1MapTargetDataset,
    summary_path: Path,
    policy: str,
    feature_metadata: dict[str, Any],
    training_protocol: dict[str, Any],
    training_uids: list[str],
) -> dict[str, Any]:
    return {
        "format": C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION,
        "step": int(step),
        "model_states": {name: _cpu_state(model) for name, model in models.items()},
        "optimizer_states": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "normalization": {name: value.cpu() for name, value in normalization.items()},
        "model_names": list(DIRECT_MODELS),
        "model_config": {
            "input_dim": int(feature_metadata["input_dim"]),
            "hidden_dim": int(args.hidden_dim),
        },
        "training_seed": int(args.seed),
        "target_mode": str(args.target_mode),
        "policy": policy,
        "feature_metadata": feature_metadata,
        "training_protocol": training_protocol,
        "training_protocol_hash": protocol_hash(training_protocol),
        "source_c1_0b_summary": str(summary_path),
        "source_c1_0b_summary_sha256": file_sha256(summary_path),
        "source_c0_report": str(dataset.report_path),
        "source_c0_checkpoint": dataset.report["checkpoint"],
        "source_c0_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_cache_config_hash": dataset.report["cache_config_hash"],
        "train_uids": list(training_uids),
        "args": vars(args),
        "flow_loaded": False,
        "flow_lora_enabled": False,
        "decoder_loaded": False,
        "target_used_as_input": False,
    }


def main() -> None:
    args = parse_args()
    if min(int(args.hidden_dim), int(args.max_steps), int(args.save_every)) <= 0:
        raise ValueError("hidden_dim/max_steps/save_every must be positive")
    if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("invalid optimizer settings")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    device = torch.device(args.device)
    summary_path = Path(args.c1_0b_summary).resolve()
    dataset = C1MapTargetDataset(args.c0_report)
    summary, policy, _ = _validate_admission(summary_path, dataset, args.seed)

    objects, feature_metadata = extract_direct_occupancy_objects(
        dataset,
        policy=policy,
        target_mode=args.target_mode,
        device=device,
        max_samples=int(args.max_samples),
        max_score_diff=float(args.max_score_diff),
    )
    normalization = fit_normalization(objects)
    models = initialize_nested_models(
        input_dim=int(feature_metadata["input_dim"]),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
    )
    shared_base_initialization = nested_base_initialization_equal(models)
    models = {name: model.to(device) for name, model in models.items()}
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        for name, model in models.items()
    }
    normalization_device = {
        name: value.to(device) for name, value in normalization.items()
    }
    training_protocol = {
        "nested_models": {
            "M0_reliability": "reliability only",
            "M1_view_geometry": (
                "reliability + frozen per-view visual/geometry mean/std"
            ),
            "M2_plus_correspondence": (
                "M1 inputs + one matched-budget correspondence weight"
            ),
        },
        "independent_models_and_optimizers": True,
        "m1_m2_base_initialization_identical": shared_base_initialization,
        "correct_branch_only_used_for_supervised_training": True,
        "corruptions_reserved_for_paired_causal_evaluation": list(
            CORRUPTION_POLICY_NAMES
        ),
        "same_object_schedule_and_voxels_for_all_models": True,
        "object_balanced_binary_loss": True,
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "target_mode": str(args.target_mode),
        "policy": policy,
    }
    rng = random.Random(int(args.seed))
    order = list(range(len(objects)))
    rng.shuffle(order)
    history: list[dict[str, Any]] = []
    for step in range(1, int(args.max_steps) + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            rng.shuffle(order)
        row = objects[order[(step - 1) % len(order)]]
        active = row["active"]
        reliability = row["reliability"][active].to(device)
        target = row["target"][active].to(device)
        base_raw = row["candidates"]["correct"]["base"][active].to(device)
        corr_raw = row["candidates"]["correct"]["correspondence"][active].to(device)
        base, correspondence = normalize_features(
            base_raw, corr_raw, normalization_device
        )
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        logits = model_logits(
            models,
            reliability=reliability,
            base=base,
            correspondence=correspondence,
        )
        losses = {
            name: balanced_binary_loss(value, target)
            for name, value in logits.items()
        }
        total_loss = torch.stack(list(losses.values())).sum()
        if not bool(torch.isfinite(total_loss).item()):
            raise RuntimeError(f"non-finite C1.1b loss at step={step}")
        total_loss.backward()
        gradients_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad.float()).all().item())
            for model in models.values()
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise RuntimeError(f"non-finite C1.1b gradients at step={step}")
        for optimizer in optimizers.values():
            optimizer.step()
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            log_row = {
                "step": step,
                "uid": row["uid"],
                "losses": {
                    name: float(value.detach().item())
                    for name, value in losses.items()
                },
            }
            history.append(log_row)
            print(f"[c1_1b_train] {json.dumps(log_row)}", flush=True)
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            payload = _checkpoint_payload(
                models=models,
                optimizers=optimizers,
                normalization=normalization,
                step=step,
                args=args,
                dataset=dataset,
                summary_path=summary_path,
                policy=policy,
                feature_metadata=feature_metadata,
                training_protocol=training_protocol,
                training_uids=[str(row["uid"]) for row in objects],
            )
            torch.save(payload, checkpoints_dir / f"step_{step:06d}.pt")
            torch.save(payload, checkpoints_dir / "last.pt")

    checks = {
        "c1_0b_summary_admitted": summary.get("passed") is True,
        "train_c0_bound_to_n3": True,
        "all_three_models_independently_trained": set(models) == set(DIRECT_MODELS),
        "m1_m2_base_initialization_identical": shared_base_initialization,
        "correct_only_supervised_training": True,
        "corruptions_not_used_as_target_labels": True,
        "all_parameters_finite": all(
            bool(torch.isfinite(parameter).all().item())
            for model in models.values()
            for parameter in model.parameters()
        ),
        "target_not_used_as_input": True,
        "flow_lora_disabled": True,
    }
    report = {
        "format": C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION,
        "stage": "C1.1b nested direct local occupancy probe training",
        "passed": all(checks.values()),
        "checks": checks,
        "completed_steps": int(args.max_steps),
        "training_seed": int(args.seed),
        "object_count": len(objects),
        "target_mode": args.target_mode,
        "policy": policy,
        "feature_metadata": feature_metadata,
        "training_protocol": training_protocol,
        "history": history,
        "checkpoint": str((checkpoints_dir / "last.pt").resolve()),
        "source_c1_0b_summary": str(summary_path),
        "source_c0_report": str(dataset.report_path),
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
