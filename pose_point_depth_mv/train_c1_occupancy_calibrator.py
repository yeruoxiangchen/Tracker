#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.c1_occupancy import (
    C1_CALIBRATOR_CHECKPOINT_VERSION,
    C1_ENRICHMENT_SUMMARY_VERSION,
    C1MapTargetDataset,
    MonotoneOccupancyCalibrator,
    balanced_binary_loss,
    c1_policy_scores,
    calibrator_parameter_values,
    file_sha256,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a monotone C1.1 occupancy calibrator. It has no XYZ, Flow, "
            "convolution, image, or target-derived input features."
        )
    )
    parser.add_argument("--c1_summary", required=True)
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weight_policy", default="summary")
    parser.add_argument("--target_mode", choices=("exact", "surface_r1"), default="exact")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_policy_ablation",
        action="store_true",
        help="Allow a policy other than the C1.0 multi-seed admitted policy.",
    )
    return parser.parse_args()


def recursive_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.float()).all().item())
    if isinstance(value, dict):
        return all(recursive_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(recursive_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def validate_summary_binding(
    summary: dict[str, Any], c0_report: Path
) -> tuple[dict[str, Any], str]:
    if summary.get("format") != C1_ENRICHMENT_SUMMARY_VERSION:
        raise ValueError("unexpected C1.0 summary format")
    if summary.get("passed") is not True:
        raise ValueError("C1.1 is blocked because C1.0 did not pass")
    matches: list[dict[str, Any]] = []
    for path in summary.get("source_reports", []):
        report = load_json(path)
        if Path(report["source_c0_report"]).resolve() == c0_report.resolve():
            matches.append(report)
    if len(matches) != 1:
        raise ValueError("training C0 report is not uniquely bound to C1.0 summary")
    report = matches[0]
    if report.get("split_name") != "train16" or report.get("passed") is not True:
        raise ValueError("C1.1 training requires the passed train16 C1.0 report")
    admitted = summary.get("admitted_policy")
    if not admitted:
        raise ValueError("C1.0 summary has no admitted policy")
    return report, str(admitted)


def checkpoint_payload(
    *,
    models: dict[str, MonotoneOccupancyCalibrator],
    optimizers: dict[str, torch.optim.Optimizer],
    step: int,
    args: argparse.Namespace,
    dataset: C1MapTargetDataset,
    summary_path: Path,
    weight_policy: str,
    initial_parameters: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "format": C1_CALIBRATOR_CHECKPOINT_VERSION,
        "step": int(step),
        "model_states": {name: model.state_dict() for name, model in models.items()},
        "optimizer_states": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "model_metadata": {name: model.metadata() for name, model in models.items()},
        "parameter_values": {
            name: calibrator_parameter_values(model) for name, model in models.items()
        },
        "initial_parameter_values": initial_parameters,
        "nested_models": {
            "M0_bias": "bias only",
            "M1_reliability": "bias + reliability",
            "M2_weight_reliability": "bias + reliability + admitted weight",
        },
        "shared_training_protocol": {
            "same_objects": True,
            "same_target": True,
            "same_optimizer": "Adam",
            "same_lr": float(args.lr),
            "same_steps": int(args.max_steps),
            "same_initialization_seed": int(args.seed),
            "same_object_balanced_loss": True,
            "independent_parameters_and_optimizer_states": True,
        },
        "training_seed": int(args.seed),
        "weight_policy": weight_policy,
        "target_mode": args.target_mode,
        "source_c1_summary": str(summary_path.resolve()),
        "source_c1_summary_sha256": file_sha256(summary_path),
        "source_c0_report": str(dataset.report_path),
        "source_c0_checkpoint": dataset.report["checkpoint"],
        "source_c0_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_cache_config_hash": dataset.report["cache_config_hash"],
        "train_uids": [str(row["uid"]) for row in dataset.records],
        "args": vars(args),
        "flow_loaded": False,
        "flow_lora_enabled": False,
        "decoder_loaded": False,
        "target_used_as_input": False,
    }


def main() -> None:
    args = parse_args()
    if int(args.max_steps) <= 0:
        raise ValueError("--max_steps must be positive")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    summary_path = Path(args.c1_summary).resolve()
    c0_path = Path(args.c0_report).resolve()
    summary = load_json(summary_path)
    _, admitted_policy = validate_summary_binding(summary, c0_path)
    weight_policy = admitted_policy if args.weight_policy == "summary" else args.weight_policy
    if weight_policy != admitted_policy and not args.allow_policy_ablation:
        raise ValueError(
            f"policy={weight_policy} differs from admitted={admitted_policy}; "
            "use --allow_policy_ablation only for an explicit control"
        )

    dataset = C1MapTargetDataset(c0_path)
    objects: list[dict[str, torch.Tensor | str]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        active = item["map"]["active_mask"].bool()
        policies = c1_policy_scores(item["map"])
        if weight_policy not in policies:
            raise ValueError(f"unknown C1 weight policy: {weight_policy}")
        target = item["targets"][args.target_mode].float()
        score = policies[weight_policy].float()
        reliability = item["map"]["audit_maps"]["raw_reliability"].float()
        objects.append(
            {
                "uid": item["uid"],
                "score": score[active],
                "reliability": reliability[active],
                "target": target[active],
            }
        )
    models = {
        "M0_bias": MonotoneOccupancyCalibrator(
            include_score=False, include_reliability=False
        ),
        "M1_reliability": MonotoneOccupancyCalibrator(
            include_score=False, include_reliability=True
        ),
        "M2_weight_reliability": MonotoneOccupancyCalibrator(
            include_score=True, include_reliability=True
        ),
    }
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=float(args.lr))
        for name, model in models.items()
    }
    initial_parameters = {
        name: calibrator_parameter_values(model) for name, model in models.items()
    }
    shared_initialization_equal = bool(
        initial_parameters["M0_bias"]["bias"]
        == initial_parameters["M1_reliability"]["bias"]
        == initial_parameters["M2_weight_reliability"]["bias"]
        and initial_parameters["M1_reliability"]["reliability_weight"]
        == initial_parameters["M2_weight_reliability"]["reliability_weight"]
    )
    history: list[dict[str, Any]] = []
    for step in range(1, int(args.max_steps) + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        losses: dict[str, torch.Tensor] = {}
        for name, model in models.items():
            object_losses = [
                balanced_binary_loss(
                    model(row["score"], row["reliability"]), row["target"]
                )
                for row in objects
            ]
            losses[name] = torch.stack(object_losses).mean()
        total_loss = torch.stack(list(losses.values())).sum()
        if not bool(torch.isfinite(total_loss).item()):
            raise RuntimeError(f"non-finite C1.1 nested loss at step={step}")
        total_loss.backward()
        gradient_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad.float()).all().item())
            for model in models.values()
            for parameter in model.parameters()
        )
        if not gradient_finite:
            raise RuntimeError(f"non-finite C1.1 gradient at step={step}")
        for optimizer in optimizers.values():
            optimizer.step()
        if not recursive_finite(
            {name: model.state_dict() for name, model in models.items()}
        ) or not recursive_finite(
            {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
        ):
            raise RuntimeError(f"non-finite C1.1 state at step={step}")
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            row = {
                "step": step,
                "losses": {
                    name: float(loss.detach().item()) for name, loss in losses.items()
                },
                "parameters": {
                    name: calibrator_parameter_values(model)
                    for name, model in models.items()
                },
            }
            history.append(row)
            print(f"[c1_calibrator] {json.dumps(row)}", flush=True)
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            payload = checkpoint_payload(
                models=models,
                optimizers=optimizers,
                step=step,
                args=args,
                dataset=dataset,
                summary_path=summary_path,
                weight_policy=weight_policy,
                initial_parameters=initial_parameters,
            )
            torch.save(payload, checkpoints_dir / f"step_{step:06d}.pt")
            torch.save(payload, checkpoints_dir / "last.pt")

    report = {
        "format": C1_CALIBRATOR_CHECKPOINT_VERSION,
        "stage": "C1.1 nested monotone occupancy calibrator training",
        "passed": True,
        "completed_steps": int(args.max_steps),
        "object_count": len(objects),
        "training_seed": int(args.seed),
        "weight_policy": weight_policy,
        "target_mode": args.target_mode,
        "model_metadata": {
            name: model.metadata() for name, model in models.items()
        },
        "parameter_values": {
            name: calibrator_parameter_values(model)
            for name, model in models.items()
        },
        "initial_parameter_values": initial_parameters,
        "history": history,
        "checks": {
            "c1_summary_passed": summary.get("passed") is True,
            "training_c0_bound_to_summary": True,
            "all_model_states_finite": recursive_finite(
                {name: model.state_dict() for name, model in models.items()}
            ),
            "all_optimizer_states_finite": recursive_finite(
                {
                    name: optimizer.state_dict()
                    for name, optimizer in optimizers.items()
                }
            ),
            "m2_score_weight_positive": calibrator_parameter_values(
                models["M2_weight_reliability"]
            )["score_weight"]
            > 0.0,
            "nested_models_independently_trained": set(models)
            == {"M0_bias", "M1_reliability", "M2_weight_reliability"},
            "shared_initialization_equal": shared_initialization_equal,
            "does_not_use_xyz": True,
            "does_not_use_flow_state": True,
            "flow_lora_disabled": True,
            "target_not_used_as_input": True,
        },
        "source_c1_summary": str(summary_path),
        "source_c0_report": str(dataset.report_path),
        "checkpoint": str((checkpoints_dir / "last.pt").resolve()),
    }
    report["passed"] = all(report["checks"].values())
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
