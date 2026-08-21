#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.correspondence_head import (
    CORRESPONDENCE_CHECKPOINT_VERSION,
    DEFAULT_HELDOUT_CONTROLS,
    ViewCorrespondenceHead,
    load_correspondence_head_state,
)
from pose_point_depth_mv.eval_local_target_probe import object_balanced, summarize
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_CONTROL_NAMES,
    build_view_identity_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Object-disjoint evaluation of the explicit C0 correspondence head."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="16-63")
    parser.add_argument("--split_name", choices=("train16", "fresh48"), required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_margin_mean", type=float, default=0.0)
    parser.add_argument("--min_correct_score", type=float, default=0.0)
    parser.add_argument("--max_control_score", type=float, default=0.0)
    parser.add_argument("--max_permutation_diff", type=float, default=1.0e-5)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def permute_evidence_views(
    evidence: dict[str, Any], permutation: torch.Tensor
) -> dict[str, Any]:
    result = dict(evidence)
    for key in ("sampled_visual", "geometry", "view_weight"):
        result[key] = evidence[key].index_select(0, permutation)
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C0 Explicit View Correspondence Evaluation",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Checkpoint step: `{report['checkpoint_step']}`",
        f"- Objects: `{report['object_count']}`",
        f"- Train controls: `{report['train_controls']}`",
        f"- Held-out controls: `{report['heldout_controls']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Correct Scores",
            "",
            "```json",
            json.dumps(report["correct_score"], indent=2),
            "```",
            "",
            "## Paired Controls",
            "",
            "```json",
            json.dumps(report["controls"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != CORRESPONDENCE_CHECKPOINT_VERSION:
        raise ValueError("unexpected C0 correspondence checkpoint format")
    saved_args = checkpoint.get("args", {})
    model_summary = checkpoint.get("model_summary", {})
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    if str(model_summary.get("cache_config_hash")) != dataset.config_hash:
        raise RuntimeError("C0 checkpoint/cache hash mismatch")
    train_objects = set(str(value) for value in model_summary.get("train_object_uids", ()))
    eval_objects = {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    overlap = sorted(train_objects & eval_objects)
    if args.split_name == "fresh48" and overlap:
        raise RuntimeError(f"fresh C0 evaluation leaks train objects: {overlap}")
    if args.split_name == "train16" and eval_objects != train_objects:
        raise RuntimeError("train16 object set differs from C0 checkpoint")

    head = ViewCorrespondenceHead(
        visual_channels=dataset.visual_feature_dim,
        hidden_dim=int(saved_args["hidden_dim"]),
        pair_hidden_dim=int(saved_args["pair_hidden_dim"]),
        min_views=int(saved_args["min_views"]),
    ).to(device).eval()
    load_correspondence_head_state(head, checkpoint["model_trainable_state"])
    amp_name = str(saved_args.get("amp_dtype", "bf16"))
    use_amp = amp_name != "none"
    amp_dtype = torch.float16 if amp_name == "fp16" else torch.bfloat16
    train_controls = tuple(model_summary["protocol"]["train_controls"])
    heldout_controls = tuple(
        mode for mode in DEFAULT_HELDOUT_CONTROLS if mode not in train_controls
    )
    count = len(dataset) if args.max_samples <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    permutation_max_abs_diff = 0.0

    for index in range(count):
        sample = dataset[index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        try:
            correct = build_view_identity_evidence(sample, device=device, mode="correct")
            controls = {
                mode: build_view_identity_evidence(sample, device=device, mode=mode)
                for mode in VIEW_IDENTITY_CONTROL_NAMES
            }
            fixed_weight = correct["view_weight"].float()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                correct_result = head(correct, view_weight_override=fixed_weight)
                control_results = {
                    mode: head(evidence, view_weight_override=fixed_weight)
                    for mode, evidence in controls.items()
                }
                permutation = torch.arange(
                    int(correct["views"]) - 1, -1, -1, device=device
                )
                permuted = permute_evidence_views(correct, permutation)
                permuted_result = head(
                    permuted,
                    view_weight_override=fixed_weight.index_select(0, permutation),
                )
            correct_score = float(correct_result["sample_score"].float().item())
            permutation_max_abs_diff = max(
                permutation_max_abs_diff,
                abs(
                    correct_score
                    - float(permuted_result["sample_score"].float().item())
                ),
            )
            row: dict[str, Any] = {
                "uid": uid,
                "object_uid": object_uid,
                "views": int(correct["views"]),
                "correct_score": correct_score,
                "correct_probability": float(torch.sigmoid(correct_result["sample_score"]).item()),
                "support_ratio": float(correct_result["support_ratio"].item()),
            }
            for mode, result in control_results.items():
                score = float(result["sample_score"].float().item())
                row[f"{mode}_score"] = score
                row[f"{mode}_probability"] = float(
                    torch.sigmoid(result["sample_score"]).item()
                )
                row[f"correct_vs_{mode}"] = correct_score - score
            records.append(row)
            print(f"[correspondence_eval] {index + 1}/{count} uid={uid}", flush=True)
        except Exception as error:  # noqa: BLE001 - preserve per-sample audit errors
            failures.append({"uid": uid, "error": repr(error)})
            print(f"[correspondence_eval] FAIL uid={uid}: {error}", flush=True)

    correct_score = object_balanced(
        records, "correct_score", bootstrap_samples=int(args.bootstrap_samples)
    )
    controls_report: dict[str, Any] = {}
    control_checks: dict[str, bool] = {}
    for mode in VIEW_IDENTITY_CONTROL_NAMES:
        margin = object_balanced(
            records,
            f"correct_vs_{mode}",
            bootstrap_samples=int(args.bootstrap_samples),
        )
        control_values = [float(row[f"{mode}_score"]) for row in records]
        role = "train" if mode in train_controls else "heldout"
        controls_report[mode] = {
            "role": role,
            "control_score": summarize(control_values),
            "correct_margin": margin,
        }
        control_checks[f"{mode}_calibrated_negative"] = (
            float(controls_report[mode]["control_score"]["mean"])
            <= float(args.max_control_score)
        )
        control_checks[f"{mode}_margin_mean"] = (
            float(margin["object"]["mean"]) > float(args.min_margin_mean)
        )
        control_checks[f"{mode}_margin_median"] = (
            float(margin["object"]["median"]) > 0.0
        )
        control_checks[f"{mode}_object_win_rate"] = (
            float(margin["object_win_rate"]) >= float(args.min_object_win_rate)
        )
        control_checks[f"{mode}_bootstrap_ci_positive"] = (
            float(margin["object_bootstrap_95_ci"][0]) > 0.0
        )

    checks = {
        "no_sample_failures": not failures and len(records) == count,
        "object_count_matches": len({row["object_uid"] for row in records})
        == len(eval_objects),
        "view_permutation_invariant": permutation_max_abs_diff
        <= float(args.max_permutation_diff),
        "correct_calibrated_positive": float(correct_score["object"]["mean"])
        >= float(args.min_correct_score),
        "every_control_passes": all(control_checks.values()),
        "every_heldout_control_passes": all(
            value
            for name, value in control_checks.items()
            if any(name.startswith(f"{mode}_") for mode in heldout_controls)
        ),
        **control_checks,
    }
    report = {
        "stage": "C0 explicit view correspondence evaluation",
        "passed": all(checks.values()),
        "split_name": args.split_name,
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "cache_manifest": str(dataset.manifest_path.resolve()),
        "cache_config_hash": dataset.config_hash,
        "protocol_hash": model_summary.get("protocol_hash"),
        "train_controls": list(train_controls),
        "heldout_controls": list(heldout_controls),
        "object_count": len({row["object_uid"] for row in records}),
        "sample_count": len(records),
        "view_permutation_max_abs_diff": permutation_max_abs_diff,
        "correct_score": correct_score,
        "controls": controls_report,
        "checks": checks,
        "failures": failures,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "correct_score": correct_score["object"],
                "view_permutation_max_abs_diff": permutation_max_abs_diff,
                "control_margins": {
                    mode: {
                        "role": row["role"],
                        "mean": row["correct_margin"]["object"]["mean"],
                        "win": row["correct_margin"]["object_win_rate"],
                        "ci": row["correct_margin"]["object_bootstrap_95_ci"],
                    }
                    for mode, row in controls_report.items()
                },
            },
            indent=2,
        )
    )
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
