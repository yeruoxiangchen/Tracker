#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import (  # noqa: E402
    _load_prior_manifest_sample,
    _set_compare,
    _xyz_set,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import (  # noqa: E402
    force_eval,
    load_images,
    normalize_image_cond,
)
from reconvggt_ar_adapter_a.projection_token_features import (  # noqa: E402
    parse_ar_pose_file,
    select_pose_records,
    summarize_pose_features,
)
from reconvggt_ar_adapter_a.run_b3_adapter_injection_smoke import (  # noqa: E402
    _component_stats,
    _load_images_with_masks,
)
from reconvggt_ar_adapter_a.run_b5_ss_cond_residual_smoke import (  # noqa: E402
    build_ar_cond_residual,
    coords_np,
    delta_report,
    summarize_tensor,
)
from reconvggt_ar_adapter_a.train_b5_ss_cond_residual_adapter import (  # noqa: E402
    SSCondResidualAdapter,
    adapter_stats,
    delta_norm_loss,
    install_dreamsim_stub,
    load_adapter_if_needed,
    save_checkpoint,
    set_frozen_eval,
)
from reconvggt_ar_adapter_a.train_b_projection_adapter import build_projection_features  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import sparse_diagnostic_metrics  # noqa: E402


@dataclass
class PreparedSession:
    name: str
    split: str
    image_names: list[str]
    mask_summaries: Any
    cond_base: torch.Tensor
    neg_cond: torch.Tensor
    ar_cond_encoding: torch.Tensor
    target_delta: torch.Tensor
    target_cond: torch.Tensor
    point_projection_summary: dict[str, Any] | None
    point_projection_source: str | None
    projection_feature_summary: dict[str, Any]
    residual_build: dict[str, Any]
    prior_sample: dict[str, Any] | None
    prior_coords: np.ndarray | None
    prior_summary: dict[str, Any] | None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def _session_arg_namespace(args: argparse.Namespace, spec: dict[str, Any]) -> argparse.Namespace:
    out = copy.copy(args)
    out.image_dir = spec["image_dir"]
    out.pose_file = spec["pose_file"]
    out.mask_dir = spec.get("mask_dir", "")
    out.colmap_sparse_dir = spec.get("colmap_sparse_dir", "")
    out.points3d_txt = spec.get("points3d_txt", "")
    out.point_prior_npz = spec.get("point_prior_npz", "")
    return out


def _prepare_session(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    args: argparse.Namespace,
    spec: dict[str, Any],
    device: torch.device,
) -> PreparedSession:
    session_args = _session_arg_namespace(args, spec)
    name = str(spec["name"])
    split = str(spec.get("split", "train"))
    print(f"[B5.3] preparing session name={name} split={split}", flush=True)

    if args.mask_mode == "apply":
        if not session_args.mask_dir:
            raise ValueError(f"Session {name} needs mask_dir when --mask_mode=apply")
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(session_args.image_dir),
            mask_dir=Path(session_args.mask_dir),
            max_views=int(args.max_views),
            mask_background=args.mask_background,
        )
    else:
        images, image_names = load_images(
            Path(session_args.image_dir),
            max_views=int(args.max_views),
            preprocess=args.preprocess,
            pipeline=pipeline,
        )
        mask_summaries = None

    pose_records_all = parse_ar_pose_file(
        session_args.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    pose_records = select_pose_records(image_names, pose_records_all)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, _ = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)
    ss_image_cond = image_cond[:, :, int(args.patch_start_idx) :]

    projection_features, _point_features, point_projection_summary, point_projection_source = build_projection_features(
        args=session_args,
        image_names=image_names,
        pose_records=pose_records,
        token_device=aggregated_tokens_list[0].device,
        token_dtype=torch.float32,
    )
    projection_features = projection_features.float()

    with torch.no_grad():
        ss_cond_base = pipeline.get_ss_cond(ss_image_cond, aggregated_tokens_list, int(args.num_samples))
    cond_base = ss_cond_base["cond"].detach()
    neg_cond = ss_cond_base["neg_cond"].detach()
    ar_cond_encoding, residual_build = build_ar_cond_residual(
        projection_features,
        cond_base,
        seed=int(args.residual_seed),
        token_mode=str(args.residual_token_mode),
    )
    projection_feature_summary = summarize_pose_features(projection_features)
    target_delta = ar_cond_encoding.detach() * float(args.target_scale)
    target_cond = (cond_base + target_delta).detach()

    prior_sample = prior_coords = prior_summary = None
    prior_manifest = str(spec.get("prior_manifest", "") or "")
    if prior_manifest:
        prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(
            Path(prior_manifest),
            str(spec.get("prior_uid", "") or ""),
        )

    # Drop large token tensors early; only cond-space tensors are needed after preparation.
    del aggregated_tokens_list, raw_image_cond, image_cond, ss_image_cond, projection_features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[B5.3] prepared {name}: views={len(image_names)} cond={tuple(cond_base.shape)} "
        f"prior={'yes' if prior_sample is not None else 'no'}",
        flush=True,
    )
    return PreparedSession(
        name=name,
        split=split,
        image_names=image_names,
        mask_summaries=mask_summaries,
        cond_base=cond_base,
        neg_cond=neg_cond,
        ar_cond_encoding=ar_cond_encoding,
        target_delta=target_delta,
        target_cond=target_cond,
        point_projection_summary=point_projection_summary,
        point_projection_source=point_projection_source,
        projection_feature_summary=projection_feature_summary,
        residual_build=residual_build,
        prior_sample=prior_sample,
        prior_coords=prior_coords,
        prior_summary=prior_summary,
    )


def _evaluate_session(
    *,
    pipeline: TrellisVGGTTo3DPipeline,
    flow,
    adapter: SSCondResidualAdapter,
    args: argparse.Namespace,
    session: PreparedSession,
    output_dir: Path,
    seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter.eval()
    with torch.no_grad():
        adapter_delta = adapter(session.cond_base, session.ar_cond_encoding)
    eval_specs = {
        "baseline": session.cond_base,
        "target_residual": session.target_cond,
        "adapter_residual": session.cond_base + adapter_delta.detach(),
    }
    ss_sampler_params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_cfg_strength),
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    torch.manual_seed(int(args.seed) + int(seed_offset))
    ss_noise = torch.randn(
        int(args.num_samples),
        int(flow.in_channels),
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        device=session.cond_base.device,
    )

    coords_by_name: dict[str, np.ndarray] = {}
    eval_reports: dict[str, Any] = {}
    for name, cond in eval_specs.items():
        spec_dir = output_dir / name
        spec_dir.mkdir(parents=True, exist_ok=True)
        ss_cond = {"cond": cond.to(dtype=session.cond_base.dtype), "neg_cond": session.neg_cond}
        print(f"[B5.3] sampling session={session.name} source={name}", flush=True)
        coords = pipeline.sample_sparse_structure(ss_cond, int(args.num_samples), ss_sampler_params, noise=ss_noise.clone())
        coords_array = coords_np(coords)
        coords_by_name[name] = coords_array
        np.savez_compressed(spec_dir / "coords.npz", coords=coords_array)
        prior_alignment = None
        if session.prior_sample is not None and session.prior_coords is not None:
            prior_alignment = sparse_diagnostic_metrics(
                "b53_sparse",
                coords_array,
                session.prior_coords,
                session.prior_sample,
                prior_radius=float(args.prior_radius),
                min_support_views=int(args.projection_min_support_views),
                min_support_ratio=float(args.projection_min_support_ratio),
                visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                grid_resolution=64,
                mask_threshold=int(args.mask_threshold),
            )
        eval_reports[name] = {
            "output_dir": str(spec_dir),
            "sparse": _component_stats(coords_array),
            "prior_alignment": prior_alignment,
        }

    baseline_coords = coords_by_name["baseline"]
    for name in ["target_residual", "adapter_residual"]:
        eval_reports[name]["delta_vs_baseline"] = delta_report(
            baseline_coords=baseline_coords,
            candidate_coords=coords_by_name[name],
            prior_sample=session.prior_sample,
            prior_coords=session.prior_coords,
            prior_radius=float(args.prior_radius),
            projection_min_support_views=int(args.projection_min_support_views),
            projection_min_support_ratio=float(args.projection_min_support_ratio),
            visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
            visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
            mask_threshold=int(args.mask_threshold),
        )
    eval_reports["adapter_vs_target_set_compare"] = _set_compare(
        _xyz_set(coords_by_name["target_residual"]),
        _xyz_set(coords_by_name["adapter_residual"]),
    )
    final_stats = adapter_stats(adapter_delta, session.target_delta)
    return {
        "name": session.name,
        "split": session.split,
        "image_names": session.image_names,
        "mask_summaries": session.mask_summaries,
        "adapter_stats": final_stats,
        "cond_base": summarize_tensor(session.cond_base),
        "target_delta": summarize_tensor(session.target_delta),
        "prior_summary": session.prior_summary,
        "point_projection_summary": session.point_projection_summary,
        "residual_build": session.residual_build,
        "eval": eval_reports,
        "pass": _judge_session(eval_reports),
    }


def _judge_session(eval_reports: dict[str, Any]) -> dict[str, Any]:
    cmp_row = eval_reports.get("adapter_vs_target_set_compare") or {}
    adapter_delta = (eval_reports.get("adapter_residual") or {}).get("delta_vs_baseline") or {}
    direction = adapter_delta.get("direction_summary") or {}
    sparse = (eval_reports.get("adapter_residual") or {}).get("sparse") or {}
    checks = {
        "adapter_vs_target_iou_ge_0p93": float(cmp_row.get("iou") or 0.0) >= 0.93,
        "mask_direction_gt_0p30": float(direction.get("added_minus_removed_projection_any_mask_hit_ratio") or 0.0) > 0.30,
        "outside_direction_le_m0p15": float(direction.get("added_minus_removed_visible_outside_mask_event_ratio") or 0.0) <= -0.15,
        "component_le_7": float(sparse.get("component_count") or 1.0e9) <= 7.0,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "adapter_vs_target_iou": cmp_row.get("iou"),
        "mask_direction": direction.get("added_minus_removed_projection_any_mask_hit_ratio"),
        "outside_direction": direction.get("added_minus_removed_visible_outside_mask_event_ratio"),
        "component_count": sparse.get("component_count"),
    }


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# B5.3 Multi-Session SS Cond Residual",
        "",
        "## Setup",
        "",
        "```text",
        f"output_dir = {report['output_dir']}",
        f"target_scale = {report['args']['target_scale']}",
        f"residual_seed = {report['args']['residual_seed']}",
        f"max_steps = {report['args']['max_steps']}",
        f"lr = {report['args']['lr']}",
        f"delta_norm_target_ratio = {report['args']['delta_norm_target_ratio']}",
        f"delta_norm_weight = {report['args']['delta_norm_weight']}",
        "flow_proxy_weight = 0",
        "x0_proxy_weight = 0",
        "```",
        "",
        "## Sessions",
        "",
        "```text",
    ]
    for sess in report["sessions"]:
        lines.append(f"{sess['name']} split={sess['split']} views={len(sess['image_names'])}")
    lines.extend(["```", "", "## Train", "", "```text"])
    for row in report["rows"]:
        lines.append(
            f"step={row['step']} loss={row['loss']:.6g} mimic={row['mimic_loss']:.6g} "
            f"norm={row['delta_norm_ratio_mean']:.5f} cos={row['delta_target_cosine']:.5f}"
        )
    lines.extend(["```", "", "## Eval", "", "```text"])
    for sess in report["sessions"]:
        evals = sess["eval"]
        cmp_row = evals.get("adapter_vs_target_set_compare") or {}
        direction = ((evals.get("adapter_residual") or {}).get("delta_vs_baseline") or {}).get("direction_summary") or {}
        sparse = (evals.get("adapter_residual") or {}).get("sparse") or {}
        target_dir = ((evals.get("target_residual") or {}).get("delta_vs_baseline") or {}).get("direction_summary") or {}
        lines.extend(
            [
                f"{sess['name']} [{sess['split']}]:",
                f"  adapter_vs_target_iou = {cmp_row.get('iou')}",
                f"  adapter_component = {sparse.get('component_count')}",
                f"  adapter_mask_direction = {direction.get('added_minus_removed_projection_any_mask_hit_ratio')}",
                f"  adapter_outside_direction = {direction.get('added_minus_removed_visible_outside_mask_event_ratio')}",
                f"  target_mask_direction = {target_dir.get('added_minus_removed_projection_any_mask_hit_ratio')}",
                f"  target_outside_direction = {target_dir.get('added_minus_removed_visible_outside_mask_event_ratio')}",
                f"  passed = {sess['pass'].get('passed')} checks={sess['pass'].get('checks')}",
                "",
            ]
        )
    lines.extend(["```", "", "## Judgment", "", report["judgment"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B5.3 multi-session SS condition residual adapter smoke.")
    parser.add_argument("--sessions_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mask_mode", choices=["none", "apply"], default="apply")
    parser.add_argument("--mask_background", choices=["black", "white"], default="black")
    parser.add_argument("--mask_projection_mode", choices=["none", "filter_points", "token_mask", "filter_points_token_mask"], default="filter_points_token_mask")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--token_mask_min_ratio", type=float, default=0.10)
    parser.add_argument("--point_mask_support_min_views", type=int, default=2)
    parser.add_argument("--point_mask_support_min_ratio", type=float, default=0.50)
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--token_grid_side", type=int, default=37)
    parser.add_argument("--point_projection_rotation_mode", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--point_projection_min_depth", type=float, default=1.0e-4)
    parser.add_argument("--default_fx", type=float, default=485.845947)
    parser.add_argument("--default_fy", type=float, default=485.744232)
    parser.add_argument("--default_cx", type=float, default=322.973236)
    parser.add_argument("--default_cy", type=float, default=237.599487)
    parser.add_argument("--default_image_width", type=int, default=640)
    parser.add_argument("--default_image_height", type=int, default=480)
    parser.add_argument("--load_dreamsim", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_cfg_strength", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--ss_rescale_t", type=float, default=3.0)
    parser.add_argument("--target_scale", type=float, default=-0.02)
    parser.add_argument("--residual_seed", type=int, default=20260707)
    parser.add_argument("--residual_token_mode", choices=["constant", "random", "ramp"], default="random")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--mimic_weight", type=float, default=1.0)
    parser.add_argument("--delta_norm_weight", type=float, default=0.02)
    parser.add_argument("--delta_norm_target_ratio", type=float, default=1.05)
    parser.add_argument("--delta_norm_loss_mode", choices=["over", "target"], default="target")
    parser.add_argument("--resume_adapter", default="")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--points3d_txt", default="")
    parser.add_argument("--point_prior_npz", default="")
    parser.add_argument("--colmap_sparse_dir", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sessions_spec = json.loads(Path(args.sessions_json).read_text(encoding="utf-8"))
    if not isinstance(sessions_spec, list) or not sessions_spec:
        raise ValueError("--sessions_json must contain a non-empty list")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if not args.load_dreamsim:
        install_dreamsim_stub()

    print(f"[B5.3] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = bool(args.low_vram)
    force_eval(pipeline)
    if hasattr(pipeline, "birefnet_model") and not pipeline.low_vram:
        pipeline.birefnet_model.to(device)
    if not pipeline.low_vram:
        for model in pipeline.models.values():
            model.to(device)
        pipeline.VGGT_model.to(device)
    force_eval(pipeline)

    torch.manual_seed(int(args.seed))
    sessions = [_prepare_session(pipeline=pipeline, args=args, spec=spec, device=device) for spec in sessions_spec]
    train_sessions = [s for s in sessions if s.split == "train"]
    if not train_sessions:
        raise ValueError("At least one session must have split='train'")

    channels = int(train_sessions[0].cond_base.shape[-1])
    adapter = SSCondResidualAdapter(channels=channels, hidden_dim=int(args.hidden_dim)).to(train_sessions[0].cond_base.device)
    loaded_adapter = load_adapter_if_needed(adapter, args.resume_adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=0.0)

    rows: list[dict[str, Any]] = []
    adapter.train()
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=train_sessions[0].cond_base.device, dtype=torch.float32)
        row_parts = []
        for session in train_sessions:
            delta = adapter(session.cond_base, session.ar_cond_encoding)
            mimic_loss = F.mse_loss(delta.float(), session.target_delta.float())
            norm_loss, norm_ratio = delta_norm_loss(
                delta,
                session.target_delta,
                target_ratio=float(args.delta_norm_target_ratio),
                mode=str(args.delta_norm_loss_mode),
            )
            loss = float(args.mimic_weight) * mimic_loss + float(args.delta_norm_weight) * norm_loss
            total_loss = total_loss + loss
            stats = adapter_stats(delta.detach(), session.target_delta)
            row_parts.append(
                {
                    "session": session.name,
                    "loss": float(loss.detach().cpu().item()),
                    "mimic_loss": float(mimic_loss.detach().cpu().item()),
                    "delta_norm_loss": float(norm_loss.detach().cpu().item()),
                    "delta_norm_ratio_mean": float(norm_ratio.mean().cpu().item()),
                    "delta_target_cosine": stats["cosine_mean"],
                    "delta_target_norm_ratio": stats["norm_ratio_mean"],
                }
            )
        total_loss = total_loss / float(len(train_sessions))
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            row = {
                "step": int(step),
                "loss": float(total_loss.detach().cpu().item()),
                "mimic_loss": float(np.mean([x["mimic_loss"] for x in row_parts])),
                "delta_norm_loss": float(np.mean([x["delta_norm_loss"] for x in row_parts])),
                "delta_norm_ratio_mean": float(np.mean([x["delta_norm_ratio_mean"] for x in row_parts])),
                "delta_target_cosine": float(np.mean([x["delta_target_cosine"] for x in row_parts])),
                "sessions": row_parts,
            }
            rows.append(row)
            print(
                f"[B5.3] step={step} loss={row['loss']:.6g} mimic={row['mimic_loss']:.6g} "
                f"norm={row['delta_norm_ratio_mean']:.5f} cos={row['delta_target_cosine']:.5f}",
                flush=True,
            )
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            save_checkpoint(ckpt_dir / f"adapter_step_{step:06d}.ckpt", adapter, args, rows)

    final_ckpt = ckpt_dir / "last.ckpt"
    save_checkpoint(final_ckpt, adapter, args, rows)

    flow = pipeline.models["sparse_structure_flow_model"].to(train_sessions[0].cond_base.device).eval()
    set_frozen_eval(flow)
    session_reports = []
    for idx, session in enumerate(sessions):
        session_reports.append(
            _evaluate_session(
                pipeline=pipeline,
                flow=flow,
                adapter=adapter,
                args=args,
                session=session,
                output_dir=output_dir / "eval" / session.name,
                seed_offset=idx,
            )
        )

    train_pass = all(s["pass"]["passed"] for s in session_reports if s["split"] == "train")
    holdout_pass = all(s["pass"]["passed"] for s in session_reports if s["split"] != "train")
    judgment = (
        "B5.3 strong-teacher residual passes both train and holdout under current strict criteria."
        if train_pass and holdout_pass
        else "B5.3 does not pass the multi-session strict criterion; treat B5.2 as single-session behavior until expanded training is fixed."
    )
    report = {
        "args": vars(args),
        "output_dir": str(output_dir),
        "checkpoint": str(final_ckpt),
        "loaded_adapter": loaded_adapter,
        "adapter": adapter.metadata(),
        "rows": rows,
        "sessions": session_reports,
        "judgment": judgment,
        "scope": "B5.3 multi-session SS-only residual adapter; frozen ReconViaGen bridge, frozen sparse flow, SLAT not touched.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_md(output_dir / "report.md", report)
    print(f"[B5.3] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
