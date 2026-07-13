#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
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

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import _set_compare, _xyz_set  # noqa: E402
from reconvggt_ar_adapter_a.inspect_and_sanity import force_eval  # noqa: E402
from reconvggt_ar_adapter_a.run_b5_ss_cond_residual_smoke import delta_report, summarize_tensor  # noqa: E402
from reconvggt_ar_adapter_a.run_b54_physical_ssgrid_smoke import (  # noqa: E402
    _token_grid_mapping_audit,
    candidate_quality_row,
)
from reconvggt_ar_adapter_a.train_b5_ss_cond_residual_adapter import (  # noqa: E402
    install_dreamsim_stub,
    set_frozen_eval,
)
from reconvggt_ar_adapter_a.train_b55_physical_proxy_adapter import (  # noqa: E402
    FEATURE_INDEX,
    PreparedSession,
    _one_step_logits,
    _prepare_session,
    _rescale_proxy_time,
    _safe_weighted_mean,
    _sample_sparse,
    _stock_trajectory_cache,
)


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


def _parse_ints(spec: str) -> list[int]:
    values = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not values:
        raise ValueError("empty integer list")
    return values


def _parse_floats(spec: str) -> list[float]:
    values = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def _scale_name(scale: float) -> str:
    body = f"{abs(float(scale)):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return ("p" if scale >= 0 else "m") + body


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1.0e-12:
        return 0.0
    return float(torch.dot(x, y).div(denom).cpu().item())


def _normalize_delta(delta: torch.Tensor, cond_base: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    masked = delta.float() * active.float()
    active_channels = active.float().expand_as(masked)
    active_count = active_channels.sum().clamp_min(1.0)
    delta_rms = torch.sqrt((masked * masked).sum() / active_count).clamp_min(1.0e-12)
    cond_rms = torch.sqrt((cond_base.float() * cond_base.float()).mean()).clamp_min(1.0e-12)
    return masked * (cond_rms / delta_rms)


def _gradient_token_stats(gradient: torch.Tensor, session: PreparedSession) -> dict[str, Any]:
    energy = torch.sqrt((gradient.float() * gradient.float()).mean(dim=-1)).reshape(-1)
    out: dict[str, Any] = {}
    for name, key in (("positive", "pos16"), ("negative", "neg16"), ("neutral", "neutral16")):
        weights = session.loss_masks[key].reshape(-1).float()
        out[name] = {
            "token_count": int((weights > 0).sum().item()),
            "mean_rms": float(_safe_weighted_mean(energy, weights).detach().cpu().item()),
            "max_rms": float(energy[weights > 0].max().detach().cpu().item()) if bool((weights > 0).any()) else 0.0,
        }
    return out


def _physical_proxy_gradient(
    *,
    flow,
    decoder,
    sampler,
    session: PreparedSession,
    x_t: torch.Tensor,
    t_model: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    flow.to(session.cond_base.device).eval()
    decoder.to(session.cond_base.device).eval()
    with torch.no_grad():
        base_logits, _ = _one_step_logits(
            flow=flow,
            decoder=decoder,
            sampler=sampler,
            x_t=x_t,
            t_model=float(t_model),
            cond=session.cond_base.float(),
            neg_cond=session.neg_cond,
            cfg_mode=str(args.proxy_cfg_mode),
            cfg_strength=float(args.ss_cfg_strength),
            cfg_interval=(float(args.ss_cfg_interval_min), float(args.ss_cfg_interval_max)),
            guidance_rescale=float(args.ss_guidance_rescale),
            autocast_enabled=(x_t.device.type == "cuda"),
        )
    cond_var = session.cond_base.detach().float().requires_grad_(True)
    logits, _ = _one_step_logits(
        flow=flow,
        decoder=decoder,
        sampler=sampler,
        x_t=x_t,
        t_model=float(t_model),
        cond=cond_var,
        neg_cond=session.neg_cond,
        cfg_mode=str(args.proxy_cfg_mode),
        cfg_strength=float(args.ss_cfg_strength),
        cfg_interval=(float(args.ss_cfg_interval_min), float(args.ss_cfg_interval_max)),
        guidance_rescale=float(args.ss_guidance_rescale),
        autocast_enabled=(x_t.device.type == "cuda"),
    )
    dlogits = logits - base_logits.detach()
    pos_w = session.loss_masks["pos64"]
    neg_w = session.loss_masks["neg64"]
    neutral_w = session.loss_masks["neutral64"]
    pos_loss = _safe_weighted_mean(F.relu(float(args.margin_pos) - dlogits).pow(2), pos_w)
    neg_loss = _safe_weighted_mean(F.relu(dlogits + float(args.margin_neg)).pow(2), neg_w)
    preserve_loss = _safe_weighted_mean(dlogits.pow(2), neutral_w)
    loss = (
        float(args.pos_weight) * pos_loss
        + float(args.neg_weight) * neg_loss
        + float(args.preserve_weight) * preserve_loss
    )
    loss.backward()
    gradient = cond_var.grad.detach().float()
    if not bool(torch.isfinite(gradient).all().item()):
        raise RuntimeError(f"non-finite condition gradient for session={session.name}")
    active = session.loss_masks["active_token"].float()
    oracle_direction = _normalize_delta(-gradient, session.cond_base, active)
    full_features = session.loss_masks["features"]
    surface_gate = full_features[:, FEATURE_INDEX["surface_contrast"]].reshape(1, -1, 1).float()
    denom = (surface_gate * surface_gate).sum(dim=1).clamp_min(1.0)
    fitted_channel = (surface_gate * oracle_direction).sum(dim=1) / denom
    fitted_channel = F.normalize(fitted_channel, dim=-1)
    record_rank1 = _normalize_delta(surface_gate * fitted_channel.unsqueeze(1), session.cond_base, active)
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "pos_loss": float(pos_loss.detach().cpu().item()),
        "neg_loss": float(neg_loss.detach().cpu().item()),
        "preserve_loss": float(preserve_loss.detach().cpu().item()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient).detach().cpu().item()),
        "gradient_abs_max": float(gradient.abs().max().detach().cpu().item()),
        "token_gradient": _gradient_token_stats(gradient, session),
        "oracle_direction": summarize_tensor(oracle_direction),
        "record_rank1_direction": summarize_tensor(record_rank1),
        "record_rank1_fit_cosine": _cosine(oracle_direction * active, record_rank1 * active),
    }
    return gradient.cpu(), {
        "oracle_direction": oracle_direction.cpu(),
        "record_rank1_direction": record_rank1.cpu(),
        "channel_vector": fitted_channel.reshape(-1).cpu(),
        "stats": stats,
    }


def _pairwise_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty gradient-record set")
    within_session = []
    for session_name in sorted({row["session"] for row in records}):
        rows = [row for row in records if row["session"] == session_name]
        for a, b in combinations(rows, 2):
            within_session.append(
                {
                    "session": session_name,
                    "seed_a": a["seed"],
                    "seed_b": b["seed"],
                    "channel_cosine": _cosine(a["channel_vector"], b["channel_vector"]),
                    "full_gradient_cosine": _cosine(a["gradient"], b["gradient"]),
                }
            )
    session_means = {}
    for session_name in sorted({row["session"] for row in records}):
        matrix = torch.stack([row["channel_vector"].float() for row in records if row["session"] == session_name])
        session_means[session_name] = F.normalize(matrix.mean(dim=0), dim=0)
    across_session = []
    for a, b in combinations(sorted(session_means), 2):
        across_session.append({"session_a": a, "session_b": b, "channel_cosine": _cosine(session_means[a], session_means[b])})
    matrix = torch.stack([row["channel_vector"].float() for row in records])
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    variance = singular.square()
    explained = variance / variance.sum().clamp_min(1.0e-12)
    shared = vh[0]
    if float(torch.dot(shared, matrix.mean(dim=0)).item()) < 0:
        shared = -shared
    return {
        "record_count": len(records),
        "session_names": sorted({row["session"] for row in records}),
        "splits": sorted({row["split"] for row in records}),
        "within_session": within_session,
        "within_session_channel_cosine_mean": float(np.mean([r["channel_cosine"] for r in within_session])) if within_session else None,
        "within_session_channel_cosine_min": float(np.min([r["channel_cosine"] for r in within_session])) if within_session else None,
        "within_session_full_gradient_cosine_mean": float(np.mean([r["full_gradient_cosine"] for r in within_session])) if within_session else None,
        "across_session": across_session,
        "across_session_channel_cosine_mean": float(np.mean([r["channel_cosine"] for r in across_session])) if across_session else None,
        "across_session_channel_cosine_min": float(np.min([r["channel_cosine"] for r in across_session])) if across_session else None,
        "singular_values": singular.detach().cpu().tolist(),
        "explained_variance_ratio": explained.detach().cpu().tolist(),
        "rank1_explained_variance": float(explained[0].detach().cpu().item()),
        "rank4_explained_variance": float(explained[:4].sum().detach().cpu().item()),
        "shared_channel": F.normalize(shared, dim=0).detach().cpu(),
    }


def _write_md(path: Path, report: dict[str, Any]) -> None:
    align_train = report["gradient_alignment_train_only"]
    align_all = report["gradient_alignment_all_records"]
    lines = [
        "# B5.9c Frozen Flow-Gradient Channel-Direction Audit",
        "",
        "```text",
        "No adapter training. Frozen bridge / flow / decoder.",
        "All rollout comparisons use the native sampler with an explicitly supplied noise tensor.",
        "Each baseline is repeated without resetting RNG and must be coordinate-identical.",
        "```",
        "",
        "## Train-Only Fit Alignment",
        "",
        f"- records/sessions: `{align_train['record_count']}` / `{align_train['session_names']}`",
        f"- within-session channel cosine mean/min: `{align_train['within_session_channel_cosine_mean']}` / `{align_train['within_session_channel_cosine_min']}`",
        f"- across-train-session channel cosine mean/min: `{align_train['across_session_channel_cosine_mean']}` / `{align_train['across_session_channel_cosine_min']}`",
        f"- rank-1 / rank-4 explained variance: `{align_train['rank1_explained_variance']}` / `{align_train['rank4_explained_variance']}`",
        "",
        "## All-Record Diagnostic Alignment",
        "",
        "This SVD includes validation and is diagnostic only. It never constructs a rollout candidate or controls PASS.",
        "",
        f"- records/sessions: `{align_all['record_count']}` / `{align_all['session_names']}`",
        f"- rank-1 / rank-4 explained variance: `{align_all['rank1_explained_variance']}` / `{align_all['rank4_explained_variance']}`",
        "",
        "## Rollout Candidates",
        "",
        "```text",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"{row['candidate']}: strict={row['strict_pass_count']}/{row['record_count']} "
            f"all={row['all_records_passed']} direction={row['direction_pass_count']} "
            f"max_changed={row['max_changed_ratio']}"
        )
    lines.extend(["```", "", "## Judgment", "", report["judgment"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B5.9c frozen flow-gradient channel-direction audit.")
    parser.add_argument("--sessions_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--gradient_t", type=float, default=0.9)
    parser.add_argument("--gradient_scales", default="0.0025,0.005")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mask_mode", choices=["none", "apply"], default="apply")
    parser.add_argument("--mask_background", choices=["black", "white"], default="black")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--patch_start_idx", type=int, default=5)
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
    parser.add_argument("--ss_cfg_interval_min", type=float, default=0.6)
    parser.add_argument("--ss_cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--ss_rescale_t", type=float, default=3.0)
    parser.add_argument("--ss_grid_side", type=int, default=16)
    parser.add_argument("--sparse_resolution", type=int, default=64)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--physical_frame_scope", choices=["selected", "fullscan"], default="selected")
    parser.add_argument("--evaluation_frame_scope", choices=["selected", "fullscan"], default="fullscan")
    parser.add_argument("--physical_distance_clip", type=float, default=8.0)
    parser.add_argument("--physical_vh_min_visible_views", type=int, default=1)
    parser.add_argument("--physical_vh_min_support_ratio", type=float, default=0.5)
    parser.add_argument("--positive_min_visible_views", type=int, default=1)
    parser.add_argument("--positive_min_support_ratio", type=float, default=0.5)
    parser.add_argument("--negative_min_visible_views", type=int, default=3)
    parser.add_argument("--negative_max_support_ratio", type=float, default=0.1)
    parser.add_argument("--negative_min_outside_ratio", type=float, default=0.9)
    parser.add_argument("--negative_prior_radius_multiplier", type=float, default=1.0)
    parser.add_argument("--loss_mask_mode", choices=["exclusive_surface"], default="exclusive_surface")
    parser.add_argument("--require_nonempty_surface_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_prior_score_positive", action="store_true")
    parser.add_argument("--visual_hull_active_weight", type=float, default=0.25)
    parser.add_argument("--proxy_state_mode", choices=["random_xt", "stock_trajectory"], default="stock_trajectory")
    parser.add_argument("--proxy_cfg_mode", choices=["unguided", "sampler"], default="sampler")
    parser.add_argument("--proxy_rescale_t", type=float, default=3.0)
    parser.add_argument("--proxy_trajectory_steps", type=int, default=12)
    parser.add_argument("--margin_pos", type=float, default=0.003)
    parser.add_argument("--margin_neg", type=float, default=0.001)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--neg_weight", type=float, default=2.0)
    parser.add_argument("--preserve_weight", type=float, default=0.1)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--candidate_min_changed_count", type=int, default=100)
    parser.add_argument("--candidate_min_changed_ratio", type=float, default=0.005)
    parser.add_argument("--candidate_max_changed_ratio", type=float, default=0.10)
    parser.add_argument("--candidate_min_set_iou", type=float, default=0.90)
    parser.add_argument("--candidate_max_absolute_outside_ratio", type=float, default=1.0)
    parser.add_argument("--candidate_require_absolute_outside_nonincrease", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate_max_component_increase", type=int, default=1)
    parser.add_argument("--candidate_min_coord_count_ratio", type=float, default=0.90)
    parser.add_argument("--candidate_max_coord_count_ratio", type=float, default=1.10)
    parser.add_argument("--verify_fixed_noise_repeat", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    mapping = _token_grid_mapping_audit(int(args.ss_grid_side), int(args.sparse_resolution))
    if not mapping["passed"]:
        raise RuntimeError(mapping)
    if args.proxy_state_mode == "stock_trajectory" and args.proxy_cfg_mode != "sampler":
        raise ValueError("stock_trajectory requires sampler CFG")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.sessions_json).read_text(encoding="utf-8"))
    seeds = _parse_ints(args.seeds)
    scales = _parse_floats(args.gradient_scales)
    if not args.load_dreamsim:
        install_dreamsim_stub()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
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
    sessions = [_prepare_session(pipeline=pipeline, args=args, spec=spec, device=device) for spec in specs]
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    set_frozen_eval(flow)
    set_frozen_eval(decoder)
    sampler = pipeline.sparse_structure_sampler

    records: list[dict[str, Any]] = []
    for session_idx, session in enumerate(sessions):
        for seed in seeds:
            torch.manual_seed(int(seed) + session_idx)
            initial_noise = torch.randn(
                int(args.num_samples),
                int(flow.in_channels),
                int(flow.resolution),
                int(flow.resolution),
                int(flow.resolution),
                device=device,
                dtype=torch.float32,
            )
            if args.proxy_state_mode == "stock_trajectory":
                cache = _stock_trajectory_cache(
                    flow=flow,
                    sampler=sampler,
                    session=session,
                    initial_noise=initial_noise,
                    requested_t_values=[float(args.gradient_t)],
                    args=args,
                )[float(args.gradient_t)]
                x_t = cache["x_t"]
                t_model = float(cache["t_model"])
                state_meta = {key: value for key, value in cache.items() if key != "x_t"}
            else:
                x_t = initial_noise
                t_model = _rescale_proxy_time(float(args.gradient_t), float(args.proxy_rescale_t))
                state_meta = {"requested_t_raw": float(args.gradient_t), "t_model": t_model, "trajectory_state_index": None}
            gradient, direction_pack = _physical_proxy_gradient(
                flow=flow,
                decoder=decoder,
                sampler=sampler,
                session=session,
                x_t=x_t,
                t_model=t_model,
                args=args,
            )
            records.append(
                {
                    "session": session.name,
                    "split": session.split,
                    "session_index": session_idx,
                    "seed": int(seed),
                    "gradient": gradient.half(),
                    "oracle_direction": direction_pack["oracle_direction"].half(),
                    "record_rank1_direction": direction_pack["record_rank1_direction"].half(),
                    "channel_vector": direction_pack["channel_vector"].float(),
                    "gradient_stats": direction_pack["stats"],
                    "state_meta": state_meta,
                }
            )
            print(
                f"[B5.9c] gradient session={session.name} seed={seed} "
                f"loss={direction_pack['stats']['loss']:.6g} norm={direction_pack['stats']['gradient_norm']:.6g}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    train_records = [row for row in records if row["split"] == "train"]
    validation_records = [row for row in records if row["split"] == "validation"]
    if not train_records:
        raise RuntimeError("B5.9c requires at least one train gradient record")
    if not validation_records:
        raise RuntimeError("B5.9c requires at least one validation gradient record")
    alignment_all = _pairwise_summary(records)
    shared_all_channel = alignment_all.pop("shared_channel")
    alignment_train = _pairwise_summary(train_records)
    shared_train_channel = alignment_train.pop("shared_channel")
    np.savez_compressed(
        output_dir / "gradient_channel_vectors.npz",
        shared_train_channel=shared_train_channel.numpy(),
        shared_all_diagnostic_channel=shared_all_channel.numpy(),
        channel_vectors=np.stack([row["channel_vector"].numpy() for row in records]),
        sessions=np.asarray([row["session"] for row in records]),
        splits=np.asarray([row["split"] for row in records]),
        seeds=np.asarray([row["seed"] for row in records]),
    )

    eval_rows: list[dict[str, Any]] = []
    fixed_noise_repeats: list[dict[str, Any]] = []
    for row in records:
        session = sessions[int(row["session_index"])]
        seed = int(row["seed"])
        torch.manual_seed(seed + int(row["session_index"]))
        noise = torch.randn(
            int(args.num_samples),
            int(flow.in_channels),
            int(flow.resolution),
            int(flow.resolution),
            int(flow.resolution),
            device=device,
            dtype=torch.float32,
        )
        base_coords, base_report = _sample_sparse(
            pipeline=pipeline,
            flow=flow,
            args=args,
            session=session,
            cond=session.cond_base,
            noise=noise,
            out_dir=output_dir / "eval" / session.name / f"seed{seed}" / "baseline",
        )
        if bool(args.verify_fixed_noise_repeat):
            repeat_coords, _ = _sample_sparse(
                pipeline=pipeline,
                flow=flow,
                args=args,
                session=session,
                cond=session.cond_base,
                noise=noise,
                out_dir=output_dir / "eval" / session.name / f"seed{seed}" / "baseline_repeat",
            )
            repeat_passed = bool(np.array_equal(base_coords, repeat_coords))
            fixed_noise_repeats.append(
                {
                    "session": session.name,
                    "seed": seed,
                    "passed": repeat_passed,
                    "baseline_coord_count": int(base_coords.shape[0]),
                    "repeat_coord_count": int(repeat_coords.shape[0]),
                }
            )
            if not repeat_passed:
                raise RuntimeError(f"fixed-noise repeat failed: session={session.name} seed={seed}")
        active = session.loss_masks["active_token"].cpu().float()
        surface = session.loss_masks["features"][:, FEATURE_INDEX["surface_contrast"]].reshape(1, -1, 1).cpu().float()
        shared_train_delta = _normalize_delta(
            surface * shared_train_channel.reshape(1, 1, -1),
            session.cond_base.cpu(),
            active,
        )
        directions = {
            "oracle_full": row["oracle_direction"].float(),
            "record_rank1": row["record_rank1_direction"].float(),
            "shared_train_rank1": shared_train_delta.float(),
        }
        for direction_name, direction_cpu in directions.items():
            for scale in scales:
                candidate_name = f"{direction_name}_{_scale_name(scale)}"
                delta = direction_cpu.to(device=session.cond_base.device, dtype=torch.float32) * float(scale)
                coords, candidate_report = _sample_sparse(
                    pipeline=pipeline,
                    flow=flow,
                    args=args,
                    session=session,
                    cond=session.cond_base.float() + delta,
                    noise=noise,
                    out_dir=output_dir / "eval" / session.name / f"seed{seed}" / candidate_name,
                )
                candidate_report["delta_vs_baseline"] = delta_report(
                    baseline_coords=base_coords,
                    candidate_coords=coords,
                    prior_sample=session.prior_sample,
                    prior_coords=session.prior_coords,
                    prior_radius=float(args.prior_radius),
                    projection_min_support_views=int(args.projection_min_support_views),
                    projection_min_support_ratio=float(args.projection_min_support_ratio),
                    visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                    visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                    mask_threshold=int(args.mask_threshold),
                )
                candidate_report["set_compare_vs_baseline"] = _set_compare(_xyz_set(base_coords), _xyz_set(coords))
                quality = candidate_quality_row(
                    session_name=session.name,
                    split=session.split,
                    baseline=base_report,
                    candidate=candidate_report,
                    args=args,
                )
                eval_rows.append(
                    {
                        "session": session.name,
                        "split": session.split,
                        "seed": seed,
                        "candidate": candidate_name,
                        "direction_type": direction_name,
                        "scale": float(scale),
                        "quality": quality,
                    }
                )
                print(
                    f"[B5.9c] eval session={session.name} seed={seed} candidate={candidate_name} "
                    f"passed={quality['passed']}",
                    flush=True,
                )

    aggregate = []
    for candidate in sorted({row["candidate"] for row in eval_rows}):
        rows = [row["quality"] for row in eval_rows if row["candidate"] == candidate]
        by_split = {}
        for split in sorted({row["split"] for row in rows}):
            split_rows = [row for row in rows if row["split"] == split]
            by_split[split] = {
                "record_count": len(split_rows),
                "strict_pass_count": int(sum(bool(row["passed"]) for row in split_rows)),
                "direction_pass_count": int(sum(bool(row["direction_passed"]) for row in split_rows)),
                "all_records_passed": bool(split_rows) and all(bool(row["passed"]) for row in split_rows),
            }
        aggregate.append(
            {
                "candidate": candidate,
                "record_count": len(rows),
                "strict_pass_count": int(sum(bool(row["passed"]) for row in rows)),
                "direction_pass_count": int(sum(bool(row["direction_passed"]) for row in rows)),
                "all_records_passed": bool(rows) and all(bool(row["passed"]) for row in rows),
                "mean_changed_ratio": float(np.mean([row["changed_ratio"] for row in rows])) if rows else None,
                "max_changed_ratio": float(np.max([row["changed_ratio"] for row in rows])) if rows else None,
                "by_split": by_split,
                "per_record": rows,
            }
        )
    aggregate.sort(key=lambda row: (row["all_records_passed"], row["strict_pass_count"], row["direction_pass_count"]), reverse=True)
    shared_train_pass = any(
        row["all_records_passed"] and row["candidate"].startswith("shared_train_rank1")
        for row in aggregate
    )
    record_pass = any(row["all_records_passed"] and row["candidate"].startswith("record_rank1") for row in aggregate)
    oracle_pass = any(row["all_records_passed"] and row["candidate"].startswith("oracle_full") for row in aggregate)
    if shared_train_pass:
        judgment = "PASS: the rank-1 direction fitted only from train gradients survives every train/validation seed under fixed-noise rollout. B5.10 adapter learning is justified."
    elif record_pass or oracle_pass:
        judgment = "PARTIAL: per-record/oracle gradient works but the train-only shared rank-1 direction does not generalize across all records. Do not train a global adapter yet; inspect higher rank or synthetic supervision."
    else:
        judgment = "FAIL: even flow-gradient oracle directions do not pass all fixed-noise rollouts. Stop SS-condition residual learning and move to direct occupancy/coordinate supervision."
    report_records = [
        {
            key: value
            for key, value in row.items()
            if key not in {"gradient", "oracle_direction", "record_rank1_direction", "channel_vector"}
        }
        for row in records
    ]
    report = {
        "args": vars(args),
        "mapping_audit": mapping,
        "sampling_backend": "native_pipeline_supplied_noise",
        "sampler_class": sampler.__class__.__name__,
        "fixed_noise_repeats": fixed_noise_repeats,
        "shared_direction_fit_policy": {
            "fit_split": "train",
            "fit_record_count": len(train_records),
            "fit_sessions": sorted({row["session"] for row in train_records}),
            "validation_gradient_used_for_fit": False,
            "evaluation_splits": sorted({row["split"] for row in records}),
        },
        "records": report_records,
        "gradient_alignment_train_only": alignment_train,
        "gradient_alignment_all_records": alignment_all,
        "eval_rows": eval_rows,
        "aggregate": aggregate,
        "judgment": judgment,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_md(output_dir / "report.md", report)
    print(f"[B5.9c] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
