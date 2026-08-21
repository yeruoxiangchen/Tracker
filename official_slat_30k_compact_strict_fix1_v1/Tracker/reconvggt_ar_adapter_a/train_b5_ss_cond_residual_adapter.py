#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
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

import trellis.pipelines.trellis_image_to_3d as trellis_image_to_3d  # noqa: E402
from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import _set_compare, _xyz_set  # noqa: E402
from reconvggt_ar_adapter_a.inspect_and_sanity import (  # noqa: E402
    DreamSimStub,
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
from reconvggt_ar_adapter_a.train_b_projection_adapter import build_projection_features  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import sparse_diagnostic_metrics  # noqa: E402


class SSCondResidualAdapter(nn.Module):
    """Zero-init residual adapter after sparse_structure_vggt_cond.

    Input is deliberately cond-shaped:

      concat(cond_base, ar_cond_encoding) -> delta_cond

    This keeps the original ReconViaGen bridge frozen and learns only a small
    residual in the SS condition space.
    """

    def __init__(self, channels: int = 1024, hidden_dim: int = 256) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.channels * 2),
            nn.Linear(self.channels * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.channels),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond_base: torch.Tensor, ar_cond_encoding: torch.Tensor) -> torch.Tensor:
        if cond_base.shape != ar_cond_encoding.shape:
            raise ValueError(
                f"cond_base and ar_cond_encoding shape mismatch: "
                f"{tuple(cond_base.shape)} vs {tuple(ar_cond_encoding.shape)}"
            )
        x = torch.cat((cond_base.float(), ar_cond_encoding.float()), dim=-1)
        return self.net(x).to(dtype=cond_base.dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "channels": self.channels,
            "hidden_dim": self.hidden_dim,
            "zero_init_last_layer": True,
            "input": "concat(cond_base, ar_cond_encoding)",
        }


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def parse_t_values(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError("t_values is empty")
    return vals


def set_frozen_eval(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def save_checkpoint(path: Path, adapter: SSCondResidualAdapter, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "metadata": adapter.metadata(),
            "args": vars(args),
            "rows": rows,
        },
        path,
    )


def load_adapter_if_needed(adapter: SSCondResidualAdapter, path: str | Path) -> dict[str, Any] | None:
    if not path:
        return None
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Adapter checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "path": str(path),
        "metadata": ckpt.get("metadata"),
        "rows": ckpt.get("rows", [])[-3:],
    }


def adapter_stats(delta: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    d = delta.detach().float()
    t = target.detach().float()
    mse = F.mse_loss(d, t).item()
    l1 = torch.mean(torch.abs(d - t)).item()
    d_flat = d.reshape(d.shape[0], -1)
    t_flat = t.reshape(t.shape[0], -1)
    cos = F.cosine_similarity(d_flat, t_flat, dim=-1)
    norm_ratio = d_flat.norm(dim=-1) / t_flat.norm(dim=-1).clamp_min(1.0e-8)
    return {
        "mse": float(mse),
        "l1": float(l1),
        "cosine_mean": float(cos.mean().item()),
        "norm_ratio_mean": float(norm_ratio.mean().item()),
        "delta": summarize_tensor(delta),
        "target": summarize_tensor(target),
    }


def delta_norm_loss(
    delta: torch.Tensor,
    target: torch.Tensor,
    target_ratio: float = 1.0,
    mode: str = "over",
) -> tuple[torch.Tensor, torch.Tensor]:
    d_flat = delta.float().reshape(delta.shape[0], -1)
    t_flat = target.float().reshape(target.shape[0], -1)
    norm_ratio = d_flat.norm(dim=-1) / t_flat.norm(dim=-1).clamp_min(1.0e-8)
    ratio_target = norm_ratio.new_full(norm_ratio.shape, float(target_ratio))
    if mode == "target":
        loss = (norm_ratio - ratio_target).pow(2).mean()
    elif mode == "over":
        loss = torch.relu(norm_ratio - ratio_target).pow(2).mean()
    else:
        raise ValueError(f"Unknown delta_norm_loss_mode: {mode}")
    return loss, norm_ratio.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="B5.2 train zero-init SS condition residual adapter.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--mask_mode", choices=["none", "apply"], default="none")
    parser.add_argument("--mask_background", choices=["black", "white"], default="black")
    parser.add_argument("--points3d_txt", default="")
    parser.add_argument("--point_prior_npz", default="")
    parser.add_argument("--colmap_sparse_dir", default="")
    parser.add_argument("--mask_projection_mode", choices=["none", "filter_points", "token_mask", "filter_points_token_mask"], default="none")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--token_mask_min_ratio", type=float, default=0.05)
    parser.add_argument("--point_mask_support_min_views", type=int, default=0)
    parser.add_argument("--point_mask_support_min_ratio", type=float, default=0.0)
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
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--mimic_weight", type=float, default=1.0)
    parser.add_argument("--delta_norm_weight", type=float, default=0.0)
    parser.add_argument("--delta_norm_target_ratio", type=float, default=1.0)
    parser.add_argument("--delta_norm_loss_mode", choices=["over", "target"], default="over")
    parser.add_argument("--flow_proxy_weight", type=float, default=0.0)
    parser.add_argument("--x0_proxy_weight", type=float, default=0.0)
    parser.add_argument("--t_values", default="0.5,0.75,0.9")
    parser.add_argument("--resume_adapter", default="")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--prior_manifest", default="")
    parser.add_argument("--prior_uid", default="")
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if not args.load_dreamsim:
        install_dreamsim_stub()

    print(f"[B5.2] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
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

    if args.mask_mode == "apply":
        if not args.mask_dir:
            raise ValueError("--mask_dir is required when --mask_mode=apply")
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(args.image_dir),
            mask_dir=Path(args.mask_dir),
            max_views=int(args.max_views),
            mask_background=args.mask_background,
        )
    else:
        images, image_names = load_images(Path(args.image_dir), max_views=int(args.max_views), preprocess=args.preprocess, pipeline=pipeline)
        mask_summaries = None

    pose_records_all = parse_ar_pose_file(
        args.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    pose_records = select_pose_records(image_names, pose_records_all)
    print(f"[B5.2] loaded {len(images)} images and {len(pose_records)} poses", flush=True)

    torch.manual_seed(int(args.seed))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, _ = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)
    ss_image_cond = image_cond[:, :, int(args.patch_start_idx) :]

    projection_features, point_features, point_projection_summary, point_projection_source = build_projection_features(
        args=args,
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
    target_delta = ar_cond_encoding.detach() * float(args.target_scale)
    target_cond = (cond_base + target_delta).detach()

    adapter = SSCondResidualAdapter(channels=int(cond_base.shape[-1]), hidden_dim=int(args.hidden_dim)).to(cond_base.device)
    loaded_adapter = load_adapter_if_needed(adapter, args.resume_adapter)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=0.0)

    flow = pipeline.models["sparse_structure_flow_model"].to(cond_base.device).eval()
    set_frozen_eval(flow)
    sampler = pipeline.sparse_structure_sampler
    t_values = parse_t_values(args.t_values)
    proxy_noise = torch.randn(
        int(args.num_samples),
        int(flow.in_channels),
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        device=cond_base.device,
        dtype=torch.float32,
    )

    rows: list[dict[str, Any]] = []
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = adapter(cond_base, ar_cond_encoding)
        mimic_loss = F.mse_loss(delta.float(), target_delta.float())
        norm_loss, norm_ratio = delta_norm_loss(
            delta,
            target_delta,
            target_ratio=float(args.delta_norm_target_ratio),
            mode=str(args.delta_norm_loss_mode),
        )
        flow_loss = delta.new_tensor(0.0).float()
        x0_loss = delta.new_tensor(0.0).float()
        if float(args.flow_proxy_weight) > 0.0 or float(args.x0_proxy_weight) > 0.0:
            t = float(t_values[(step - 1) % len(t_values)])
            t_tensor = torch.tensor([1000.0 * t] * int(args.num_samples), device=cond_base.device, dtype=torch.float32)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(cond_base.device.type == "cuda")):
                teacher_v = flow(proxy_noise, t_tensor, target_cond)
                teacher_x0 = sampler._pred_to_xstart(proxy_noise, t, teacher_v)
            with torch.cuda.amp.autocast(enabled=(cond_base.device.type == "cuda")):
                student_cond = cond_base + delta
                student_v = flow(proxy_noise, t_tensor, student_cond)
                student_x0 = sampler._pred_to_xstart(proxy_noise, t, student_v)
            flow_loss = F.mse_loss(student_v.float(), teacher_v.float())
            x0_loss = F.mse_loss(student_x0.float(), teacher_x0.float())
        loss = (
            float(args.mimic_weight) * mimic_loss
            + float(args.delta_norm_weight) * norm_loss
            + float(args.flow_proxy_weight) * flow_loss
            + float(args.x0_proxy_weight) * x0_loss
        )
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            stats = adapter_stats(delta, target_delta)
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "mimic_loss": float(mimic_loss.detach().cpu().item()),
                "delta_norm_loss": float(norm_loss.detach().cpu().item()),
                "delta_norm_ratio_mean": float(norm_ratio.mean().cpu().item()),
                "flow_proxy_loss": float(flow_loss.detach().cpu().item()),
                "x0_proxy_loss": float(x0_loss.detach().cpu().item()),
                "delta_target_mse": stats["mse"],
                "delta_target_l1": stats["l1"],
                "delta_target_cosine": stats["cosine_mean"],
                "delta_target_norm_ratio": stats["norm_ratio_mean"],
            }
            rows.append(row)
            print(
                "[B5.2] "
                f"step={step} loss={row['loss']:.6g} mimic={row['mimic_loss']:.6g} "
                f"norm_loss={row['delta_norm_loss']:.6g} norm={row['delta_norm_ratio_mean']:.5f} "
                f"flow={row['flow_proxy_loss']:.6g} x0={row['x0_proxy_loss']:.6g} "
                f"cos={row['delta_target_cosine']:.5f}",
                flush=True,
            )
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            save_checkpoint(ckpt_dir / f"adapter_step_{step:06d}.ckpt", adapter, args, rows)

    adapter.eval()
    with torch.no_grad():
        adapter_delta = adapter(cond_base, ar_cond_encoding)
    final_adapter_stats = adapter_stats(adapter_delta, target_delta)
    final_ckpt = ckpt_dir / "last.ckpt"
    save_checkpoint(final_ckpt, adapter, args, rows)

    prior_sample = prior_coords = prior_summary = None
    if args.prior_manifest:
        from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import _load_prior_manifest_sample

        prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(Path(args.prior_manifest), args.prior_uid)

    ss_sampler_params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_cfg_strength),
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    torch.manual_seed(int(args.seed))
    ss_noise = torch.randn(
        int(args.num_samples),
        flow.in_channels,
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        device=cond_base.device,
    )

    eval_specs = {
        "baseline": cond_base,
        "target_residual": target_cond,
        "adapter_residual": cond_base + adapter_delta.detach(),
    }
    eval_reports: dict[str, Any] = {}
    coords_by_name: dict[str, np.ndarray] = {}
    for name, cond in eval_specs.items():
        out_dir = output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ss_cond = {"cond": cond.to(dtype=cond_base.dtype), "neg_cond": neg_cond}
        print(f"[B5.2] sampling {name}", flush=True)
        coords = pipeline.sample_sparse_structure(ss_cond, int(args.num_samples), ss_sampler_params, noise=ss_noise.clone())
        coords_array = coords_np(coords)
        coords_by_name[name] = coords_array
        np.savez_compressed(out_dir / "coords.npz", coords=coords_array)
        prior_alignment = None
        if prior_sample is not None and prior_coords is not None:
            prior_alignment = sparse_diagnostic_metrics(
                "b52_sparse",
                coords_array,
                prior_coords,
                prior_sample,
                prior_radius=float(args.prior_radius),
                min_support_views=int(args.projection_min_support_views),
                min_support_ratio=float(args.projection_min_support_ratio),
                visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                grid_resolution=64,
                mask_threshold=int(args.mask_threshold),
            )
        eval_reports[name] = {
            "output_dir": str(out_dir),
            "sparse": _component_stats(coords_array),
            "prior_alignment": prior_alignment,
        }

    baseline_coords = coords_by_name["baseline"]
    for name in ["target_residual", "adapter_residual"]:
        eval_reports[name]["delta_vs_baseline"] = delta_report(
            baseline_coords=baseline_coords,
            candidate_coords=coords_by_name[name],
            prior_sample=prior_sample,
            prior_coords=prior_coords,
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

    report = {
        "args": vars(args),
        "scope": "B5.2 SS-only residual adapter; frozen VGGT/sparse_structure_vggt_cond/sparse flow; SLAT not touched",
        "image_names": image_names,
        "mask_summaries": mask_summaries,
        "loaded_adapter": loaded_adapter,
        "adapter": adapter.metadata(),
        "checkpoint": str(final_ckpt),
        "rows": rows,
        "final_adapter_stats": final_adapter_stats,
        "cond_base": summarize_tensor(cond_base),
        "ar_cond_encoding": summarize_tensor(ar_cond_encoding),
        "target_delta": summarize_tensor(target_delta),
        "residual_build": residual_build,
        "point_projection_source": point_projection_source,
        "point_projection_summary": point_projection_summary,
        "projection_features": summarize_pose_features(projection_features),
        "prior_summary": prior_summary,
        "eval": eval_reports,
        "differentiability_note": {
            "b52a": "mimic deterministic B5.1 residual direction; fully differentiable MSE in condition space",
            "b52b": "flow/pred_x0 proxy distills frozen sparse-flow outputs under target residual condition; decoded coords remain eval only",
            "b52c": "optional delta_norm_loss stabilizes residual amplitude before adding weak proxy",
            "decoded_coords": "not used as training loss",
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# B5.2 SS Cond Residual Adapter",
        "",
        f"- checkpoint: `{final_ckpt}`",
        f"- final_adapter_stats: `{final_adapter_stats}`",
        f"- point_projection_summary: `{point_projection_summary}`",
        "",
        "## Eval",
        "",
        "```text",
    ]
    for name, row in eval_reports.items():
        if not isinstance(row, dict) or "sparse" not in row:
            continue
        direction = ((row.get("delta_vs_baseline") or {}).get("direction_summary") or {})
        lines.extend(
            [
                f"{name}:",
                f"  sparse = {row.get('sparse')}",
                f"  added_minus_removed_within_prior_radius_ratio = {direction.get('added_minus_removed_within_prior_radius_ratio')}",
                f"  added_minus_removed_projection_any_mask_hit_ratio = {direction.get('added_minus_removed_projection_any_mask_hit_ratio')}",
                f"  added_minus_removed_visible_outside_mask_event_ratio = {direction.get('added_minus_removed_visible_outside_mask_event_ratio')}",
                f"  adapter_minus_baseline_visible_outside_mask_event_ratio = {direction.get('adapter_minus_baseline_visible_outside_mask_event_ratio')}",
                "",
            ]
        )
    lines.extend(["```", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[B5.2] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
