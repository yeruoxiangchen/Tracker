#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import _load_prior_manifest_sample  # noqa: E402
from reconvggt_ar_adapter_a.run_b54_physical_ssgrid_smoke import _build_physical_features, _stats  # noqa: E402
from reconvggt_ar_adapter_a.train_b55_physical_proxy_adapter import (  # noqa: E402
    FEATURE_NAMES,
    TRAIN_FEATURE_NAMES,
    LowRankPhysicalSSCondResidualAdapter,
    _build_loss_masks,
    _safe_weighted_mean,
    _smoothness_loss,
    _train_lowrank_gate_loss,
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


def _tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(y)
    if y.numel() == 0:
        return {
            "numel": 0,
            "finite_count": 0,
            "finite_ratio": 1.0,
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "abs_max": 0.0,
        }
    finite_y = y[finite]
    if finite_y.numel() == 0:
        finite_stats = {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
        }
    else:
        finite_stats = {
            "min": float(finite_y.min().item()),
            "max": float(finite_y.max().item()),
            "mean": float(finite_y.mean().item()),
            "std": float(finite_y.std(unbiased=False).item()),
            "abs_max": float(finite_y.abs().max().item()),
        }
    return {
        "numel": int(y.numel()),
        "finite_count": int(finite.sum().item()),
        "finite_ratio": float(finite.float().mean().item()),
        "nan_count": int(torch.isnan(y).sum().item()),
        "posinf_count": int(torch.isposinf(y).sum().item()),
        "neginf_count": int(torch.isneginf(y).sum().item()),
        **finite_stats,
    }


def _feature_channel_stats(features_np: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        x = torch.from_numpy(np.asarray(features_np[:, idx], dtype=np.float32))
        out[name] = _tensor_stats(x)
    return out


def _mask_sum_stats(loss_masks: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in ["pos16", "neg16", "neutral16", "active16"]:
        x = loss_masks[name].detach().float().cpu()
        row = _tensor_stats(x)
        row["sum"] = float(x.sum().item())
        row["nonzero_ratio"] = float((x != 0).float().mean().item())
        out[name] = row
    return out


def _grad_stats(module: nn.Module) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    first_nonfinite = None
    total_nonfinite = 0
    total_numel = 0
    for name, param in module.named_parameters():
        if param.grad is None:
            rows[name] = {"has_grad": False}
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        nonfinite = int((~finite).sum().item())
        total_nonfinite += nonfinite
        total_numel += int(grad.numel())
        if nonfinite and first_nonfinite is None:
            idx = torch.nonzero(~finite, as_tuple=False)[0].detach().cpu().tolist()
            first_nonfinite = {"name": name, "index": idx, "value": str(grad[tuple(idx)].item())}
        finite_grad = grad[finite]
        if finite_grad.numel() == 0:
            rows[name] = {
                "has_grad": True,
                "numel": int(grad.numel()),
                "finite_count": 0,
                "nonfinite_count": nonfinite,
                "finite_ratio": 0.0,
                "abs_max": None,
                "mean_abs": None,
                "norm": None,
            }
        else:
            rows[name] = {
                "has_grad": True,
                "numel": int(grad.numel()),
                "finite_count": int(finite.sum().item()),
                "nonfinite_count": nonfinite,
                "finite_ratio": float(finite.float().mean().item()),
                "abs_max": float(finite_grad.abs().max().item()),
                "mean_abs": float(finite_grad.abs().mean().item()),
                "norm": float(torch.linalg.vector_norm(finite_grad).item()),
            }
    return {
        "total_numel": total_numel,
        "total_nonfinite": total_nonfinite,
        "all_finite": total_nonfinite == 0,
        "first_nonfinite": first_nonfinite,
        "by_param": rows,
    }


class ToyLinearGate(nn.Module):
    def __init__(self, feature_dim: int, rank: int, init: str) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, rank)
        if init == "zero":
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        elif init == "small":
            nn.init.normal_(self.linear.weight, std=1.0e-4)
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError(f"Unknown init: {init}")

    def forward(self, features: torch.Tensor, batch: int) -> torch.Tensor:
        return self.linear(features.float()).unsqueeze(0).expand(int(batch), -1, -1)


class FakeSession:
    def __init__(self, cond_base: torch.Tensor, physical_features: torch.Tensor, loss_masks: dict[str, torch.Tensor]) -> None:
        self.cond_base = cond_base
        self.physical_features = physical_features
        self.loss_masks = loss_masks


def _gate_loss(
    gates: torch.Tensor,
    loss_masks: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gate_score = gates.mean(dim=-1)
    pos_w = loss_masks["pos16"].reshape(1, -1)
    neg_w = loss_masks["neg16"].reshape(1, -1)
    neutral_w = loss_masks["neutral16"].reshape(1, -1)
    pos_loss = _safe_weighted_mean((gate_score - float(args.gate_pos_target)).pow(2), pos_w)
    neg_loss = _safe_weighted_mean((gate_score + float(args.gate_neg_target)).pow(2), neg_w)
    preserve_loss = _safe_weighted_mean(gate_score.pow(2), neutral_w)
    gate_l2_loss = (gates.float() * gates.float()).mean()
    loss = (
        float(args.pos_weight) * pos_loss
        + float(args.neg_weight) * neg_loss
        + float(args.preserve_weight) * preserve_loss
        + float(args.gate_l2_weight) * gate_l2_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "pos_loss": float(pos_loss.detach().cpu().item()),
        "neg_loss": float(neg_loss.detach().cpu().item()),
        "preserve_loss": float(preserve_loss.detach().cpu().item()),
        "gate_l2_loss": float(gate_l2_loss.detach().cpu().item()),
        "gate_abs_mean": float(gates.detach().abs().mean().cpu().item()),
        "score_pos_mean": float(_safe_weighted_mean(gate_score.detach(), pos_w).cpu().item()),
        "score_neg_mean": float(_safe_weighted_mean(gate_score.detach(), neg_w).cpu().item()),
        "score_neutral_abs_mean": float(_safe_weighted_mean(gate_score.detach().abs(), neutral_w).cpu().item()),
    }


def _lowrank_delta(
    gates: torch.Tensor,
    basis: torch.Tensor,
    active_gate: torch.Tensor,
    cond_base: torch.Tensor,
) -> torch.Tensor:
    basis_norm = F.normalize(basis.float(), dim=-1)
    delta = torch.einsum("btr,rc->btc", gates.float(), basis_norm)
    return delta * active_gate.float()


def _run_backward_case(
    *,
    name: str,
    module: nn.Module,
    features: torch.Tensor,
    loss_masks: dict[str, torch.Tensor],
    cond_base: torch.Tensor,
    args: argparse.Namespace,
    include_delta_norm: bool,
    include_smooth: bool,
    use_actual_lowrank_loss: bool,
) -> dict[str, Any]:
    module.zero_grad(set_to_none=True)
    torch.manual_seed(int(args.seed))
    if use_actual_lowrank_loss:
        fake = FakeSession(cond_base=cond_base, physical_features=features, loss_masks=loss_masks)
        loss, stats = _train_lowrank_gate_loss(adapter=module, session=fake, args=args)
    else:
        if not isinstance(module, ToyLinearGate):
            raise TypeError("non-actual case expects ToyLinearGate")
        gates = module(features, batch=int(cond_base.shape[0]))
        loss, stats = _gate_loss(gates, loss_masks, args)
        if include_delta_norm or include_smooth:
            basis = torch.randn(
                gates.shape[-1],
                cond_base.shape[-1],
                device=features.device,
                dtype=torch.float32,
            ) * float(args.lowrank_basis_init_std)
            basis.requires_grad_(True)
            delta = _lowrank_delta(gates, basis, loss_masks["active_token"], cond_base)
            cond_power = (cond_base.float() * cond_base.float()).mean().clamp_min(1.0e-12)
            delta_norm_loss = (delta.float() * delta.float()).mean() / cond_power
            delta_norm = torch.sqrt(delta_norm_loss.detach().clamp_min(0.0))
            if include_delta_norm:
                loss = loss + float(args.delta_norm_weight) * delta_norm_loss
                stats["delta_norm_ratio"] = float(delta_norm.detach().cpu().item())
            if include_smooth:
                smooth_loss = _smoothness_loss(delta)
                loss = loss + float(args.smooth_weight) * smooth_loss
                stats["smooth_loss"] = float(smooth_loss.detach().cpu().item())
    loss_is_finite = bool(torch.isfinite(loss.detach()).item())
    if loss_is_finite:
        loss.backward()
    return {
        "name": name,
        "loss_is_finite": loss_is_finite,
        "stats": stats,
        "grad": _grad_stats(module),
    }


def _prepare_session(spec: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(
        Path(spec["prior_manifest"]),
        str(spec.get("prior_uid", "") or ""),
    )
    physical_np, physical_summary = _build_physical_features(sample=prior_sample, prior_coords=prior_coords, args=args)
    loss_masks = _build_loss_masks(physical_np, device, args)
    features = loss_masks["model_features"].to(device=device)
    cond_base = torch.ones(
        1,
        int(args.ss_grid_side) ** 3,
        int(args.channels),
        device=device,
        dtype=torch.float32,
    )
    return {
        "name": str(spec["name"]),
        "split": str(spec.get("split", "train")),
        "prior_summary": prior_summary,
        "physical_summary": physical_summary,
        "feature_channel_stats": _feature_channel_stats(physical_np),
        "loss_mask_stats": _mask_sum_stats(loss_masks),
        "features": features,
        "loss_masks": loss_masks,
        "cond_base": cond_base,
    }


def _write_md(path: Path, report: dict[str, Any], command: str) -> None:
    lines = [
        "# B5.7 Gradient Source Audit",
        "",
        "## Command",
        "",
        "```bash",
        command.strip(),
        "```",
        "",
        "## Scope",
        "",
        "```text",
        "不加载 ReconViaGen / VGGT / sparse sampler。",
        "只检查 B5.5/B5.6 使用的 physical_features、pos/neg/neutral mask、toy gate loss、实际 LowRank gate loss 的 backward。",
        "目标是定位非有限梯度来源，避免继续生成 no-op adapter report。",
        "```",
        "",
        "## Session Feature Sanity",
        "",
        "```text",
    ]
    for sess in report["sessions"]:
        lines.append(f"{sess['name']} split={sess['split']}")
        lines.append(f"  physical_sanity={sess['physical_summary']['sanity']}")
        for mask_name, row in sess["loss_mask_stats"].items():
            lines.append(
                f"  {mask_name}: sum={row['sum']:.6g} nonzero={row['nonzero_ratio']:.6g} "
                f"finite={row['finite_ratio']:.6g} min={row['min']} max={row['max']}"
            )
        bad = [
            name
            for name, row in sess["feature_channel_stats"].items()
            if float(row["finite_ratio"]) < 1.0
        ]
        lines.append(f"  nonfinite_feature_channels={bad}")
    lines.extend(["```", "", "## Backward Cases", "", "```text"])
    for row in report["backward_cases"]:
        grad = row["grad"]
        lines.append(
            f"{row['session']}::{row['case']}: finite_loss={row['loss_is_finite']} "
            f"loss={row['stats'].get('loss')} all_finite_grad={grad['all_finite']} "
            f"nonfinite_grad={grad['total_nonfinite']} first={grad['first_nonfinite']}"
        )
        lines.append(
            f"  pos={row['stats'].get('pos_loss')} neg={row['stats'].get('neg_loss')} "
            f"pres={row['stats'].get('preserve_loss')} gate_abs={row['stats'].get('gate_abs_mean')} "
            f"delta_norm={row['stats'].get('delta_norm_ratio')} smooth={row['stats'].get('smooth_loss')}"
        )
    lines.extend(["```", "", "## Judgment", "", report["judgment"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B5.7 physical proxy gradient source audit.")
    parser.add_argument("--sessions_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_grid_side", type=int, default=16)
    parser.add_argument("--sparse_resolution", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--physical_vh_min_visible_views", type=int, default=1)
    parser.add_argument("--physical_vh_min_support_ratio", type=float, default=0.5)
    parser.add_argument("--physical_distance_clip", type=float, default=8.0)
    parser.add_argument("--positive_min_visible_views", type=int, default=1)
    parser.add_argument("--positive_min_support_ratio", type=float, default=0.5)
    parser.add_argument("--negative_min_visible_views", type=int, default=3)
    parser.add_argument("--negative_max_support_ratio", type=float, default=0.1)
    parser.add_argument("--negative_min_outside_ratio", type=float, default=0.9)
    parser.add_argument("--negative_prior_radius_multiplier", type=float, default=1.0)
    parser.add_argument("--loss_mask_mode", choices=["exclusive_surface", "legacy"], default="exclusive_surface")
    parser.add_argument(
        "--require_nonempty_surface_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--visual_hull_active_weight", type=float, default=0.25)
    parser.add_argument("--use_prior_score_positive", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lowrank_rank", type=int, default=4)
    parser.add_argument("--lowrank_basis_init_std", type=float, default=0.02)
    parser.add_argument("--gate_pos_target", type=float, default=0.02)
    parser.add_argument("--gate_neg_target", type=float, default=0.01)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--neg_weight", type=float, default=2.0)
    parser.add_argument("--preserve_weight", type=float, default=0.1)
    parser.add_argument("--gate_l2_weight", type=float, default=0.01)
    parser.add_argument("--delta_norm_weight", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--delta_clip_abs", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    sessions_spec = json.loads(Path(args.sessions_json).read_text(encoding="utf-8"))
    sessions = [_prepare_session(spec, args, device) for spec in sessions_spec]

    backward_rows: list[dict[str, Any]] = []
    train_sessions = [s for s in sessions if s["split"] == "train"]
    for sess in train_sessions:
        features = sess["features"]
        loss_masks = sess["loss_masks"]
        cond_base = sess["cond_base"]
        cases: list[tuple[str, nn.Module, bool, bool, bool]] = [
            (
                "toy_linear_zero_gate_only",
                ToyLinearGate(len(TRAIN_FEATURE_NAMES), int(args.lowrank_rank), "zero").to(device),
                False,
                False,
                False,
            ),
            (
                "toy_linear_zero_gate_delta_norm",
                ToyLinearGate(len(TRAIN_FEATURE_NAMES), int(args.lowrank_rank), "zero").to(device),
                True,
                False,
                False,
            ),
            (
                "toy_linear_zero_gate_delta_norm_smooth",
                ToyLinearGate(len(TRAIN_FEATURE_NAMES), int(args.lowrank_rank), "zero").to(device),
                True,
                True,
                False,
            ),
            (
                "toy_linear_small_gate_only",
                ToyLinearGate(len(TRAIN_FEATURE_NAMES), int(args.lowrank_rank), "small").to(device),
                False,
                False,
                False,
            ),
            (
                "actual_lowrank_full_loss",
                LowRankPhysicalSSCondResidualAdapter(
                    channels=int(args.channels),
                    feature_dim=len(TRAIN_FEATURE_NAMES),
                    hidden_dim=int(args.hidden_dim),
                    rank=int(args.lowrank_rank),
                    basis_init_std=float(args.lowrank_basis_init_std),
                ).to(device),
                False,
                False,
                True,
            ),
        ]
        for case_name, module, include_delta_norm, include_smooth, actual in cases:
            torch.manual_seed(int(args.seed))
            row = _run_backward_case(
                name=case_name,
                module=module,
                features=features,
                loss_masks=loss_masks,
                cond_base=cond_base,
                args=args,
                include_delta_norm=include_delta_norm,
                include_smooth=include_smooth,
                use_actual_lowrank_loss=actual,
            )
            row["session"] = sess["name"]
            row["case"] = case_name
            backward_rows.append(row)
            print(
                f"[B5.7] {sess['name']} {case_name} "
                f"loss_finite={row['loss_is_finite']} grad_finite={row['grad']['all_finite']} "
                f"nonfinite={row['grad']['total_nonfinite']} first={row['grad']['first_nonfinite']}",
                flush=True,
            )

    all_features_finite = all(
        float(ch_stats["finite_ratio"]) == 1.0
        for sess in sessions
        for ch_stats in sess["feature_channel_stats"].values()
    )
    toy_all_finite = all(
        row["grad"]["all_finite"]
        for row in backward_rows
        if str(row["case"]).startswith("toy_")
    )
    actual_all_finite = all(
        row["grad"]["all_finite"]
        for row in backward_rows
        if row["case"] == "actual_lowrank_full_loss"
    )
    if not all_features_finite:
        judgment = "B5.7 found non-finite physical feature channels; fix feature construction before any training."
    elif toy_all_finite and not actual_all_finite:
        judgment = (
            "B5.7 found toy linear gradients are finite but actual LowRank loss has non-finite gradients. "
            "The issue is inside the current LowRank adapter/loss path, not the physical features themselves."
        )
    elif toy_all_finite and actual_all_finite:
        judgment = (
            "B5.7 found physical features and standalone LowRank loss gradients are finite. "
            "Previous non-finite gradients likely come from real cond_base / pipeline-coupled training state; audit cond_base next."
        )
    else:
        judgment = "B5.7 found even toy linear gradients are non-finite; inspect raw feature scale and loss masks before any adapter training."

    report_sessions = []
    for sess in sessions:
        report_sessions.append(
            {
                "name": sess["name"],
                "split": sess["split"],
                "prior_summary": sess["prior_summary"],
                "physical_summary": sess["physical_summary"],
                "feature_channel_stats": sess["feature_channel_stats"],
                "loss_mask_stats": sess["loss_mask_stats"],
            }
        )
    report = {
        "args": vars(args),
        "command": " ".join(sys.argv),
        "sessions": report_sessions,
        "backward_cases": backward_rows,
        "summary": {
            "all_features_finite": all_features_finite,
            "toy_all_finite": toy_all_finite,
            "actual_lowrank_all_finite": actual_all_finite,
        },
        "judgment": judgment,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_md(output_dir / "report.md", report, " ".join(sys.argv))
    print(f"[B5.7] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
