#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

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

from reconvggt_ar_adapter_a.inspect_and_sanity import force_eval  # noqa: E402
from reconvggt_ar_adapter_a.run_b54_physical_ssgrid_smoke import _token_grid_mapping_audit  # noqa: E402
from reconvggt_ar_adapter_a.train_b5_ss_cond_residual_adapter import (  # noqa: E402
    install_dreamsim_stub,
    set_frozen_eval,
)
from reconvggt_ar_adapter_a.train_b55_physical_proxy_adapter import (  # noqa: E402
    FEATURE_NAMES,
    TRAIN_FEATURE_NAMES,
    LowRankPhysicalSSCondResidualAdapter,
    _one_step_logits,
    _prepare_session,
    _rescale_proxy_time,
    _train_step_loss,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def _grad_group(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    rows = []
    norm_sq = 0.0
    nonfinite = 0
    nonzero = 0
    grad_numel = 0
    for name, param in named_params:
        if param.grad is None:
            rows.append({"name": name, "has_grad": False})
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        finite_grad = grad[finite]
        grad_numel += int(grad.numel())
        nonfinite += int((~finite).sum().item())
        nonzero += int((finite_grad != 0).sum().item())
        if finite_grad.numel() > 0:
            norm_sq += float((finite_grad * finite_grad).sum().cpu().item())
        rows.append(
            {
                "name": name,
                "has_grad": True,
                "numel": int(grad.numel()),
                "nonfinite": int((~finite).sum().item()),
                "nonzero": int((finite_grad != 0).sum().item()),
                "abs_max": float(finite_grad.abs().max().cpu().item()) if finite_grad.numel() else None,
            }
        )
    return {
        "grad_numel": grad_numel,
        "nonfinite": nonfinite,
        "nonzero": nonzero,
        "norm": norm_sq ** 0.5,
        "all_finite": nonfinite == 0,
        "rows": rows,
    }


def _collect_gradients(adapter: LowRankPhysicalSSCondResidualAdapter) -> dict[str, Any]:
    named = dict(adapter.named_parameters())
    gate_output_names = [name for name in named if name.startswith("gate_mlp.2.")]
    gate_hidden_names = [name for name in named if name.startswith("gate_mlp.0.")]
    return {
        "all": _grad_group(named.items()),
        "gate_output": _grad_group((name, named[name]) for name in gate_output_names),
        "gate_hidden": _grad_group((name, named[name]) for name in gate_hidden_names),
        "basis": _grad_group([("basis", adapter.basis)]),
    }


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# B5.10 Flow-Gradient Low-Rank Audit",
        "",
        "```text",
        "This is a gradient-connectivity audit, not a reconstruction-quality result.",
        "The sparse flow and sparse decoder are frozen; autograd must still traverse both into the SS-condition adapter.",
        "At zero gate initialization, gate-output gradients must be nonzero while basis gradients may be zero.",
        "After one probe optimizer step, both gate-output and basis gradients must be finite and nonzero.",
        "```",
        "",
        f"- session: `{report['session']['name']}` ({report['session']['split']})",
        f"- t: `{report['args']['t']}`",
        f"- label overlap: `{report['session']['loss_masks']['positive_negative_overlap_count']}`",
        f"- initial loss: `{report['initial']['loss']}`",
        f"- post-step loss: `{report['post_step']['loss']}`",
        f"- initial gate-output grad norm: `{report['initial']['gradients']['gate_output']['norm']}`",
        f"- initial basis grad norm: `{report['initial']['gradients']['basis']['norm']}`",
        f"- post-step gate-output grad norm: `{report['post_step']['gradients']['gate_output']['norm']}`",
        f"- post-step gate-hidden grad norm: `{report['post_step']['gradients']['gate_hidden']['norm']}`",
        f"- post-step basis grad norm: `{report['post_step']['gradients']['basis']['norm']}`",
        f"- frozen flow/decoder trainable params: `{report['frozen_modules']}`",
        f"- passed: `{report['passed']}`",
        "",
        "## Judgment",
        "",
        report["judgment"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit low-rank adapter gradients through frozen SS flow and decoder.")
    parser.add_argument("--sessions_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_session_name", default="")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
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
    parser.add_argument("--loss_mask_mode", choices=["exclusive_surface", "legacy"], default="exclusive_surface")
    parser.add_argument(
        "--require_nonempty_surface_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use_prior_score_positive", action="store_true")
    parser.add_argument("--visual_hull_active_weight", type=float, default=0.25)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lowrank_rank", type=int, default=4)
    parser.add_argument("--lowrank_basis_init_std", type=float, default=0.02)
    parser.add_argument("--t", type=float, default=0.9)
    parser.add_argument("--proxy_cfg_mode", choices=["unguided", "sampler"], default="unguided")
    parser.add_argument("--proxy_rescale_t", type=float, default=1.0)
    parser.add_argument("--ss_cfg_strength", type=float, default=7.5)
    parser.add_argument("--ss_cfg_interval_min", type=float, default=0.6)
    parser.add_argument("--ss_cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--probe_lr", type=float, default=1.0e-4)
    parser.add_argument("--train_runtime_scale", type=float, default=1.0)
    parser.add_argument("--margin_pos", type=float, default=0.003)
    parser.add_argument("--margin_neg", type=float, default=0.001)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--neg_weight", type=float, default=2.0)
    parser.add_argument("--preserve_weight", type=float, default=0.1)
    parser.add_argument("--delta_norm_weight", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--delta_clip_abs", type=float, default=0.0)
    args = parser.parse_args()

    startup_mapping_audit = _token_grid_mapping_audit(
        int(args.ss_grid_side),
        int(args.sparse_resolution),
    )
    if not startup_mapping_audit["passed"]:
        raise RuntimeError(f"SS token mapping/round-trip audit failed before pipeline load: {startup_mapping_audit}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.sessions_json).read_text(encoding="utf-8"))
    train_specs = [spec for spec in specs if str(spec.get("split", "train")) == "train"]
    if args.audit_session_name:
        train_specs = [spec for spec in train_specs if str(spec["name"]) == args.audit_session_name]
    if not train_specs:
        raise ValueError("No matching train session for flow-gradient audit")
    spec = train_specs[0]

    if not args.load_dreamsim:
        install_dreamsim_stub()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    torch.manual_seed(int(args.seed))
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

    session = _prepare_session(pipeline=pipeline, args=args, spec=spec, device=device)
    overlap_count = int(session.loss_masks["summary"]["positive_negative_overlap_count"])
    flow = pipeline.models["sparse_structure_flow_model"].to(session.cond_base.device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(session.cond_base.device).eval()
    set_frozen_eval(flow)
    set_frozen_eval(decoder)
    sampler = pipeline.sparse_structure_sampler
    adapter = LowRankPhysicalSSCondResidualAdapter(
        channels=int(session.cond_base.shape[-1]),
        feature_dim=len(TRAIN_FEATURE_NAMES),
        hidden_dim=int(args.hidden_dim),
        rank=int(args.lowrank_rank),
        basis_init_std=float(args.lowrank_basis_init_std),
    ).to(session.cond_base.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.probe_lr), weight_decay=0.0)

    torch.manual_seed(int(args.seed) + 991)
    x_t = torch.randn(
        int(args.num_samples),
        int(flow.in_channels),
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        device=session.cond_base.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        t_model = _rescale_proxy_time(float(args.t), float(args.proxy_rescale_t))
        base_logits, _ = _one_step_logits(
            flow=flow,
            decoder=decoder,
            sampler=sampler,
            x_t=x_t,
            t_model=t_model,
            cond=session.cond_base.float(),
            neg_cond=session.neg_cond,
            cfg_mode=str(args.proxy_cfg_mode),
            cfg_strength=float(args.ss_cfg_strength),
            cfg_interval=(float(args.ss_cfg_interval_min), float(args.ss_cfg_interval_max)),
            guidance_rescale=float(args.ss_guidance_rescale),
            autocast_enabled=(session.cond_base.device.type == "cuda"),
        )

    def backward_snapshot() -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        optimizer.zero_grad(set_to_none=True)
        loss, stats = _train_step_loss(
            adapter=adapter,
            session=session,
            base_logits=base_logits,
            flow=flow,
            decoder=decoder,
            sampler=sampler,
            x_t=x_t,
            t=t_model,
            args=args,
        )
        loss.backward()
        return loss, stats, _collect_gradients(adapter)

    initial_loss, initial_stats, initial_grad = backward_snapshot()
    optimizer.step()
    post_loss, post_stats, post_grad = backward_snapshot()

    flow_trainable = int(sum(p.numel() for p in flow.parameters() if p.requires_grad))
    decoder_trainable = int(sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    pos_count = int(session.loss_masks["summary"]["positive_count"])
    neg_count = int(session.loss_masks["summary"]["negative_count"])
    mapping_passed = bool(
        ((session.physical_summary.get("grid") or {}).get("token_mapping_audit") or {}).get("passed")
    )
    passed = bool(
        overlap_count == 0
        and pos_count > 0
        and neg_count > 0
        and mapping_passed
        and flow_trainable == 0
        and decoder_trainable == 0
        and torch.isfinite(initial_loss.detach()).item()
        and torch.isfinite(post_loss.detach()).item()
        and initial_grad["all"]["all_finite"]
        and post_grad["all"]["all_finite"]
        and float(initial_grad["gate_output"]["norm"]) > 0.0
        and float(post_grad["gate_output"]["norm"]) > 0.0
        and float(post_grad["gate_hidden"]["norm"]) > 0.0
        and float(post_grad["basis"]["norm"]) > 0.0
    )
    judgment = (
        "PASS: exclusive physical labels are conflict-free and the decoder-logit objective sends finite gradients through frozen flow/decoder into both low-rank gates and channel basis after one probe update. This only permits a B5.10 smoke; it does not prove rollout quality."
        if passed
        else "FAIL: the B5.10 decoder-logit path is not yet interpretable. Do not launch adapter training; inspect label overlap and the reported gate/basis gradients."
    )
    report = {
        "args": vars(args),
        "scope": "One-session, one-time-step gradient connectivity audit through frozen sparse flow and decoder.",
        "startup_mapping_audit": startup_mapping_audit,
        "session": {
            "name": session.name,
            "split": session.split,
            "feature_frame_scope": session.physical_summary.get("feature_frame_scope"),
            "evaluation_frame_scope": session.physical_summary.get("evaluation_frame_scope"),
            "physical_sanity": session.physical_summary.get("sanity"),
            "loss_masks": session.loss_masks["summary"],
        },
        "initial": {
            "loss": float(initial_loss.detach().cpu().item()),
            "stats": initial_stats,
            "gradients": initial_grad,
        },
        "post_step": {
            "loss": float(post_loss.detach().cpu().item()),
            "stats": post_stats,
            "gradients": post_grad,
        },
        "frozen_modules": {
            "flow_trainable_parameters": flow_trainable,
            "decoder_trainable_parameters": decoder_trainable,
        },
        "passed": passed,
        "judgment": judgment,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_md(output_dir / "report.md", report)
    print(f"[B5.10 audit] passed={passed} wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
