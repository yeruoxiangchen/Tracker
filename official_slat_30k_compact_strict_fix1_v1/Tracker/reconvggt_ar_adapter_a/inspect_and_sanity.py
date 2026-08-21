#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import torch
from torch import nn
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import trellis.pipelines.trellis_image_to_3d as trellis_image_to_3d  # noqa: E402
from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.token_adapter import (  # noqa: E402
    ZeroInitResidualTokenAdapter,
    infer_token_layout,
    max_abs_tree_diff,
    parse_layer_indices,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class DreamSimStub(nn.Module):
    """Minimal DreamSim replacement for A-stage token/condition sanity checks."""

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], torch.Tensor):
            batch = int(args[0].shape[0]) if args[0].ndim > 0 else 1
            return torch.zeros((batch,), device=args[0].device, dtype=args[0].dtype)
        return torch.zeros((1,))


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def force_eval(obj: object) -> None:
    seen: set[int] = set()

    def _visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, nn.Module):
            value.eval()

    if isinstance(obj, nn.Module):
        _visit(obj)
    if hasattr(obj, "models") and isinstance(getattr(obj, "models"), dict):
        for value in getattr(obj, "models").values():
            _visit(value)
    for value in vars(obj).values():
        _visit(value)


def load_images(image_dir: Path, *, max_views: int, preprocess: bool, pipeline: TrellisVGGTTo3DPipeline) -> tuple[list[Image.Image], list[str]]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_views > 0:
        paths = paths[:max_views]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    images: list[Image.Image] = []
    names: list[str] = []
    for path in paths:
        image = Image.open(path).convert("RGBA")
        if preprocess:
            image = pipeline.preprocess_image(image)
        else:
            image = image.convert("RGB")
        images.append(image)
        names.append(str(path))
    return images, names


def tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach()
    yf = y.float()
    return {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "device": str(y.device),
        "mean": float(yf.mean().item()) if y.numel() else 0.0,
        "std": float(yf.std().item()) if y.numel() > 1 else 0.0,
        "abs_max": float(yf.abs().max().item()) if y.numel() else 0.0,
    }


def normalize_image_cond(raw_image_cond: torch.Tensor, *, batch: int, views: int) -> torch.Tensor:
    if raw_image_cond.ndim == 3:
        image_cond = raw_image_cond.unsqueeze(0)
    elif raw_image_cond.ndim == 4:
        image_cond = raw_image_cond
    else:
        raise ValueError(f"Unexpected image_cond shape: {tuple(raw_image_cond.shape)}")
    if image_cond.shape[0] != batch or image_cond.shape[1] != views:
        raise ValueError(
            f"image_cond view layout mismatch: image_cond={tuple(image_cond.shape)}, "
            f"expected batch/views=({batch}, {views})"
        )
    if image_cond.shape[-1] != 1024:
        raise ValueError(f"Expected image_cond last dim 1024, got {image_cond.shape[-1]}")
    return image_cond


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="ReconViaGen VGGT token layout and zero-init adapter sanity check.")
    parser.add_argument("--image_dir", required=True, help="Directory of input images.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true", help="Run ReconViaGen background-removal preprocessing before VGGT.")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_layers", default="4,11,17,23")
    parser.add_argument("--patch_start_idx", type=int, default=5, help="ReconViaGen sparse_structure_vggt_cond currently skips the first 5 tokens.")
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--check_slat", action="store_true")
    parser.add_argument("--load_dreamsim", action="store_true", help="Load DreamSim during pipeline init. A-stage sanity does not need it.")
    parser.add_argument(
        "--ss_image_cond_mode",
        choices=["skip_prefix", "full"],
        default="skip_prefix",
        help="ReconViaGen run() currently passes image_cond[:, :, 5:] to get_ss_cond; keep skip_prefix unless checking an alternate code path.",
    )
    parser.add_argument("--save_adapter", default="", help="Optional path to save zero-init adapter checkpoint.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[A-sanity] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
    if not args.load_dreamsim:
        install_dreamsim_stub()
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

    images, image_names = load_images(Path(args.image_dir), max_views=args.max_views, preprocess=args.preprocess, pipeline=pipeline)
    print(f"[A-sanity] loaded {len(images)} images from {args.image_dir}", flush=True)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, input_tensor = pipeline.vggt_feat(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    raw_image_cond = pipeline.encode_image(images)
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)

    selected_layers = parse_layer_indices(args.adapter_layers)
    layouts = [
        infer_token_layout(
            idx,
            aggregated_tokens_list[idx],
            prefix_tokens=args.patch_start_idx,
            image_resolution=args.image_resolution,
        ).__dict__
        for idx in range(len(aggregated_tokens_list))
    ]
    selected_layouts = {str(idx): layouts[idx] for idx in selected_layers}

    image_cond_layout = {
        "shape": list(image_cond.shape),
        "raw_shape": list(raw_image_cond.shape),
        "prefix_tokens": int(args.patch_start_idx),
        "spatial_tokens_after_prefix": int(image_cond.shape[2] - args.patch_start_idx),
    }
    image_cond_spatial_tokens = max(0, int(image_cond.shape[2] - args.patch_start_idx))
    side = int(round(image_cond_spatial_tokens ** 0.5))
    image_cond_layout["square_spatial_grid_after_prefix"] = image_cond_spatial_tokens > 0 and side * side == image_cond_spatial_tokens
    image_cond_layout["spatial_side_after_prefix"] = side if image_cond_layout["square_spatial_grid_after_prefix"] else None
    image_cond_layout["pixel_per_token"] = (
        float(args.image_resolution) / float(side)
        if image_cond_layout["square_spatial_grid_after_prefix"] and side > 0
        else None
    )

    adapter = ZeroInitResidualTokenAdapter.from_tokens(
        aggregated_tokens_list,
        hidden_dim=args.adapter_hidden_dim,
        layer_indices=selected_layers,
    ).to(device=aggregated_tokens_list[0].device, dtype=aggregated_tokens_list[0].dtype)
    adapted_tokens_list = adapter(aggregated_tokens_list)

    token_diffs = {
        str(idx): max_abs_tree_diff(aggregated_tokens_list[idx], adapted_tokens_list[idx])
        for idx in selected_layers
    }

    ss_image_cond = image_cond[:, :, args.patch_start_idx :] if args.ss_image_cond_mode == "skip_prefix" else image_cond
    ss_cond_base = pipeline.get_ss_cond(ss_image_cond, aggregated_tokens_list, num_samples=1)
    ss_cond_adapted = pipeline.get_ss_cond(ss_image_cond, adapted_tokens_list, num_samples=1)
    ss_cond_max_abs_diff = max_abs_tree_diff(ss_cond_base["cond"], ss_cond_adapted["cond"])

    slat_cond_max_abs_diff = None
    if args.check_slat:
        slat_cond_base = pipeline.get_slat_cond(image_cond, aggregated_tokens_list, num_samples=1)
        slat_cond_adapted = pipeline.get_slat_cond(image_cond, adapted_tokens_list, num_samples=1)
        slat_cond_max_abs_diff = max_abs_tree_diff(slat_cond_base["cond"], slat_cond_adapted["cond"])

    token_passed = all(v == 0.0 for v in token_diffs.values())
    ss_passed = ss_cond_max_abs_diff == 0.0
    slat_passed = True if slat_cond_max_abs_diff is None else slat_cond_max_abs_diff == 0.0

    report: dict[str, Any] = {
        "args": vars(args),
        "image_names": image_names,
        "input_tensor": tensor_summary(input_tensor),
        "num_vggt_layers": len(aggregated_tokens_list),
        "selected_layers": selected_layers,
        "selected_layouts": selected_layouts,
        "all_layouts": layouts,
        "image_cond_layout": image_cond_layout,
        "reconviagen_assumptions": {
            "sparse_structure_vggt_cond_layers": [4, 11, 17, 23],
            "sparse_structure_patch_start_idx": 5,
            "ss_image_cond_mode": args.ss_image_cond_mode,
            "slat_vggt_cond_layers": [4, 11, 17, 23],
            "slat_uses_full_vggt_tokens": True,
            "view_dimension_expected": "[B, V, T, C]",
            "dreamsim_loaded": bool(args.load_dreamsim),
        },
        "adapter": adapter.metadata(),
        "zero_init_sanity": {
            "selected_token_max_abs_diff": token_diffs,
            "ss_cond_max_abs_diff": ss_cond_max_abs_diff,
            "slat_cond_max_abs_diff": slat_cond_max_abs_diff,
            "token_passed": token_passed,
            "ss_passed": ss_passed,
            "slat_passed": slat_passed,
            "passed": token_passed and ss_passed and slat_passed,
        },
    }

    if args.save_adapter:
        save_path = Path(args.save_adapter)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": adapter.state_dict(), "metadata": adapter.metadata(), "report": report}, save_path)
        report["adapter_checkpoint"] = str(save_path)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# ReconVGGT AR Adapter A Sanity Report",
        "",
        f"- images: {len(images)}",
        f"- selected layers: {selected_layers}",
        f"- ss_cond_max_abs_diff: `{ss_cond_max_abs_diff}`",
        f"- slat_cond_max_abs_diff: `{slat_cond_max_abs_diff}`",
        f"- zero_init_passed: `{report['zero_init_sanity']['passed']}`",
        f"- token_passed: `{token_passed}`",
        f"- ss_passed: `{ss_passed}`",
        f"- slat_passed: `{slat_passed}`",
        f"- dreamsim_loaded: `{bool(args.load_dreamsim)}`",
        f"- ss_image_cond_mode: `{args.ss_image_cond_mode}`",
        "",
        "## Selected Token Layouts",
        "",
        "| layer | shape | prefix | spatial tokens | side | pixel/token | square |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for idx in selected_layers:
        row = selected_layouts[str(idx)]
        lines.append(
            "| {layer} | {shape} | {prefix} | {spatial} | {side} | {ppt} | {square} |".format(
                layer=idx,
                shape=row["shape"],
                prefix=row["prefix_tokens"],
                spatial=row["spatial_tokens"],
                side=row["spatial_side"],
                ppt=row["pixel_per_token"],
                square=row["square_spatial_grid"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- A 阶段只验证 adapter 插入后不破坏 ReconViaGen/VGGT condition。",
            "- token spatial mapping 只在 `spatial_tokens = side^2` 时成立；否则不能把 VGGT token 直接当作像素网格。",
            "- 当前 ReconViaGen sparse condition 代码显式使用 `patch_start_idx=5`，因此前 5 个 token 先按 prefix/global/register token 处理。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[A-sanity] wrote {output_dir / 'report.json'}", flush=True)
    print(f"[A-sanity] wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
