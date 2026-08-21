#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")

import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from ar_ss_flow.build_pose_lifting_cache import extract_stock_condition  # noqa: E402
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    install_unused_model_stubs,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_native_pipeline(pretrained: str, device: torch.device):
    """Load the unmodified ReconViaGen condition path without replacing VGGT."""
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    keep_models = {"image_cond_model", "sparse_structure_vggt_cond"}
    for name in list(pipeline.models):
        if name not in keep_models:
            del pipeline.models[name]
    for name in ("slat_flow_model", "slat_vggt_cond", "sparse_structure_flow_model"):
        if hasattr(pipeline, name):
            delattr(pipeline, name)
    # Do not replace pipeline.VGGT_model here. This is the baseline under audit.
    pipeline.VGGT_model.to(device).eval()
    pipeline.models["image_cond_model"].to(device).eval()
    pipeline.models["sparse_structure_vggt_cond"].to(device).eval()
    for module in (
        pipeline.VGGT_model,
        pipeline.models["image_cond_model"],
        pipeline.models["sparse_structure_vggt_cond"],
    ):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return pipeline


def comparison(native: torch.Tensor, cached: torch.Tensor) -> dict[str, Any]:
    native_fp16 = native.detach().cpu().to(torch.float16)
    cached_fp16 = cached.detach().cpu().to(torch.float16)
    if native_fp16.shape != cached_fp16.shape:
        return {
            "shape_equal": False,
            "native_shape": list(native_fp16.shape),
            "cached_shape": list(cached_fp16.shape),
        }
    difference = native_fp16.float() - cached_fp16.float()
    cosine = F.cosine_similarity(
        native_fp16.float().flatten()[None],
        cached_fp16.float().flatten()[None],
    ).item()
    return {
        "shape_equal": True,
        "native_shape": list(native_fp16.shape),
        "cached_shape": list(cached_fp16.shape),
        "torch_equal_fp16": bool(torch.equal(native_fp16, cached_fp16)),
        "max_abs_diff": float(difference.abs().max().item()),
        "rms_diff": float(difference.square().mean().sqrt().item()),
        "cosine": float(cosine),
        "native_rms": float(native_fp16.float().square().mean().sqrt().item()),
        "cached_rms": float(cached_fp16.float().square().mean().sqrt().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cached stock conditions with the unmodified ReconViaGen path."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0,1,5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_abs_tolerance", type=float, default=0.0)
    parser.add_argument("--rms_tolerance", type=float, default=0.0)
    parser.add_argument("--min_cosine", type=float, default=1.0)
    parser.add_argument("--require_fp16_equal", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    resolved_indices = str(args.indices)
    if resolved_indices == "auto_2_4_8":
        manifest = json.loads(
            Path(args.cache_manifest).read_text(encoding="utf-8")
        )
        selected: dict[int, int] = {}
        for index, row in enumerate(manifest.get("samples", ())):
            view_count = int(row.get("view_count", 0))
            if view_count in (2, 4, 8) and view_count not in selected:
                selected[view_count] = index
        if set(selected) != {2, 4, 8}:
            raise ValueError(
                "cache does not contain all required 2/4/8-view samples: "
                f"found={sorted(selected)}"
            )
        resolved_indices = ",".join(
            str(selected[view_count]) for view_count in (2, 4, 8)
        )
    dataset = PoseLiftingCacheDataset(
        args.cache_manifest,
        indices=resolved_indices,
    )
    manifest_path = Path(args.cache_manifest).resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = list(manifest_payload.get("samples", ()))
    pipeline = build_native_pipeline(args.pretrained, device)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        native = extract_stock_condition(pipeline, sample)
        metrics = comparison(native, sample["stock_condition"])
        metrics.update(
            {
                "uid": str(sample["uid"]),
                "object_uid": str(sample["object_uid"]),
                "view_count": int(len(sample["view_ids"])),
            }
        )
        metrics["passed"] = bool(
            metrics.get("shape_equal", False)
            and metrics.get("max_abs_diff", float("inf"))
            <= float(args.max_abs_tolerance)
            and metrics.get("rms_diff", float("inf")) <= float(args.rms_tolerance)
            and metrics.get("cosine", -1.0) >= float(args.min_cosine)
            and (
                not args.require_fp16_equal
                or metrics.get("torch_equal_fp16", False)
            )
        )
        rows.append(metrics)
        print(
            f"[stock_condition_audit] {index + 1}/{len(dataset)} "
            f"uid={sample['uid']} views={metrics['view_count']} "
            f"max={metrics.get('max_abs_diff')} rms={metrics.get('rms_diff')} "
            f"equal={metrics.get('torch_equal_fp16')}",
            flush=True,
        )
    view_counts = {int(row["view_count"]) for row in rows}
    checks = {
        "covers_2_4_8_views": {2, 4, 8}.issubset(view_counts),
        "all_conditions_match_native": all(row["passed"] for row in rows),
        "native_pipeline_vggt_not_replaced": True,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_config_hash": str(manifest_payload.get("config_hash", "")),
        "cache_schema_hash": json_hash({
            key: manifest_payload.get(key)
            for key in (
                "format", "stock_condition_source", "lifting_feature_source",
                "visual_feature_dim", "feature_metadata", "metadata_names",
                "metadata_schema_hash", "config", "config_hash",
            )
        }),
        "uid_hash": json_hash(sorted(str(row.get("uid", "")) for row in all_rows)),
        "object_uid_hash": json_hash(sorted({
            str(row.get("object_uid", row.get("uid", ""))) for row in all_rows
        })),
        "pretrained": args.pretrained,
        "indices": args.indices,
        "resolved_indices": resolved_indices,
        "thresholds": {
            "max_abs_tolerance": float(args.max_abs_tolerance),
            "rms_tolerance": float(args.rms_tolerance),
            "min_cosine": float(args.min_cosine),
            "require_fp16_equal": bool(args.require_fp16_equal),
        },
        "samples": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Cached Stock Condition Audit",
        "",
        f"- passed: `{report['passed']}`",
        f"- covered view counts: `{sorted(view_counts)}`",
        "- baseline: unmodified ReconViaGen VGGT + image encoder + get_ss_cond.",
        "",
        "| uid | views | fp16 equal | max abs | RMS | cosine | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['uid']} | {row['view_count']} | "
            f"{row.get('torch_equal_fp16')} | {row.get('max_abs_diff')} | "
            f"{row.get('rms_diff')} | {row.get('cosine')} | {row['passed']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
