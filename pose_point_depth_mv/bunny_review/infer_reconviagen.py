#!/usr/bin/env python3
"""Run ReconViaGen stock or its native LoRA-checkpoint variant on Bunny."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

from PIL import Image

from .common import (
    binding,
    code_bindings,
    load_method_result,
    load_protocol,
    method_dir,
    sha256_file,
    write_method_result,
)


TRACKER_ROOT = Path(__file__).resolve().parents[2]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"


def export_mesh_atomic(mesh, path: Path, *, file_type: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    mesh.export(temporary, file_type=file_type)
    os.replace(temporary, path)


def checkpoint_kind(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    if "format" in payload and "state_dict" not in payload:
        raise ValueError(
            f"{path} is checkpoint format={payload.get('format')!r}, not a native "
            "ReconViaGen Lightning LoRA checkpoint. Use the command adapter for "
            "Direct-SS/Direct-SLAT checkpoints."
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint lacks non-empty state_dict: {path}")
    metadata = {
        "binding": binding(path),
        "top_level_keys": sorted(payload),
        "state_key_count": len(state),
    }
    return payload, metadata


def load_native_lora(
    pipeline,
    *,
    stage: str,
    checkpoint: Path,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model

    payload, metadata = checkpoint_kind(checkpoint)
    state = payload["state_dict"]
    if stage == "ss":
        flow_key = "sparse_structure_flow_model"
        cond_key = "sparse_structure_vggt_cond"
        flow_prefix = "ss_flow_model."
        cond_prefix = "ss_cond."
        rank, alpha = 64, 128
    elif stage == "slat":
        flow_key = "slat_flow_model"
        cond_key = "slat_vggt_cond"
        flow_prefix = "slat_flow_model."
        cond_prefix = "slat_cond."
        rank, alpha = 128, 256
    else:
        raise ValueError(stage)
    flow_state = {
        key.removeprefix(flow_prefix): value
        for key, value in state.items()
        if key.startswith(flow_prefix)
    }
    cond_state = {
        key.removeprefix(cond_prefix): value
        for key, value in state.items()
        if key.startswith(cond_prefix)
    }
    if not flow_state or not cond_state:
        raise RuntimeError(
            f"{stage} checkpoint does not contain both {flow_prefix}* and "
            f"{cond_prefix}* states"
        )
    peft_model = get_peft_model(
        pipeline.models[flow_key],
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
        ),
    )
    incompatible = peft_model.load_state_dict(flow_state, strict=False)
    cond_incompatible = pipeline.models[cond_key].load_state_dict(
        cond_state, strict=False
    )
    pipeline.models[flow_key] = peft_model.merge_and_unload()
    setattr(pipeline, flow_key, pipeline.models[flow_key])
    metadata.update(
        {
            "stage": stage,
            "flow_key_count": len(flow_state),
            "condition_key_count": len(cond_state),
            "lora_rank": rank,
            "lora_alpha": alpha,
            "flow_missing_count": len(incompatible.missing_keys),
            "flow_unexpected_count": len(incompatible.unexpected_keys),
            "condition_missing_count": len(cond_incompatible.missing_keys),
            "condition_unexpected_count": len(cond_incompatible.unexpected_keys),
        }
    )
    return metadata


def run(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    destination_dir = method_dir(protocol_path, args.method_id)
    result_path = destination_dir / "result.json"
    if result_path.is_file():
        result = load_method_result(protocol_path, args.method_id)
        print(
            json.dumps(
                {
                    "status": "reused",
                    "method_id": args.method_id,
                    "mesh": result["mesh"]["path"],
                },
                indent=2,
            )
        )
        return
    if destination_dir.exists() and any(destination_dir.iterdir()):
        raise RuntimeError(
            f"partial ReconViaGen output exists; preserve and inspect: "
            f"{destination_dir}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    from trellis.pipelines import TrellisVGGTTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = torch.device(args.device)
    pipeline.low_vram = bool(args.low_vram)
    checkpoint_audit: dict[str, Any] = {}
    if args.ss_checkpoint:
        checkpoint_audit["ss"] = load_native_lora(
            pipeline,
            stage="ss",
            checkpoint=args.ss_checkpoint.resolve(),
        )
    if args.slat_checkpoint:
        checkpoint_audit["slat"] = load_native_lora(
            pipeline,
            stage="slat",
            checkpoint=args.slat_checkpoint.resolve(),
        )
    if not pipeline.low_vram:
        for model in pipeline.models.values():
            model.to(pipeline._device)
        pipeline.VGGT_model.to(pipeline._device)

    view_indices = [int(value) for value in protocol["view_indices"]]
    image_paths = [
        Path(view["rgba"]["path"])
        for view in protocol["views"]
        if int(view["view_index"]) in view_indices
    ]
    raw_images = [Image.open(path).convert("RGBA") for path in image_paths]
    images = [pipeline.preprocess_image(image) for image in raw_images]
    print(
        f"[bunny_reconviagen] method={args.method_id} views={view_indices} "
        f"seed={args.seed}",
        flush=True,
    )
    outputs, coords, ss_noise = pipeline.run(
        image=images,
        seed=int(args.seed),
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": int(args.ss_steps),
            "cfg_strength": float(args.ss_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        slat_sampler_params={
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.slat_guidance_rescale),
            "rescale_t": float(args.slat_rescale_t),
        },
        mode=args.multiimage_algo,
    )
    decoded = outputs["mesh"][0]
    gaussian = outputs["gaussian"][0]
    canonical = decoded.to_trimesh(transform_pose=False)
    view_mesh = decoded.to_trimesh(transform_pose=True)
    canonical_path = destination_dir / "mesh_canonical.obj"
    view_path = destination_dir / "mesh_view.glb"
    export_mesh_atomic(canonical, canonical_path, file_type="obj")
    export_mesh_atomic(view_mesh, view_path, file_type="glb")

    auxiliary: dict[str, Path] = {"view_glb": view_path}
    if not args.skip_textured_glb:
        glb = postprocessing_utils.to_glb(
            gaussian,
            decoded,
            simplify=float(args.mesh_simplify),
            texture_size=int(args.texture_size),
            verbose=True,
        )
        textured_path = destination_dir / "mesh_textured.glb"
        temporary = textured_path.with_name(
            f".{textured_path.stem}.tmp-{os.getpid()}{textured_path.suffix}"
        )
        glb.export(temporary)
        os.replace(temporary, textured_path)
        auxiliary["textured_glb"] = textured_path
        del glb

    backend_kind = (
        "reconviagen_native_lora"
        if checkpoint_audit
        else "reconviagen_stock"
    )
    backend = {
        "kind": backend_kind,
        "pretrained": str(args.pretrained),
        "seed": int(args.seed),
        "input_view_indices": view_indices,
        "input_camera_calibrated": False,
        "pose_conditioning": "ReconViaGen VGGT aggregation from the five images",
        "multiimage_algo": args.multiimage_algo,
        "low_vram": bool(args.low_vram),
        "sampling": {
            "ss_steps": int(args.ss_steps),
            "ss_guidance": float(args.ss_guidance),
            "slat_steps": int(args.slat_steps),
            "slat_guidance": float(args.slat_guidance),
        },
        "sparse_coordinate_count": int(coords.shape[0]),
        "ss_noise_shape": list(ss_noise.shape),
        "vertex_count": int(len(canonical.vertices)),
        "face_count": int(len(canonical.faces)),
        "checkpoints": checkpoint_audit,
        "code_bindings": code_bindings(
            {
                "runner": Path(__file__).resolve(),
                "reconviagen_pipeline": (
                    RECONVIAGEN_ROOT
                    / "trellis"
                    / "pipelines"
                    / "trellis_image_to_3d.py"
                ),
            }
        ),
    }
    result_path = write_method_result(
        protocol_path=protocol_path,
        method_id=args.method_id,
        display_name=args.display_name,
        mesh_path=canonical_path,
        auxiliary_meshes=auxiliary,
        input_view_indices=view_indices,
        backend=backend,
        notes=[
            "All five frozen Bunny thumbnails are consumed as uncalibrated multi-view RGB.",
            "Canonical OBJ is primary; view-frame and optional textured GLB are auxiliary.",
        ],
    )
    del (
        canonical,
        view_mesh,
        decoded,
        gaussian,
        outputs,
        coords,
        ss_noise,
        pipeline,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "mesh": str(canonical_path),
                "mesh_sha256": sha256_file(canonical_path),
                "auxiliary": {key: str(path) for key, path in auxiliary.items()},
                "checkpoint_stages": sorted(checkpoint_audit),
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method_id", default="reconviagen_stock")
    parser.add_argument("--display_name", default="ReconViaGen stock (5 views)")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument(
        "--ss_checkpoint",
        type=Path,
        default=None,
        help="optional native ReconViaGen Lightning SS LoRA checkpoint",
    )
    parser.add_argument(
        "--slat_checkpoint",
        type=Path,
        default=None,
        help="optional native ReconViaGen Lightning SLAT LoRA checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_steps", type=int, default=30)
    parser.add_argument("--ss_guidance", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_guidance", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument(
        "--multiimage_algo",
        choices=("multidiffusion", "stochastic"),
        default="multidiffusion",
    )
    parser.add_argument("--mesh_simplify", type=float, default=0.95)
    parser.add_argument("--texture_size", type=int, default=1024)
    parser.add_argument("--skip_textured_glb", action="store_true")
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
