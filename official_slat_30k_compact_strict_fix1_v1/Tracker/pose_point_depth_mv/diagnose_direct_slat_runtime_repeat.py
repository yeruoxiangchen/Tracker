#!/usr/bin/env python3
"""Diagnose decoder and SLAT repeatability on frozen seen-object cases."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
NINJA_WRAPPER_ROOT = TRACKER_ROOT / "pose_point_depth_mv" / "tools"
if (NINJA_WRAPPER_ROOT / "ninja").is_file():
    os.environ["PATH"] = (
        f"{NINJA_WRAPPER_ROOT}{os.pathsep}{os.environ.get('PATH', '')}"
    )
for search_path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pose_point_depth_mv.direct_slat_blind import (  # noqa: E402
    PROTOCOL_FORMAT,
    atomic_json,
    canonical_sha256,
    pair_identity,
    runtime_selection_rows,
    sha256_file,
    validate_binding_tree,
)
from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset  # noqa: E402
from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_FLOW_VERSION,
    build_direct_slat_components,
    canonical_json_sha256,
    load_strict_trainable_state,
)
from pose_point_depth_mv.direct_slat_runtime_repeat import (  # noqa: E402
    PROCESS_REPORT_FORMAT,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    load_canonical_gt,
    mesh_structure_metrics,
    sparse_from_payload,
    sparse_noise_from_master,
    sparse_payload,
    surface_metrics,
    tensor_sha256,
)
from pose_point_depth_mv.export_direct_slat_blind_holdout import (  # noqa: E402
    atomic_torch_save,
    completion_valid,
    load_coord_pair,
    sample_slat,
)
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--blind_key_file", required=True)
    parser.add_argument("--source_preflight_output", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--runtime_id", required=True)
    parser.add_argument("--process_index", type=int, required=True)
    parser.add_argument("--object_uids", required=True)
    parser.add_argument("--joint_seeds", required=True)
    parser.add_argument("--branches", default="stock,full")
    parser.add_argument("--decoder_repeats", type=int, default=5)
    parser.add_argument("--slat_repeats", type=int, default=5)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--attention_backend", choices=("flash_attn", "xformers"), required=True
    )
    parser.add_argument(
        "--spconv_algo", choices=("native", "implicit_gemm"), required=True
    )
    parser.add_argument(
        "--amp_dtype", choices=("bf16", "fp16", "none"), required=True
    )
    parser.add_argument(
        "--deterministic_algorithms", choices=("off", "warn", "on"), default="off"
    )
    parser.add_argument("--allow_tf32", choices=("false", "true"), default="false")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_unique_csv(value: str, cast) -> list[Any]:
    parsed = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("CSV values must be non-empty and unique")
    return parsed


def load_source_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    body = dict(protocol)
    saved = str(body.pop("protocol_sha256", ""))
    if protocol.get("format") != PROTOCOL_FORMAT or canonical_sha256(body) != saved:
        raise RuntimeError("source preflight protocol is invalid")
    # Its code bindings intentionally describe the preserved v2_fix1 run, not
    # the current diagnostic implementation.
    validate_binding_tree(protocol["bindings"], "bindings")
    validate_binding_tree(protocol["sample_bindings"], "sample_bindings")
    validate_binding_tree(protocol["runtime_bindings"], "runtime_bindings")
    return protocol


def configure_runtime(args: argparse.Namespace) -> dict[str, Any]:
    actual_attention = os.environ.get("ATTN_BACKEND", "flash_attn")
    actual_sparse_attention = os.environ.get(
        "SPARSE_ATTN_BACKEND", actual_attention
    )
    actual_spconv = os.environ.get("SPCONV_ALGO", "native")
    if actual_attention != args.attention_backend:
        raise RuntimeError(
            f"ATTN_BACKEND={actual_attention!r}, expected {args.attention_backend!r}"
        )
    if actual_sparse_attention != args.attention_backend:
        raise RuntimeError(
            "SPARSE_ATTN_BACKEND differs from the requested attention backend"
        )
    if actual_spconv != args.spconv_algo:
        raise RuntimeError(
            f"SPCONV_ALGO={actual_spconv!r}, expected {args.spconv_algo!r}"
        )
    if args.deterministic_algorithms == "off":
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(
            True, warn_only=args.deterministic_algorithms == "warn"
        )
    allow_tf32 = args.allow_tf32 == "true"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    return {
        "runtime_id": str(args.runtime_id),
        "attention_backend": actual_attention,
        "sparse_attention_backend": actual_sparse_attention,
        "spconv_algo": actual_spconv,
        "amp_dtype": str(args.amp_dtype),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms": str(args.deterministic_algorithms),
        "allow_tf32": allow_tf32,
        "cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
    }


def latent_identity(path: Path, payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "coords_sha256": tensor_sha256(payload["coords"]),
        "features_sha256": tensor_sha256(payload["feats"]),
        "coord_count": int(payload["coords"].shape[0]),
        "feature_dtype": str(payload["feats"].dtype),
    }


def hard_integrity(structure: dict[str, Any], surface: dict[str, Any]) -> dict[str, bool]:
    surface_finite = bool(
        np.isfinite([float(value) for value in surface.values()]).all()
    )
    checks = {
        "mesh_success": bool(structure["mesh_success"]),
        "vertices_finite": bool(structure["vertices_finite"]),
        "winding_consistent": bool(structure["is_winding_consistent"]),
        "surface_metrics_finite": surface_finite,
        "output_complete": True,
    }
    return {**checks, "passed": bool(all(checks.values()))}


@torch.no_grad()
def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    if args.decoder_repeats < 1 or args.slat_repeats < 1:
        raise ValueError("repeat counts must be positive")
    if args.process_index < 0:
        raise ValueError("process_index must be nonnegative")
    runtime = configure_runtime(args)
    protocol_path = Path(args.protocol).resolve()
    protocol = load_source_protocol(protocol_path)
    key_path = Path(args.blind_key_file).resolve()
    blind_key = bytes.fromhex(key_path.read_text(encoding="ascii").strip())
    source_root = Path(args.source_preflight_output).resolve()
    completion = json.loads(
        (source_root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    completion_valid(source_root, completion)
    if completion.get("complete") is not True or completion.get("mode") != "preflight":
        raise RuntimeError("source output is not a complete preflight artifact")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    atomic_json(
        output_dir / "run_identity.json",
        {
            "format": PROCESS_REPORT_FORMAT,
            "complete": False,
            "protocol_sha256": protocol["protocol_sha256"],
            "source_completion_sha256": sha256_file(
                source_root / "completion_manifest.json"
            ),
            "runtime": runtime,
            "process_index": int(args.process_index),
        },
    )

    def progress(stage: str, **details: Any) -> None:
        payload = {
            "format": "pose_point_depth_mv.direct_slat_runtime_progress.v1",
            "complete": stage == "complete",
            "stage": stage,
            "runtime_id": runtime["runtime_id"],
            "process_index": int(args.process_index),
            "protocol_sha256": protocol["protocol_sha256"],
            "details": details,
        }
        atomic_json(output_dir / "progress.json", payload)
        print(
            "[runtime_repeat:stage] "
            + json.dumps(
                {
                    "stage": stage,
                    "runtime_id": runtime["runtime_id"],
                    "process_index": int(args.process_index),
                    **details,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    progress("output_initialized")

    object_uids = set(parse_unique_csv(args.object_uids, str))
    seeds = parse_unique_csv(args.joint_seeds, int)
    branches = parse_unique_csv(args.branches, str)
    if set(branches) - {"stock", "full"}:
        raise ValueError(f"unsupported branches={branches}")
    selected = [
        row
        for row in runtime_selection_rows(protocol)
        if str(row["object_uid"]) in object_uids
    ]
    if {str(row["object_uid"]) for row in selected} != object_uids:
        raise RuntimeError("requested diagnostic object is absent from preflight")
    if set(seeds) - set(int(value) for value in protocol["sampling"]["joint_seeds"]):
        raise RuntimeError("requested diagnostic seed is absent from preflight")
    progress(
        "case_selection_validated",
        object_uids=sorted(object_uids),
        joint_seeds=seeds,
        branches=branches,
    )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("runtime repeat diagnosis requires CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    dataset = DirectSLatCacheDataset(
        protocol["bindings"]["holdout_cache_manifest"]["path"],
        verify_hashes=False,
    )
    progress("dataset_loaded", dataset_size=len(dataset))
    checkpoint = torch.load(
        protocol["bindings"]["direct_slat_checkpoint"]["path"],
        map_location="cpu",
    )
    if checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION:
        raise RuntimeError("Direct-SLAT checkpoint format differs")
    progress("checkpoint_loaded")
    saved_args = dict(checkpoint["args"])
    sampler, model, defaults, normalization, _, pipeline = build_direct_slat_components(
        pretrained=protocol["pretrained"],
        adapter_hidden_dim=int(saved_args["adapter_hidden_dim"]),
        lora_rank=int(saved_args["lora_rank"]),
        lora_alpha=int(saved_args["lora_alpha"]),
        gradient_checkpointing=False,
        device=device,
        retain_pipeline=True,
    )
    progress("components_built")
    runtime_normalization = {
        key: [float(item) for item in value] for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from source cache")
    load_strict_trainable_state(model, checkpoint["model_trainable_state"])
    model.eval()
    progress("model_ready")
    params = dict(defaults)
    params.update(
        {
            key: value
            for key, value in protocol["sampling"]["slat"].items()
            if key in {"steps", "cfg_strength", "cfg_interval", "rescale_t"}
        }
    )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    latent_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for frozen in selected:
        object_position = int(frozen["object_position"])
        for seed in seeds:
            pair_id, mapping = pair_identity(
                protocol["protocol_name"], str(frozen["uid"]), seed, blind_key
            )
            coords = load_coord_pair(source_root / "sealed" / "coords" / f"{pair_id}.npz")
            cache_index = int(frozen["cache_indices"][str(seed)])
            sample = dataset[cache_index]
            condition = to_device_tree(sample["condition"], device)
            support = (
                sample["corrected_ss"].to(device=device),
                sample["occupancy_logits64"].to(device=device),
                sample["physical_tokens16"].to(device=device),
            )
            master_seed = int(seed) * 2000003 + object_position * 2017 + 7919
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (64, 64, 64, 8),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            for side in ("A", "B"):
                branch = str(mapping[side])
                if branch not in branches:
                    continue
                initial = sparse_noise_from_master(coords[side], master, device=device)
                case = {
                    "pair_id": pair_id,
                    "side": side,
                    "branch": branch,
                    "object_uid": str(frozen["object_uid"]),
                    "uid": str(frozen["uid"]),
                    "object_position": object_position,
                    "seed": int(seed),
                    "cache_index": cache_index,
                    "master_seed": master_seed,
                    "initial_noise_sha256": tensor_sha256(initial.feats),
                }
                case_rows.append(case)
                for run_index in range(int(args.slat_repeats)):
                    progress(
                        "slat_sample_start",
                        pair_id=pair_id,
                        side=side,
                        branch=branch,
                        seed=int(seed),
                        run_index=run_index,
                    )
                    sampled = sample_slat(
                        branch=branch,
                        model=model,
                        sampler=sampler,
                        condition=condition,
                        support=support,
                        initial=initial,
                        params=params,
                        normalization=runtime_normalization,
                        amp_dtype=amp_dtype,
                        use_amp=use_amp,
                    )
                    progress(
                        "slat_sample_returned",
                        pair_id=pair_id,
                        side=side,
                        branch=branch,
                        seed=int(seed),
                        run_index=run_index,
                    )
                    relative = (
                        Path("latents")
                        / pair_id
                        / side
                        / f"run_{run_index:03d}.pt"
                    )
                    destination = output_dir / relative
                    atomic_torch_save(sparse_payload(sampled), destination)
                    payload = torch.load(destination, map_location="cpu")
                    latent_rows.append(
                        {
                            **case,
                            "run_index": run_index,
                            "latent": {
                                **latent_identity(destination, payload),
                                "path": str(relative),
                            },
                        }
                    )
                    del sampled, payload
                del initial
            del sample, condition, support, master
            torch.cuda.empty_cache()
            print(
                f"[runtime_repeat:slat] {runtime['runtime_id']} "
                f"process={args.process_index} pair={pair_id}",
                flush=True,
            )

    progress("slat_generation_complete", case_count=len(case_rows))
    model.cpu()
    pipeline.models["slat_flow_model"].cpu()
    del model, sampler
    gc.collect()
    torch.cuda.empty_cache()
    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    progress("mesh_decoder_ready")
    records: list[dict[str, Any]] = []

    def decode_record(
        *,
        stage: str,
        case: dict[str, Any],
        run_index: int,
        latent_path: Path,
        stored_path: str,
    ) -> dict[str, Any]:
        progress(
            "mesh_decode_start",
            decode_stage=stage,
            pair_id=case["pair_id"],
            side=case["side"],
            branch=case["branch"],
            run_index=int(run_index),
        )
        payload = torch.load(latent_path, map_location="cpu")
        decoded = mesh_decoder(sparse_from_payload(payload, device))[0]
        mesh = decoded.to_trimesh(transform_pose=False)
        structure = mesh_structure_metrics(mesh)
        target_sample = dataset[int(case["cache_index"])]
        target_mesh, _ = load_canonical_gt(target_sample)
        metric_seed = int(case["seed"]) * 1009 + int(case["object_position"]) * 9173
        surface = surface_metrics(
            mesh,
            target_mesh,
            count=int(args.surface_samples),
            seed=metric_seed,
            thresholds=(0.01, 0.02, 0.05),
        )
        integrity = hard_integrity(structure, surface)
        result = {
            "stage": stage,
            **case,
            "run_index": int(run_index),
            "latent": {
                **latent_identity(latent_path, payload),
                "path": stored_path,
            },
            "structure": structure,
            "surface": surface,
            "hard_integrity": integrity,
        }
        del payload, decoded, mesh, target_sample, target_mesh
        torch.cuda.empty_cache()
        progress(
            "mesh_decode_complete",
            decode_stage=stage,
            pair_id=case["pair_id"],
            side=case["side"],
            branch=case["branch"],
            run_index=int(run_index),
        )
        return result

    by_case = {
        (row["pair_id"], row["side"]): row for row in case_rows
    }
    for case in case_rows:
        pair_id, side = str(case["pair_id"]), str(case["side"])
        baseline = source_root / "sealed" / "latents" / pair_id / f"{side}.pt"
        baseline_sha = sha256_file(baseline)
        for run_index in range(int(args.decoder_repeats)):
            records.append(
                decode_record(
                    stage="decoder_only",
                    case=case,
                    run_index=run_index,
                    latent_path=baseline,
                    stored_path=str(baseline),
                )
            )
            if records[-1]["latent"]["sha256"] != baseline_sha:
                raise RuntimeError("decoder-only input latent changed during diagnosis")
    for latent_row in latent_rows:
        case = by_case[(latent_row["pair_id"], latent_row["side"])]
        relative = str(latent_row["latent"]["path"])
        records.append(
            decode_record(
                stage="slat",
                case=case,
                run_index=int(latent_row["run_index"]),
                latent_path=output_dir / relative,
                stored_path=relative,
            )
        )
    mesh_decoder.cpu()
    del mesh_decoder, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    if not all(row["hard_integrity"]["passed"] for row in records):
        runtime_complete = False
    else:
        runtime_complete = True
    code_paths = (
        Path(__file__).resolve(),
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_slat_runtime_repeat.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "export_direct_slat_blind_holdout.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_slat_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "export_direct_flow_mesh_pairs.py",
    )
    report = {
        "format": PROCESS_REPORT_FORMAT,
        "complete": runtime_complete,
        "formal": False,
        "science_decision_emitted": False,
        "process_index": int(args.process_index),
        "runtime": runtime,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "source_preflight_completion": {
            "path": str(source_root / "completion_manifest.json"),
            "sha256": sha256_file(source_root / "completion_manifest.json"),
        },
        "case_selection": {
            "object_uids": sorted(object_uids),
            "joint_seeds": seeds,
            "branches": branches,
            "case_count": len(case_rows),
        },
        "decoder_repeats": int(args.decoder_repeats),
        "slat_repeats": int(args.slat_repeats),
        "surface_samples": int(args.surface_samples),
        "code_bindings": {
            str(path.relative_to(TRACKER_ROOT)): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in code_paths
        },
        "records": records,
        "scope_guard": (
            "seen-object runtime repeatability engineering only; this report cannot "
            "support a stock/full science claim"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output_dir / "report.json", report)
    atomic_json(
        output_dir / "run_identity.json",
        {
            "format": PROCESS_REPORT_FORMAT,
            "complete": runtime_complete,
            "protocol_sha256": protocol["protocol_sha256"],
            "source_completion_sha256": report["source_preflight_completion"][
                "sha256"
            ],
            "runtime": runtime,
            "process_index": int(args.process_index),
            "report_sha256": report["report_sha256"],
        },
    )
    progress("complete", report_sha256=report["report_sha256"])
    print(
        json.dumps(
            {
                "runtime_id": runtime["runtime_id"],
                "process_index": int(args.process_index),
                "cases": len(case_rows),
                "decoder_records": sum(
                    row["stage"] == "decoder_only" for row in records
                ),
                "slat_records": sum(row["stage"] == "slat" for row in records),
                "complete": runtime_complete,
                "report": str(output_dir / "report.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    if not runtime_complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
