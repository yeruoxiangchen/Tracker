#!/usr/bin/env python3
"""Freeze the exact T3 Native/Stock object set for external baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    FORMAT,
    atomic_json,
    binding,
    canonical_sha256,
    mask_array,
    sha256_file,
    validate_protocol,
)
from pose_point_depth_mv.prepare_native_ss_pixal3d_review import (
    SPLIT_FORMAT,
    T3_FORMAT,
    export_target_atomic,
    resolve_cache_file,
    save_rgba_atomic,
    tensor_list,
)


SELECTION_MODE = "t3_exact_objects_matched_seed_v1"


def validate_matched_protocol(path: Path) -> dict[str, Any]:
    protocol = validate_protocol(path.resolve())
    comparison = protocol.get("comparison", {})
    if comparison.get("selection_mode") != SELECTION_MODE:
        raise RuntimeError("protocol is not the exact T3 matched baseline set")
    if comparison.get("mesh_branches") != {"full": "native", "stock": "stock"}:
        raise RuntimeError("unexpected Native/Stock branch mapping")
    if comparison.get("selection_is_exact_t3_run_identity") is not True:
        raise RuntimeError("protocol does not bind the exact T3 run identity")
    if int(comparison.get("object_count", 0)) != len(protocol.get("cases", [])):
        raise RuntimeError("matched baseline object count changed")
    for case in protocol["cases"]:
        for key in ("native_full_mesh", "stock_mesh", "pair_record", "cache_payload"):
            item = case.get(key, {})
            source = Path(str(item.get("path", "")))
            if not source.is_file() or sha256_file(source) != item.get("sha256"):
                raise RuntimeError(f"matched case binding changed: {case['case_id']}.{key}")
        if case.get("current_mesh") != case.get("native_full_mesh"):
            raise RuntimeError("Pixal current_mesh must bind Native Full")
    return protocol


def command_prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        protocol = validate_matched_protocol(protocol_path)
        expected = {
            "t3_report": binding(args.t3_report.resolve()),
            "cache_manifest": binding(args.cache_manifest.resolve()),
            "split_audit": binding(args.split_audit.resolve()),
        }
        if any(protocol["bindings"].get(key) != value for key, value in expected.items()):
            raise RuntimeError("existing matched protocol input bindings changed")
        if protocol["comparison"].get("mesh_seed") != int(args.seed):
            raise RuntimeError("existing matched protocol seed changed")
        if len(protocol["cases"]) != int(args.expected_objects):
            raise RuntimeError("existing matched protocol object count changed")
        print(
            json.dumps(
                {
                    "status": "reused",
                    "protocol": str(protocol_path),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "cases": len(protocol["cases"]),
                },
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        unexpected = {
            item.name for item in output_dir.iterdir() if item.name not in {"inputs", "targets"}
        }
        if unexpected:
            raise RuntimeError(
                f"partial matched protocol has unexpected files {sorted(unexpected)}: {output_dir}"
            )
        print(f"resume inspected partial protocol preparation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    t3_path = args.t3_report.resolve()
    manifest_path = args.cache_manifest.resolve()
    split_path = args.split_audit.resolve()
    t3 = json.loads(t3_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if t3.get("format") != T3_FORMAT or t3.get("runtime_passed") is not True:
        raise RuntimeError("T3 report is unsupported or incomplete")
    if split.get("format") != SPLIT_FORMAT or split.get("passed") is not True:
        raise RuntimeError("source-balanced split audit is unsupported or failed")
    run_identity = t3.get("run_identity", {})
    if (
        Path(str(run_identity.get("cache_manifest", ""))).resolve() != manifest_path
        or run_identity.get("cache_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("T3 report is not bound to the requested cache manifest")
    split_identity = run_identity.get("split", {})
    if (
        Path(str(split_identity.get("path", ""))).resolve() != split_path
        or split_identity.get("sha256") != sha256_file(split_path)
        or split_identity.get("phase") != "final"
    ):
        raise RuntimeError("T3 report is not bound to the final split")
    selected_uids = [str(value) for value in run_identity.get("selected_uids", [])]
    if len(selected_uids) != int(args.expected_objects) or len(set(selected_uids)) != len(selected_uids):
        raise RuntimeError("T3 selected_uids are not the expected unique object set")
    if int(args.seed) not in [int(value) for value in run_identity.get("joint_seeds", [])]:
        raise RuntimeError("requested matched seed was not evaluated by T3")

    manifest_by_uid = {str(row["uid"]): row for row in manifest["samples"]}
    split_by_uid = {
        str(row["uid"]): row for row in split.get("phases", {}).get("final", [])
    }
    split_position_by_uid = {
        str(row["uid"]): position
        for position, row in enumerate(split.get("phases", {}).get("final", []))
    }
    support_by_key = {
        (str(row["uid"]), int(row["seed"])): row
        for row in t3.get("ss_support_records", [])
    }
    t3_root = t3_path.parent
    cases = []
    for position, uid in enumerate(selected_uids):
        print(
            f"[matched_baseline_protocol] {position + 1}/{len(selected_uids)} {uid}",
            flush=True,
        )
        manifest_row = manifest_by_uid.get(uid)
        selected = split_by_uid.get(uid)
        support = support_by_key.get((uid, int(args.seed)))
        if manifest_row is None or selected is None or support is None:
            raise RuntimeError(f"T3 exact object lacks manifest/split/support row: {uid}")
        if support.get("passed") is not True:
            raise RuntimeError(f"T3 support failed for matched seed: {uid}")
        source = str(selected["source"])
        view_count = int(selected["view_count"])
        pair_id = str(support["pair_id"])
        pair_path = (t3_root / "mesh_pairs" / pair_id / "pair_record.json").resolve()
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        if (
            pair.get("passed") is not True
            or str(pair.get("uid")) != uid
            or int(pair.get("seed", -1)) != int(args.seed)
        ):
            raise RuntimeError(f"T3 pair identity failed: {pair_path}")
        branches = {str(row["branch"]): row for row in pair.get("branches", [])}
        if set(branches) != {"stock", "native"}:
            raise RuntimeError(f"T3 pair lacks exact stock/native branches: {pair_path}")
        native_path = Path(str(branches["native"]["canonical_obj"])).resolve()
        stock_path = Path(str(branches["stock"]["canonical_obj"])).resolve()
        for branch, mesh_path in (("native", native_path), ("stock", stock_path)):
            if (
                not mesh_path.is_file()
                or branches[branch].get("canonical_obj_sha256") != sha256_file(mesh_path)
            ):
                raise RuntimeError(f"T3 {branch} mesh binding failed: {mesh_path}")

        cache_path = resolve_cache_file(manifest, manifest_row)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            str(payload.get("uid")) != uid
            or str(payload.get("object_uid")) != str(selected["object_uid"])
            or str(payload.get("extrinsics_type")) != "c2w"
        ):
            raise RuntimeError(f"cache identity/frame failed: {cache_path}")
        image_paths = [Path(str(value)).resolve() for value in payload["image_paths"]]
        mask_paths = [Path(str(value)).resolve() for value in payload["mask_paths"]]
        if len(image_paths) != view_count or len(mask_paths) != view_count:
            raise RuntimeError(f"cache view count differs for uid={uid}")
        mask_areas = [
            int(np.count_nonzero(mask_array(mask_path) > 127))
            for mask_path in mask_paths
        ]
        selected_input_position = max(
            range(view_count), key=lambda index: (mask_areas[index], -index)
        )
        image_path = image_paths[selected_input_position]
        mask_path = mask_paths[selected_input_position]
        short_uid = str(selected["object_uid"]).replace("objaverse_", "")[:12]
        case_id = f"m{position:02d}_{source}_v{view_count:02d}_{short_uid}_s{int(args.seed)}"
        rgba_path = (output_dir / "inputs" / f"{case_id}.png").resolve()
        image_metadata = save_rgba_atomic(image_path, mask_path, rgba_path)
        target_path = (output_dir / "targets" / f"{case_id}.obj").resolve()
        reused_target_export = target_path.is_file() and target_path.stat().st_size > 0
        if not reused_target_export:
            export_target_atomic(payload["ss_latent"], target_path)
        # Always bind the frozen T3 target metadata.  This keeps a resumed
        # protocol byte-identical to a one-pass preparation; export metadata
        # returned by the helper must not depend on whether the OBJ existed.
        target_metadata = dict(branches["native"].get("target", {}))

        extrinsics = tensor_list(payload["extrinsics"])
        intrinsics = tensor_list(payload["source_intrinsics"])
        view_ids = [int(value) for value in tensor_list(payload["view_ids"])]
        selected_extrinsic = extrinsics[selected_input_position]
        selected_intrinsic = intrinsics[selected_input_position]
        native_binding = binding(native_path)
        cases.append(
            {
                "case_id": case_id,
                "uid": uid,
                "object_uid": str(selected["object_uid"]),
                "source": source,
                "view_count": view_count,
                "current_seed": int(args.seed),
                "pixal3d_seed": int(args.seed),
                "dataset_index": int(manifest["samples"].index(manifest_row)),
                "pair_id": pair_id,
                "t3_run_identity_position": position,
                "t3_split_phase_position": int(split_position_by_uid[uid]),
                "single_view_policy": "largest foreground mask among exact Native inputs",
                "selected_input_position": int(selected_input_position),
                "selected_view_id": int(view_ids[selected_input_position]),
                "selected_frame": {
                    "extrinsics_type": "c2w",
                    "extrinsic": selected_extrinsic,
                    "source_view_index": int(view_ids[selected_input_position]),
                },
                "selected_intrinsic": selected_intrinsic,
                "mask_foreground_pixels": image_metadata["foreground_pixels"],
                "source_width": image_metadata["width"],
                "source_height": image_metadata["height"],
                "source_fx_pixels": float(selected_intrinsic[0][0]),
                "input_rgba": binding(rgba_path),
                "source_image": binding(image_path),
                "source_mask": binding(mask_path),
                "current_mesh": native_binding,
                "native_full_mesh": native_binding,
                "stock_mesh": binding(stock_path),
                "target_mesh": binding(target_path),
                "target_metadata": target_metadata,
                "pair_record": binding(pair_path),
                "cache_payload": binding(cache_path),
                "t3_ss_support": support,
                "t3_mesh_surface": {
                    branch: branches[branch].get("surface")
                    for branch in ("native", "stock")
                },
            }
        )

    body = {
        "format": FORMAT,
        "formal": False,
        "purpose": (
            "retrospective matched external-baseline evaluation on the exact "
            "T3 Native Full/Stock object set and seed"
        ),
        "comparison": {
            "selection_mode": SELECTION_MODE,
            "selection_is_exact_t3_run_identity": True,
            "selection_is_performance_independent": False,
            "selection_pool": "exact T3 final32 run_identity.selected_uids in original order",
            "object_count": len(cases),
            "mesh_seed": int(args.seed),
            "t3_available_joint_seeds": [int(value) for value in run_identity["joint_seeds"]],
            "baseline_seed_policy": (
                "one matched seed to bound cost; do not merge with T3 three-seed "
                "object-averaged statistics"
            ),
            "pixal3d_input_budget": 1,
            "genrecon_input_budget": "all exact posed views",
            "single_view_policy": "largest foreground mask among exact Native inputs",
            "mesh_branches": {"full": "native", "stock": "stock"},
            "primary_metric_frame": "canonical pose without per-method fitting",
        },
        "bindings": {
            "t3_report": binding(t3_path),
            "cache_manifest": binding(manifest_path),
            "split_audit": binding(split_path),
            "prepare_code": binding(Path(__file__).resolve()),
        },
        "cases": cases,
    }
    body["protocol_sha256"] = canonical_sha256(body)
    atomic_json(protocol_path, body)
    protocol = validate_matched_protocol(protocol_path)
    print(
        json.dumps(
            {
                "status": "created",
                "protocol": str(protocol_path),
                "protocol_sha256": protocol["protocol_sha256"],
                "cases": len(protocol["cases"]),
                "seed": int(args.seed),
            },
            indent=2,
        )
    )


def command_verify(args: argparse.Namespace) -> None:
    protocol = validate_matched_protocol(args.protocol.resolve())
    print(
        json.dumps(
            {
                "passed": True,
                "protocol": str(args.protocol.resolve()),
                "protocol_sha256": protocol["protocol_sha256"],
                "cases": len(protocol["cases"]),
                "seed": protocol["comparison"]["mesh_seed"],
                "sources": {
                    source: sum(case["source"] == source for case in protocol["cases"])
                    for source in sorted({case["source"] for case in protocol["cases"]})
                },
                "view_counts": {
                    str(view): sum(case["view_count"] == view for case in protocol["cases"])
                    for view in sorted({case["view_count"] for case in protocol["cases"]})
                },
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--t3_report", type=Path, required=True)
    prepare.add_argument("--cache_manifest", type=Path, required=True)
    prepare.add_argument("--split_audit", type=Path, required=True)
    prepare.add_argument("--output_dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--expected_objects", type=int, default=32)
    prepare.set_defaults(handler=command_prepare)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.set_defaults(handler=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
