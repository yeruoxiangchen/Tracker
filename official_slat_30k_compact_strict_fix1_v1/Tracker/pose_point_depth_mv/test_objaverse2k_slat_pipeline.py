#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from pose_point_depth_mv.build_local_lh_slats import (
    LOCAL_LH_SLAT_VERSION,
    NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
    NATIVE_OBJAVERSE_TARGET_CONTRACT,
)
from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_CACHE_VERSION,
    canonical_json_sha256,
)
from pose_point_depth_mv.objaverse2k_slat_pipeline import (
    CACHE_MARKER,
    SPLIT_MARKER,
    TARGET_MARKER,
    TARGET_PREFLIGHT,
    assign_object_workers,
    command_finalize_targets,
    command_merge_cache,
    command_prepare,
    command_preflight_targets,
    load_json,
    resolve_native_objaverse_normalization_bindings,
    sha256_file,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class Objaverse2KSLatPipelineTest(unittest.TestCase):
    def _source_bundle(self, root: Path) -> tuple[Path, Path, str]:
        rows = []
        render_rows = []
        latent_root = root / "render" / "ss_latents"
        latent_root.mkdir(parents=True)
        source_glb = root / "render" / "mesh.glb"
        source_glb.write_bytes(b"mesh")
        for index in range(8):
            object_uid = f"obj_{index:02d}"
            for sequence in range(1 + index % 2):
                uid = f"{object_uid}_seq{sequence:03d}"
                latent = latent_root / f"{uid}.npz"
                coords = np.asarray(
                    [[value, index, 1] for value in range(1, 9)], dtype=np.int32
                )
                if sequence:
                    coords[-1] = [9, index, 1]
                np.savez_compressed(
                    latent,
                    target_coords=coords,
                    z=np.full(
                        (8, 16, 16, 16), sequence, dtype=np.float16
                    ),
                    pixal3d_rotation=np.eye(3, dtype=np.float32),
                    source_glb=np.array(str(source_glb)),
                    uid=np.array(uid),
                    object_uid=np.array(object_uid),
                    renderer=np.array("blender"),
                )
                rows.append(
                    {
                        "uid": uid,
                        "object_uid": object_uid,
                        "cache_file": str(root / "cache" / f"{uid}.pt"),
                        "ss_latent": str(latent),
                    }
                )
                render_rows.append(
                    {
                        "uid": uid,
                        "object_uid": object_uid,
                        "ss_latent": f"{uid}.npz",
                        "source_glb": str(source_glb),
                        "num_voxels": len(coords),
                        "quality_flags": {"renderer": "blender"},
                        "renderer_audit": {
                            "normalization_policy": "imported_frame_center_scale_v2"
                        },
                        "frames": [],
                    }
                )
        source = {
            "format": LIFTING_CACHE_VERSION,
            "output_dir": str(root / "source"),
            "sample_count": len(rows),
            "object_count": 8,
            "failure_count": 0,
            "metadata_names": [],
            "metadata_schema_hash": "schema",
            "visual_feature_dim": 1024,
            "feature_metadata": {},
            "config": {"builder": "test"},
            "config_hash": "lifting-config",
            "samples": rows,
            "passed": True,
            "training_ready": True,
        }
        source_path = root / "source" / "lifting_manifest.json"
        write_json(source_path, source)
        audit_rows = [row for row in rows if row["object_uid"] == "obj_00"][:1]
        audit = {
            **source,
            "output_dir": str(root / "audit"),
            "sample_count": 1,
            "object_count": 1,
            "samples": audit_rows,
        }
        audit_path = root / "audit" / "lifting_manifest.json"
        write_json(audit_path, audit)
        render = {
            "format": "pixal3d_multiview.objaverse_sparse.v1",
            "extrinsics_type": "c2w",
            "renderer": "blender",
            "image_root": str(root / "render" / "images"),
            "mask_root": str(root / "render" / "masks"),
            "latent_root": str(latent_root),
            "voxel_resolution": 64,
            "image_size": 512,
            "camera_forward_sign": 1.0,
            "trajectory_mode": "ar_random",
            "num_views": 8,
            "canonical_latent_frame": "pixal3d_sparse_structure",
            "build_config": {"selected_views": 8, "canonical_margin": 0.9},
            "code_bindings": {"dataset_builder": {"sha256": "a" * 64}},
            "samples": render_rows,
        }
        render_path = root / "render" / "manifest.json"
        write_json(render_path, render)
        return source_path, audit_path, str(render_path)

    def _prepare(self, root: Path) -> Path:
        source, audit, render = self._source_bundle(root)
        output = root / "split"
        command_prepare(
            argparse.Namespace(
                source_lifting_manifest=str(source),
                audit_lifting_manifest=str(audit),
                render_manifests=render,
                output_dir=str(output),
                dev_objects=2,
                seed=20260811,
                num_workers=2,
                expected_source_objects=8,
                expected_source_samples=12,
                expected_audit_objects=1,
            )
        )
        command_preflight_targets(
            argparse.Namespace(
                split_bundle=str(output),
                render_manifests=render,
                min_native_sequence_target_iou=0.75,
            )
        )
        return output

    def test_object_worker_partition_never_splits_sequences(self) -> None:
        rows = [
            {"uid": "a0", "object_uid": "a"},
            {"uid": "a1", "object_uid": "a"},
            {"uid": "b0", "object_uid": "b"},
            {"uid": "c0", "object_uid": "c"},
            {"uid": "c1", "object_uid": "c"},
        ]
        workers = assign_object_workers(rows, 2)
        owners = {
            object_uid: worker["worker_index"]
            for worker in workers
            for object_uid in worker["object_uids"]
        }
        self.assertEqual(set(owners), {"a", "b", "c"})
        for worker in workers:
            self.assertEqual(
                {
                    rows[index]["object_uid"]
                    for index in worker["indices"]
                },
                set(worker["object_uids"]),
            )

    def test_prepare_keeps_audit_objects_in_train_and_freezes_dev(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self._prepare(Path(directory))
            marker = load_json(output / SPLIT_MARKER)
            train = load_json(marker["manifests"]["train"]["path"])
            dev = load_json(marker["manifests"]["dev"]["path"])
            train_objects = {row["object_uid"] for row in train["samples"]}
            dev_objects = {row["object_uid"] for row in dev["samples"]}
            self.assertIn("obj_00", train_objects)
            self.assertNotIn("obj_00", dev_objects)
            self.assertEqual(len(dev_objects), 2)
            self.assertFalse(train_objects.intersection(dev_objects))
            self.assertEqual(len(train["objaverse2k_split"]["workers"]), 2)
            preflight = load_json(output / TARGET_PREFLIGHT)
            self.assertTrue(preflight["passed"])
            self.assertEqual(preflight["object_count"], 8)
            self.assertEqual(preflight["sequence_target_iou"]["minimum"], 7 / 9)
            self.assertEqual(
                [
                    (row["object_uid"], row["uid"])
                    for row in train["samples"]
                ],
                sorted(
                    (row["object_uid"], row["uid"])
                    for row in train["samples"]
                ),
            )

    def test_legacy_margin_resolves_only_through_frozen_render_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = self._prepare(root)
            source_path = split_root / "dev" / "lifting_manifest.json"
            source = load_json(source_path)
            row = source["samples"][0]
            source_glb = Path(root / "render" / "mesh.glb").resolve()
            cache = {
                "source_lifting_manifest": str(source_path),
                "source_lifting_manifest_sha256": sha256_file(source_path),
                "objects": [
                    {
                        "object_uid": row["object_uid"],
                        "ss_latent": row["ss_latent"],
                        "source_glb": str(source_glb),
                        "source_glb_sha256": sha256_file(source_glb),
                    }
                ],
            }
            cache_path = root / "cache" / "manifest.json"
            write_json(cache_path, cache)
            bindings = resolve_native_objaverse_normalization_bindings(
                cache_path, cache, cache["objects"]
            )
            binding = bindings[str(Path(row["ss_latent"]).resolve())]
            self.assertEqual(binding["canonical_margin"], 0.9)
            self.assertEqual(
                binding["normalization_policy"],
                "imported_frame_center_scale_v2",
            )

            inventory_path = split_root / "render_manifest_inventory.json"
            inventory = load_json(inventory_path)
            render_path = Path(inventory["manifests"][0]["path"])
            render = load_json(render_path)
            render["build_config"]["canonical_margin"] = 0.8
            write_json(render_path, render)
            with self.assertRaisesRegex(RuntimeError, "render manifest changed"):
                resolve_native_objaverse_normalization_bindings(
                    cache_path, cache, cache["objects"]
                )

    def test_non_objaverse_cache_does_not_require_render_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "lifting_manifest.json"
            write_json(source_path, {"format": LIFTING_CACHE_VERSION})
            cache = {
                "source_lifting_manifest": str(source_path),
                "source_lifting_manifest_sha256": sha256_file(source_path),
            }
            self.assertEqual(
                resolve_native_objaverse_normalization_bindings(
                    root / "cache_manifest.json", cache, []
                ),
                {},
            )

    def test_finalize_targets_requires_complete_native_contract_union(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = self._prepare(root)
            marker = load_json(split_root / SPLIT_MARKER)
            objects = sorted(
                {
                    row["object_uid"]
                    for name in ("train", "dev")
                    for row in load_json(marker["manifests"][name]["path"])["samples"]
                }
            )
            target_root = root / "targets"
            target_root.mkdir()
            preflight = load_json(split_root / TARGET_PREFLIGHT)
            config = {
                "format": LOCAL_LH_SLAT_VERSION,
                "target_contract": NATIVE_OBJAVERSE_TARGET_CONTRACT,
                "native_primary_target_policy": NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
                "min_native_sequence_target_iou": 0.75,
                "render_manifests": preflight["render_manifests"],
                "lifting_manifests": [
                    {
                        "path": marker["manifests"][name]["path"],
                        "sha256": marker["manifests"][name]["sha256"],
                    }
                    for name in ("train", "dev")
                ],
            }
            write_json(
                target_root / "run_config.json",
                {"config": config, "config_hash": canonical_json_sha256(config)},
            )
            midpoint = len(objects) // 2
            for rank, assigned in enumerate((objects[:midpoint], objects[midpoint:])):
                records = []
                for object_uid in assigned:
                    output = target_root / f"{object_uid}.npz"
                    output.write_bytes(object_uid.encode())
                    records.append(
                        {
                            "object_uid": object_uid,
                            "output": str(output),
                            "output_sha256": sha256_file(output),
                            "source_latents": [
                                {
                                    "target_contract": NATIVE_OBJAVERSE_TARGET_CONTRACT,
                                    "is_primary_target": True,
                                }
                            ],
                            "target_selection": {
                                "policy": NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
                                "minimum_sequence_target_iou": 0.75,
                                "observed_min_target_iou": 7 / 9,
                            },
                        }
                    )
                write_json(
                    target_root / f"rank_{rank:03d}_report.json",
                    {
                        "format": LOCAL_LH_SLAT_VERSION,
                        "config_hash": canonical_json_sha256(config),
                        "rank": rank,
                        "world_size": 2,
                        "passed": True,
                        "records": records,
                        "failures": [],
                    },
                )
            command_finalize_targets(
                argparse.Namespace(
                    split_bundle=str(split_root),
                    target_root=str(target_root),
                    world_size=2,
                )
            )
            target_marker = load_json(target_root / TARGET_MARKER)
            self.assertTrue(target_marker["passed"])
            self.assertEqual(target_marker["object_count"], 8)

    def test_merge_cache_reorders_worker_rows_to_frozen_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = self._prepare(root)
            split_manifest_path = split_root / "train" / "lifting_manifest.json"
            split = load_json(split_manifest_path)
            seeds = [42]
            input_dirs = []
            normalization = {"mean": [0.0] * 8, "std": [1.0] * 8}
            for worker in split["objaverse2k_split"]["workers"]:
                input_dir = root / f"cache_worker_{worker['worker_index']}"
                input_dirs.append(input_dir)
                assigned_rows = [split["samples"][index] for index in worker["indices"]]
                objects = sorted({row["object_uid"] for row in assigned_rows})
                config = {
                    "pretrained": "test",
                    "source_lifting_manifest": str(split_manifest_path),
                    "source_lifting_manifest_sha256": sha256_file(split_manifest_path),
                    "slat_root": str(root / "targets"),
                    "ss_seeds": seeds,
                }
                rows = [
                    {
                        "uid": row["uid"],
                        "object_uid": row["object_uid"],
                        "support_seed": 42,
                        "target_file": f"targets/{row['object_uid']}.npz",
                        "support_file": f"support/{row['uid']}.pt",
                        "physical_file": f"physical/{row['uid']}.pt",
                        "condition_file": f"condition/{row['uid']}.pt",
                        "source_lh_slat": str(root / "targets" / f"{row['object_uid']}.npz"),
                        "source_glb": str(root / "mesh.glb"),
                        "ss_latent": row["ss_latent"],
                    }
                    for row in reversed(assigned_rows)
                ]
                object_rows = [
                    {
                        "object_uid": object_uid,
                        "target_file": f"targets/{object_uid}.npz",
                        "source_lh_slat": str(root / "targets" / f"{object_uid}.npz"),
                        "source_glb": str(root / "mesh.glb"),
                        "ss_latent": next(
                            row["ss_latent"]
                            for row in assigned_rows
                            if row["object_uid"] == object_uid
                        ),
                    }
                    for object_uid in objects
                ]
                write_json(
                    input_dir / "manifest.json",
                    {
                        "format": DIRECT_SLAT_CACHE_VERSION,
                        "materialized": True,
                        "output_dir": str(input_dir),
                        "config": config,
                        "config_hash": canonical_json_sha256(config),
                        "slat_normalization": normalization,
                        "slat_normalization_hash": canonical_json_sha256(normalization),
                        "frozen_ss": {"checkpoint_sha256": "ss"},
                        "samples": rows,
                        "objects": object_rows,
                    },
                )
            merged = root / "merged"
            command_merge_cache(
                argparse.Namespace(
                    split_bundle=str(split_root),
                    split="train",
                    input_dirs=",".join(str(path) for path in input_dirs),
                    output_dir=str(merged),
                    ss_seeds="42",
                )
            )
            manifest = load_json(merged / "manifest.json")
            expected = [row["uid"] for row in split["samples"]]
            self.assertEqual([row["uid"] for row in manifest["samples"]], expected)
            self.assertTrue(all(Path(row["target_file"]).is_absolute() for row in manifest["samples"]))
            self.assertTrue(load_json(merged / CACHE_MARKER)["training_ready"])


if __name__ == "__main__":
    unittest.main()
