#!/usr/bin/env python3

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    _aggregate_occupancy,
    _load_frozen_target_mesh_bindings,
    _load_frozen_stock_floor_records,
    _load_cuda_branch_failure_marker,
    _materialize_frozen_target_mesh,
    _cuda_context_poisoning_mesh_decode_error,
    _recordable_mesh_decode_error,
    _select_transfer_branches,
    _ss_record,
    _validate_worker_shard_local_bindings,
    _validate_ss_evidence_domain,
    _worker_global_run_identity,
    pair_id,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import mesh_structure_metrics
from pose_point_depth_mv.native_ss_genrecon import sha256_file
from pose_point_depth_mv.proobjaverse_official_ss import official_domain_contract
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256
import trimesh


class OfficialNativeSSStockSLatTests(unittest.TestCase):
    def test_shard_local_target_and_floor_bindings_are_validated_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_report = root / "floor.json"
            source_report.write_text("{}\n", encoding="utf-8")
            uid = "object-a"
            seeds = [42, 43, 44]
            branches = ["stock", "native", "native_trained"]
            structure = {"mesh_success": True, "vertex_count": 4, "face_count": 4}
            records = [
                {
                    "pair_id": pair_id(uid, seed),
                    "branch": branch,
                    "object_uid": uid,
                    "seed": seed,
                    "target_structure": structure,
                    "target_mesh_sha256": "1" * 64,
                    "target_mesh_policy": (
                        "exact_npz_from_frozen_strict_reconviagen_reports"
                    ),
                    "passed": True,
                }
                for seed in seeds
                for branch in branches
            ]
            target_binding = canonical_sha256(
                [
                    {
                        "object_uid": uid,
                        "target_mesh_sha256": "1" * 64,
                        "target_structure": structure,
                    }
                ]
            )
            by_key = {
                (row["object_uid"], row["seed"], row["branch"]): row
                for row in records
            }
            floor_rows = [
                {
                    "object_uid": uid,
                    "seed": seed,
                    "stock": by_key[(uid, seed, "stock")],
                    "native": by_key[(uid, seed, "native")],
                }
                for seed in seeds
            ]
            identity = {
                "object_start": 16,
                "object_end": 17,
                "object_uids": [uid],
                "joint_seeds": seeds,
                "paired_branches": branches,
                "target_mesh_policy": (
                    "exact_npz_from_frozen_strict_reconviagen_reports"
                ),
                "frozen_target_binding_sha256": target_binding,
                "frozen_stock_floor_reuse": {
                    "policy": "reuse_targetlocked_stock_and_native_rows",
                    "source_report": str(source_report),
                    "source_report_sha256": sha256_file(source_report),
                    "source_checkpoint": "/checkpoint/step_030000.pt",
                    "source_checkpoint_sha256": "2" * 64,
                    "source_step": 30000,
                    "row_binding_sha256": canonical_sha256(floor_rows),
                    "pair_count": 3,
                },
                "global_field": "same-across-shards",
            }
            payload = {"run_identity": identity, "mesh_branch_records": records}
            binding = _validate_worker_shard_local_bindings(
                payload,
                report_path=root / "worker.json",
                seeds=seeds,
            )
            self.assertEqual(
                binding["frozen_target_binding_sha256"], target_binding
            )
            self.assertEqual(
                binding["frozen_stock_floor_global_identity"]["source_step"],
                30000,
            )

            other_shard = deepcopy(identity)
            other_shard.update(
                {
                    "object_start": 17,
                    "object_end": 18,
                    "object_uids": ["object-b"],
                    "frozen_target_binding_sha256": "3" * 64,
                    "frozen_stock_floor_reuse": {"different": "local"},
                }
            )
            self.assertEqual(
                _worker_global_run_identity(identity),
                _worker_global_run_identity(other_shard),
            )

            tampered = deepcopy(payload)
            tampered["run_identity"]["frozen_target_binding_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "target binding differs"):
                _validate_worker_shard_local_bindings(
                    tampered,
                    report_path=root / "worker.json",
                    seeds=seeds,
                )

    def test_targetlocked_stock_floor_rows_are_checkpoint_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uid = "object-a"
            seeds = [42, 43, 44]
            structure = {"mesh_success": True, "vertices": 4, "faces": 4}
            target_sha = "1" * 64
            identity = {
                "cache_manifest_sha256": "a" * 64,
                "lifting_cache_manifest_sha256": "b" * 64,
                "native_ss_report_sha256": "c" * 64,
                "stock_slat_freeze_sha256": "d" * 64,
                "trained_slat_checkpoint": "/checkpoint/step_010000.pt",
                "trained_slat_checkpoint_sha256": "e" * 64,
                "expected_trained_slat_step": 10000,
                "object_uids": [uid],
                "joint_seeds": seeds,
                "weights": "ema",
                "amp_dtype": "bf16",
                "surface_samples": 20000,
                "target_mesh_policy": (
                    "exact_npz_from_frozen_strict_reconviagen_reports"
                ),
                "frozen_target_binding_sha256": "f" * 64,
                "paired_branches": ["stock", "native", "native_trained"],
            }
            records = []
            for seed in seeds:
                for branch in ("stock", "native", "native_trained"):
                    records.append(
                        {
                            "pair_id": pair_id(uid, seed),
                            "branch": branch,
                            "object_uid": uid,
                            "seed": seed,
                            "target_structure": structure,
                            "target_mesh_sha256": target_sha,
                            "target_mesh_policy": identity["target_mesh_policy"],
                            "flow_summary": {"adapted": branch == "native_trained"},
                            "passed": True,
                        }
                    )
            report = {
                "format": (
                    "pose_point_depth_mv.proobjaverse_official_native_ss_slat_"
                    "end_to_end_worker.v1"
                ),
                "complete": True,
                "passed": True,
                "run_identity": identity,
                "object_count": 1,
                "record_count": 3,
                "ss_records": [
                    {"object_uid": uid, "seed": seed, "passed": True}
                    for seed in seeds
                ],
                "mesh_branch_records": records,
            }
            report["report_sha256"] = canonical_sha256(report)
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            rows, binding = _load_frozen_stock_floor_records(
                str(path),
                selected_uids=[uid],
                seeds=seeds,
                cache_manifest_sha256="a" * 64,
                lifting_cache_manifest_sha256="b" * 64,
                native_ss_report_sha256="c" * 64,
                stock_slat_freeze_sha256="d" * 64,
                target_policy=identity["target_mesh_policy"],
                target_binding_sha256="f" * 64,
                frozen_targets={
                    uid: {"sha256": target_sha, "structure": structure}
                },
                amp_dtype="bf16",
                surface_samples=20000,
            )
            self.assertEqual(set(rows[(uid, 42)]), {"stock", "native"})
            self.assertEqual(binding["source_step"], 10000)
            self.assertEqual(binding["pair_count"], 3)

            # A completed worker remains reusable when a model output failure
            # is explicitly recorded.  ``passed=false`` must not be confused
            # with an interrupted/incomplete worker.
            recorded_failure = deepcopy(report)
            recorded_failure.pop("report_sha256")
            stock_seed42 = next(
                row
                for row in recorded_failure["mesh_branch_records"]
                if row["branch"] == "stock" and row["seed"] == 42
            )
            stock_seed42.pop("flow_summary")
            stock_seed42["passed"] = False
            stock_seed42["slat_active_point_count"] = 100_000
            stock_seed42["slat_active_point_limit"] = 80_000
            stock_seed42["error"] = {
                "type": "RuntimeError",
                "message": (
                    "SLat decoder input exceeds safe active-point limit: "
                    "points=100000 limit=80000"
                ),
                "stage": "stock_slat_mesh_decode",
            }
            recorded_failure["passed"] = False
            recorded_failure["report_sha256"] = canonical_sha256(recorded_failure)
            failure_path = root / "recorded_failure_report.json"
            failure_path.write_text(json.dumps(recorded_failure), encoding="utf-8")
            failed_rows, _ = _load_frozen_stock_floor_records(
                str(failure_path),
                selected_uids=[uid],
                seeds=seeds,
                cache_manifest_sha256="a" * 64,
                lifting_cache_manifest_sha256="b" * 64,
                native_ss_report_sha256="c" * 64,
                stock_slat_freeze_sha256="d" * 64,
                target_policy=identity["target_mesh_policy"],
                target_binding_sha256="f" * 64,
                frozen_targets={
                    uid: {"sha256": target_sha, "structure": structure}
                },
                amp_dtype="bf16",
                surface_samples=20000,
            )
            self.assertIs(failed_rows[(uid, 42)]["stock"]["passed"], False)

            unregistered_failure = deepcopy(recorded_failure)
            unregistered_failure.pop("report_sha256")
            next(
                row
                for row in unregistered_failure["mesh_branch_records"]
                if row["branch"] == "stock" and row["seed"] == 42
            )["error"]["message"] = "CUDA out of memory"
            unregistered_failure["report_sha256"] = canonical_sha256(
                unregistered_failure
            )
            invalid_path = root / "unregistered_failure_report.json"
            invalid_path.write_text(
                json.dumps(unregistered_failure), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "unregistered failure"):
                _load_frozen_stock_floor_records(
                    str(invalid_path),
                    selected_uids=[uid],
                    seeds=seeds,
                    cache_manifest_sha256="a" * 64,
                    lifting_cache_manifest_sha256="b" * 64,
                    native_ss_report_sha256="c" * 64,
                    stock_slat_freeze_sha256="d" * 64,
                    target_policy=identity["target_mesh_policy"],
                    target_binding_sha256="f" * 64,
                    frozen_targets={
                        uid: {"sha256": target_sha, "structure": structure}
                    },
                    amp_dtype="bf16",
                    surface_samples=20000,
                )

            with self.assertRaisesRegex(RuntimeError, "branch identity differs"):
                _load_frozen_stock_floor_records(
                    str(path),
                    selected_uids=[uid],
                    seeds=seeds,
                    cache_manifest_sha256="a" * 64,
                    lifting_cache_manifest_sha256="b" * 64,
                    native_ss_report_sha256="c" * 64,
                    stock_slat_freeze_sha256="d" * 64,
                    target_policy=identity["target_mesh_policy"],
                    target_binding_sha256="f" * 64,
                    frozen_targets={
                        uid: {"sha256": "0" * 64, "structure": structure}
                    },
                    amp_dtype="bf16",
                    surface_samples=20000,
                )

    def test_frozen_strict_target_is_content_addressed_and_materialized_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            vertices = np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float64,
            )
            faces = np.asarray(
                [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                dtype=np.int64,
            )
            np.savez_compressed(source, vertices=vertices, faces=faces)
            structure = mesh_structure_metrics(
                trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            )
            source_sha = sha256_file(source)
            uid = "object-a"
            report = {
                "format": (
                    "pose_point_depth_mv.proobjaverse_official_reconviagen_worker.v1"
                ),
                "complete": True,
                "seeds": [42, 43, 44],
                "official_protocol_sha256": "a" * 64,
                "records": [
                    {
                        "object_uid": uid,
                        "seed": seed,
                        "target_mesh": str(source),
                        "target_mesh_sha256": source_sha,
                        "target_structure": structure,
                    }
                    for seed in (42, 43, 44)
                ],
            }
            report["report_sha256"] = canonical_sha256(report)
            report_path = root / "strict.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            bindings, identity = _load_frozen_target_mesh_bindings(
                str(report_path),
                selected_uids=[uid],
                seeds=[42, 43, 44],
                protocol_sha256="a" * 64,
            )
            target = root / "worker" / "target_mesh_cache" / f"{uid}.npz"
            mesh, target_sha = _materialize_frozen_target_mesh(bindings[uid], target)
            self.assertEqual(identity["policy"], "exact_npz_from_frozen_strict_reconviagen_reports")
            self.assertEqual(target_sha, source_sha)
            self.assertEqual(sha256_file(target), source_sha)
            self.assertEqual(mesh_structure_metrics(mesh), structure)

    def test_pair_id_is_stable_and_seed_specific(self) -> None:
        self.assertEqual(pair_id("object-a", 42), pair_id("object-a", 42))
        self.assertNotEqual(pair_id("object-a", 42), pair_id("object-a", 43))

    def test_ss_record_uses_official_coordinates_only_as_target(self) -> None:
        target = np.asarray([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32)
        stock = np.asarray([[0, 1, 1, 1]], dtype=np.int32)
        native = target.copy()
        row = _ss_record(
            object_uid="object-a",
            seed=42,
            stock_coords=stock,
            native_coords=native,
            target_coords=target,
            wrapper={"enabled": True},
        )
        self.assertAlmostEqual(row["stock"]["iou"], 0.5)
        self.assertAlmostEqual(row["native"]["iou"], 1.0)
        self.assertAlmostEqual(row["iou_gain"], 0.5)
        self.assertTrue(row["same_initial_noise"])

    def test_positive_occupancy_records_pass_development_gate(self) -> None:
        records = []
        for index in range(8):
            records.append(
                {
                    "object_uid": f"object-{index}",
                    "seed": 42,
                    "passed": True,
                    "iou_gain": 0.1,
                    "precision_gain": 0.05,
                    "recall_gain": 0.08,
                    "native_stock_count_ratio": 1.0,
                }
            )
        result = _aggregate_occupancy(records, bootstrap_samples=200)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_only_model_mesh_failures_are_recordable(self) -> None:
        topology = RuntimeError(
            "FlexiCubes topology index is inconsistent: context=surface_edge "
            "invalid=1 total=8 size=7 min=-1 max=6"
        )
        invalid_mesh = RuntimeError("decoded native Mesh is invalid: object-a")
        oversized = RuntimeError(
            "SLat decoder input exceeds safe active-point limit: "
            "points=131384 limit=80000"
        )
        cuda_topology = RuntimeError(
            "SLat decoder CUDA topology failure: branch=stock uid=object-a "
            "seed=42: merge_sort failed with cudaErrorIllegalAddress"
        )
        raw_cuda_fault = RuntimeError(
            "merge_sort failed with cudaErrorIllegalAddress"
        )
        oom = RuntimeError("CUDA out of memory")
        self.assertTrue(_recordable_mesh_decode_error(topology))
        self.assertTrue(_recordable_mesh_decode_error(invalid_mesh))
        self.assertTrue(_recordable_mesh_decode_error(oversized))
        self.assertTrue(_recordable_mesh_decode_error(cuda_topology))
        self.assertFalse(_recordable_mesh_decode_error(raw_cuda_fault))
        self.assertFalse(_recordable_mesh_decode_error(oom))
        self.assertFalse(_recordable_mesh_decode_error(ValueError("bad input")))

    def test_only_decoder_cuda_context_poisoning_faults_request_restart(self) -> None:
        self.assertTrue(
            _cuda_context_poisoning_mesh_decode_error(
                RuntimeError(
                    "merge_sort: failed to synchronize: "
                    "cudaErrorIllegalAddress: an illegal memory access was encountered"
                )
            )
        )
        self.assertTrue(
            _cuda_context_poisoning_mesh_decode_error(
                RuntimeError("CUDA error: device-side assert triggered")
            )
        )
        self.assertFalse(
            _cuda_context_poisoning_mesh_decode_error(
                RuntimeError("CUDA out of memory")
            )
        )

    def test_cuda_branch_failure_marker_is_identity_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stock.json"
            row = {
                "pair_id": "pair-a",
                "branch": "stock",
                "object_uid": "object-a",
                "seed": 42,
                "passed": False,
                "error": {
                    "type": "RuntimeError",
                    "stage": "stock_slat_mesh_decode",
                    "message": (
                        "SLat decoder CUDA topology failure: branch=stock "
                        "uid=object-a seed=42: cudaErrorIllegalAddress"
                    ),
                },
            }
            path.write_text(
                json.dumps(
                    {
                        "format": (
                            "pose_point_depth_mv.proobjaverse_official_slat_"
                            "cuda_branch_failure.v1"
                        ),
                        "pair_id": "pair-a",
                        "object_uid": "object-a",
                        "seed": 42,
                        "branch": "stock",
                        "row": row,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _load_cuda_branch_failure_marker(
                    path,
                    current_pair_id="pair-a",
                    object_uid="object-a",
                    seed=42,
                    branch="stock",
                ),
                row,
            )
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                _load_cuda_branch_failure_marker(
                    path,
                    current_pair_id="pair-a",
                    object_uid="object-b",
                    seed=42,
                    branch="stock",
                )

    def test_named_end_to_end_branches_project_without_semantic_loss(self) -> None:
        records = [
            {"branch": "stock", "object_uid": "a", "seed": 42, "passed": True},
            {"branch": "native", "object_uid": "a", "seed": 42, "passed": True},
            {
                "branch": "native_trained",
                "object_uid": "a",
                "seed": 42,
                "passed": True,
            },
        ]
        projected = _select_transfer_branches(
            records, baseline="stock", candidate="native_trained"
        )
        self.assertEqual([row["branch"] for row in projected], ["stock", "native"])
        self.assertEqual(
            [row["source_branch"] for row in projected],
            ["stock", "native_trained"],
        )
        self.assertEqual(records[2]["branch"], "native_trained")

    def test_native_ss_evidence_must_bind_the_same_official_protocol(self) -> None:
        payload = {
            "official_ss_domain_contract": official_domain_contract(
                protocol_sha256="a" * 64,
                encoder_pretrained="encoder",
                decoder_pretrained="Stable-X/trellis-vggt-v0-2",
                latent_dtype="float16",
                minimum_roundtrip_iou=0.9,
            )
        }
        _validate_ss_evidence_domain(
            payload,
            target_contract={"protocol_sha256": "a" * 64},
            pretrained="Stable-X/trellis-vggt-v0-2",
        )
        with self.assertRaisesRegex(RuntimeError, "official protocols differ"):
            _validate_ss_evidence_domain(
                payload,
                target_contract={"protocol_sha256": "b" * 64},
                pretrained="Stable-X/trellis-vggt-v0-2",
            )


if __name__ == "__main__":
    unittest.main()
