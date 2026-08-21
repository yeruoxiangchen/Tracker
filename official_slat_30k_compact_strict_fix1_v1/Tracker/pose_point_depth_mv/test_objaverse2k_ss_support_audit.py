#!/usr/bin/env python3

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from pose_point_depth_mv.audit_objaverse2k_ss_support import (
    SELECTION_FORMAT,
    SELECTION_MARKER,
    SELECTION_MARKER_FORMAT,
    SHARD_REPORT_FORMAT,
    aggregate_reports,
    atomic_json,
    canonical_json_sha256,
    select_balanced_rows,
    sha256_file,
    validate_selection_bundle,
)


def fake_record(object_uid: str, seed: int, *, control: bool = False) -> dict:
    full_iou = 0.25 if not control else 0.20
    stock_iou = 0.10
    return {
        "uid": f"{object_uid}_seq000",
        "object_uid": object_uid,
        "seed": seed,
        "projection_mode": "pose_cyclic1" if control else "correct",
        "same_initial_noise": True,
        "stock": {
            "iou": stock_iou,
            "precision": 0.15,
            "recall": 0.16,
            "coord_count_ratio": 1.0,
        },
        "full": {
            "iou": full_iou,
            "precision": 0.35 if not control else 0.30,
            "recall": 0.40 if not control else 0.32,
            "coord_count_ratio": 1.10,
        },
        "stock_count": 100,
        "full_count": 110,
        "full_minus_stock_count": 10,
        "full_stock_count_ratio": 1.10,
        "iou_gain": full_iou - stock_iou,
        "precision_gain": (0.35 if not control else 0.30) - 0.15,
        "recall_gain": (0.40 if not control else 0.32) - 0.16,
        "latent_mse_gain": 0.05,
    }


def fake_candidate(records: list[dict]) -> dict:
    return {
        "object_count": len({str(row["object_uid"]) for row in records}),
        "record_count": len(records),
        "records": records,
        "disabled_stock_equivalence": {"passed": True},
    }


class BalancedSelectionTest(unittest.TestCase):
    def test_selects_one_sequence_for_equal_objects_per_shard(self) -> None:
        rows = []
        for shard in range(2):
            for object_index in range(4):
                object_uid = f"s{shard}_o{object_index}"
                for sequence in range(2):
                    rows.append(
                        {
                            "uid": f"{object_uid}_seq{sequence:03d}",
                            "object_uid": object_uid,
                            "source_shard_position": shard,
                        }
                    )
        selected, metadata = select_balanced_rows(
            rows,
            expected_shards=2,
            objects_per_shard=3,
            seed=42,
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["object_uid"] for row in selected}), 6)
        self.assertEqual(
            [sum(row["source_shard_position"] == shard for row in metadata) for shard in range(2)],
            [3, 3],
        )
        repeated, repeated_metadata = select_balanced_rows(
            rows,
            expected_shards=2,
            objects_per_shard=3,
            seed=42,
        )
        self.assertEqual(selected, repeated)
        self.assertEqual(metadata, repeated_metadata)

    def test_rejects_missing_source_shard(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source shards differ"):
            select_balanced_rows(
                [
                    {
                        "uid": "a_seq000",
                        "object_uid": "a",
                        "source_shard_position": 0,
                    }
                ],
                expected_shards=2,
                objects_per_shard=1,
                seed=42,
            )


class SelectionBundleTest(unittest.TestCase):
    def test_accepts_complete_bundle_and_rejects_marker_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "uid": f"obj_{index:03d}_seq000",
                    "object_uid": f"obj_{index:03d}",
                    "source_shard_position": index // 8,
                }
                for index in range(64)
            ]
            selection = {
                "format": SELECTION_FORMAT,
                "formal": False,
                "future_slat_training_overlap": True,
                "training_object_disjoint_from_frozen_ss": True,
                "object_count": 64,
                "object_uid_hash": canonical_json_sha256(
                    sorted(str(row["object_uid"]) for row in rows)
                ),
                "rows": [dict(row) for row in rows],
            }
            selection_path = root / "selection.json"
            atomic_json(selection_path, selection)
            manifest = {
                "format": LIFTING_CACHE_VERSION,
                "passed": True,
                "training_ready": True,
                "sample_count": 64,
                "object_count": 64,
                "source_cache_manifest": str(selection_path),
                "audit_selection": {
                    "format": SELECTION_FORMAT,
                    "selection_sha256": sha256_file(selection_path),
                },
                "samples": rows,
            }
            manifest_path = root / "lifting_manifest.json"
            atomic_json(manifest_path, manifest)
            marker = {
                "format": SELECTION_MARKER_FORMAT,
                "passed": True,
                "manifest_sha256": sha256_file(manifest_path),
                "selection_sha256": sha256_file(selection_path),
                "checkpoint_sha256": "checkpoint",
                "object_count": 64,
            }
            marker_path = root / SELECTION_MARKER
            atomic_json(marker_path, marker)

            validated, validated_selection = validate_selection_bundle(
                manifest_path,
                checkpoint_sha256="checkpoint",
                expected_objects=64,
            )
            self.assertEqual(validated["object_count"], 64)
            self.assertEqual(validated_selection["object_count"], 64)

            marker["checkpoint_sha256"] = "changed"
            atomic_json(marker_path, marker)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                validate_selection_bundle(
                    manifest_path,
                    checkpoint_sha256="checkpoint",
                    expected_objects=64,
                )


class AggregateReportsTest(unittest.TestCase):
    def make_inputs(self) -> tuple[list[dict], dict, dict]:
        object_uids = [f"obj_{index:03d}" for index in range(64)]
        selection = {"samples": [{"object_uid": uid} for uid in object_uids]}
        invariant = {
            "num_workers": 4,
            "cache_manifest_sha256": "cache",
            "checkpoint_sha256": "checkpoint",
            "reference_report_sha256": "reference",
            "frozen_deployment": {
                "cfg_strength": 3.0,
                "joint_seeds": [42, 43, 44],
            },
            "cache_compatibility": {"mode": "runtime_semantics_equal_source_config_distinct"},
        }
        reports = []
        for worker in range(4):
            worker_objects = object_uids[worker * 16 : (worker + 1) * 16]
            correct = [
                fake_record(uid, seed)
                for uid in worker_objects
                for seed in (42, 43, 44)
            ]
            control = [
                fake_record(uid, seed, control=True)
                for uid in worker_objects
                for seed in (42, 43, 44)
            ]
            reports.append(
                {
                    "format": SHARD_REPORT_FORMAT,
                    "complete": True,
                    "worker_index": worker,
                    **invariant,
                    "object_start": worker * 16,
                    "object_end": (worker + 1) * 16,
                    "object_uids": worker_objects,
                    "integrity_checks": {"complete": True},
                    "correct": fake_candidate(correct),
                    "pose_cyclic_control": fake_candidate(control),
                }
            )
        reference = {
            "correct": {
                "records": [
                    fake_record(uid, seed)
                    for uid in object_uids[:32]
                    for seed in (42, 43, 44)
                ]
            }
        }
        return reports, selection, reference

    def test_reaggregates_all_object_seed_records(self) -> None:
        reports, selection, reference = self.make_inputs()
        result = aggregate_reports(
            reports,
            selection_manifest=selection,
            reference_report=reference,
            bootstrap_samples=100,
            min_absolute_retention=0.8,
            min_count_ratio=0.85,
            max_count_ratio=1.25,
            min_iou_win_rate=0.5,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["object_count"], 64)
        self.assertEqual(result["record_count_per_branch"], 192)
        self.assertAlmostEqual(result["correct"]["summary"]["iou_gain"]["mean"], 0.15)
        self.assertAlmostEqual(result["correct_over_pose_control_iou"]["mean"], 0.05)

    def test_rejects_worker_object_overlap(self) -> None:
        reports, selection, reference = self.make_inputs()
        reports = copy.deepcopy(reports)
        reports[1]["object_uids"][0] = reports[0]["object_uids"][0]
        with self.assertRaisesRegex(RuntimeError, "frozen selection slice"):
            aggregate_reports(
                reports,
                selection_manifest=selection,
                reference_report=reference,
                bootstrap_samples=100,
                min_absolute_retention=0.8,
                min_count_ratio=0.85,
                max_count_ratio=1.25,
                min_iou_win_rate=0.5,
            )

    def test_rejects_worker_outside_frozen_slice(self) -> None:
        reports, selection, reference = self.make_inputs()
        reports = copy.deepcopy(reports)
        reports[0]["object_uids"][0], reports[1]["object_uids"][0] = (
            reports[1]["object_uids"][0],
            reports[0]["object_uids"][0],
        )
        with self.assertRaisesRegex(RuntimeError, "frozen selection slice"):
            aggregate_reports(
                reports,
                selection_manifest=selection,
                reference_report=reference,
                bootstrap_samples=100,
                min_absolute_retention=0.8,
                min_count_ratio=0.85,
                max_count_ratio=1.25,
                min_iou_win_rate=0.5,
            )

    def test_rejects_duplicate_seed_record(self) -> None:
        reports, selection, reference = self.make_inputs()
        reports = copy.deepcopy(reports)
        reports[0]["correct"]["records"][1]["seed"] = 42
        with self.assertRaisesRegex(RuntimeError, "record identities differ"):
            aggregate_reports(
                reports,
                selection_manifest=selection,
                reference_report=reference,
                bootstrap_samples=100,
                min_absolute_retention=0.8,
                min_count_ratio=0.85,
                max_count_ratio=1.25,
                min_iou_win_rate=0.5,
            )


if __name__ == "__main__":
    unittest.main()
