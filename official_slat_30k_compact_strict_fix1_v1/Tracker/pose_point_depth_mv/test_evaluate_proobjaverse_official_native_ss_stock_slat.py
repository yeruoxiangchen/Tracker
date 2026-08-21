#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    _aggregate_occupancy,
    _recordable_mesh_decode_error,
    _select_transfer_branches,
    _ss_record,
    _validate_ss_evidence_domain,
    pair_id,
)
from pose_point_depth_mv.proobjaverse_official_ss import official_domain_contract


class OfficialNativeSSStockSLatTests(unittest.TestCase):
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
        oom = RuntimeError("CUDA out of memory")
        self.assertTrue(_recordable_mesh_decode_error(topology))
        self.assertTrue(_recordable_mesh_decode_error(invalid_mesh))
        self.assertTrue(_recordable_mesh_decode_error(oversized))
        self.assertFalse(_recordable_mesh_decode_error(oom))
        self.assertFalse(_recordable_mesh_decode_error(ValueError("bad input")))

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
