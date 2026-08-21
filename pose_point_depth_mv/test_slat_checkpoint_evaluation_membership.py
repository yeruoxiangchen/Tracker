#!/usr/bin/env python3
"""CPU-only regression tests for registered cross-protocol evaluation."""

from __future__ import annotations

import unittest

from pose_point_depth_mv import (
    aggregate_proobjaverse_official_ss_slat_vs_reconviagen as strict_aggregate,
)
from pose_point_depth_mv import (
    evaluate_proobjaverse_official_native_ss_stock_slat as predicted_eval,
)
from pose_point_depth_mv.slat_checkpoint_evaluation_membership import (
    audit_checkpoint_evaluation_membership,
)


def checkpoint(training_uids: list[str], protocol: str = "checkpoint-protocol"):
    return {
        "data_identity": {
            "object_uids": training_uids,
            "target_decoder_audit": {"protocol_sha256": protocol},
        }
    }


class MembershipAuditTest(unittest.TestCase):
    def test_cross_protocol_all_training_passes(self) -> None:
        report = audit_checkpoint_evaluation_membership(
            checkpoint(["a", "b", "c"]),
            evaluation_protocol_sha256="legacy-protocol",
            evaluation_object_uids=["b", "c"],
            expected_membership="all_training",
        )
        self.assertEqual(report["protocol_relation"], "different")
        self.assertEqual(report["training_overlap_count"], 2)
        self.assertTrue(report["all_evaluation_objects_in_checkpoint_training"])

    def test_all_training_rejects_missing_uid(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not entirely contained"):
            audit_checkpoint_evaluation_membership(
                checkpoint(["a", "b"]),
                evaluation_protocol_sha256="legacy-protocol",
                evaluation_object_uids=["b", "c"],
                expected_membership="all_training",
            )

    def test_all_disjoint_is_enforced(self) -> None:
        report = audit_checkpoint_evaluation_membership(
            checkpoint(["a", "b"]),
            evaluation_protocol_sha256="other",
            evaluation_object_uids=["c", "d"],
            expected_membership="all_disjoint",
        )
        self.assertTrue(
            report["all_evaluation_objects_disjoint_from_checkpoint_training"]
        )
        with self.assertRaisesRegex(RuntimeError, "not disjoint"):
            audit_checkpoint_evaluation_membership(
                checkpoint(["a", "b"]),
                evaluation_protocol_sha256="other",
                evaluation_object_uids=["b", "c"],
                expected_membership="all_disjoint",
            )

    def test_duplicate_evaluation_uids_fail(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            audit_checkpoint_evaluation_membership(
                checkpoint(["a", "b"]),
                evaluation_protocol_sha256="other",
                evaluation_object_uids=["a", "a"],
            )


class ParserCompatibilityTest(unittest.TestCase):
    def test_predicted_worker_default_keeps_cross_protocol_disabled(self) -> None:
        parser = predicted_eval.make_parser()
        args = parser.parse_args(
            [
                "worker",
                "--cache_manifest",
                "cache.json",
                "--lifting_cache_manifest",
                "lifting.json",
                "--native_ss_report",
                "ss.json",
                "--stock_slat_freeze",
                "freeze.json",
                "--output_dir",
                "out",
            ]
        )
        self.assertFalse(args.allow_trained_slat_target_protocol_mismatch)
        self.assertEqual(args.expected_checkpoint_training_membership, "any")

    def test_strict_aggregate_default_remains_heldout(self) -> None:
        parser = strict_aggregate.make_parser()
        args = parser.parse_args(
            [
                "--dev_split",
                "dev.json",
                "--cache_report",
                "cache.json",
                "--target_report",
                "target.json",
                "--target_mesh_root",
                "targets",
                "--recon_reports",
                "r.json",
                "--current_reports",
                "c.json",
                "--expected_current_sha256",
                "0" * 64,
                "--output_dir",
                "out",
            ]
        )
        self.assertEqual(args.evaluation_membership_scope, "heldout")


if __name__ == "__main__":
    unittest.main()
