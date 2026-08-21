from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from official_ss_with_vggt_perf_v1.reuse_endpoint_artifacts import prepare_reuse
from official_ss_with_vggt_perf_v1.ss_slat_endpoint import endpoint_contract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReuseEndpointArtifactsTest(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source"
        target = root / "target"
        source.mkdir()
        source_checkpoint = root / "step_010000.pt"
        target_checkpoint = root / "step_015000.pt"
        source_checkpoint.write_bytes(b"checkpoint-10000")
        target_checkpoint.write_bytes(b"checkpoint-15000")
        object_uids = ["object-a", "object-b"]
        seeds = [42, 43]
        identity = {
            "trained_slat_checkpoint": str(source_checkpoint.resolve()),
            "trained_slat_checkpoint_sha256": sha256(source_checkpoint),
            "expected_trained_slat_step": 10000,
            "object_uids": object_uids,
            "joint_seeds": seeds,
        }
        (source / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
        report = {"complete": True, "passed": True, "run_identity": identity}
        (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
        coords = source / "ss_coords"
        coords.mkdir()
        for object_uid in object_uids:
            for seed in seeds:
                stem = f"{object_uid}-{seed}"
                npz = coords / f"{stem}.npz"
                npz.write_bytes(f"coords:{stem}".encode())
                audit = {
                    "object_uid": object_uid,
                    "seed": seed,
                    "coords_npz_sha256": sha256(npz),
                }
                (coords / f"{stem}.json").write_text(json.dumps(audit), encoding="utf-8")
        meshes = source / "target_mesh_cache"
        meshes.mkdir()
        for object_uid in object_uids:
            (meshes / f"{object_uid}.npz").write_bytes(f"mesh:{object_uid}".encode())
        return source, target, source_checkpoint, target_checkpoint

    def test_hardlink_reuse_is_complete_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_checkpoint, target_checkpoint = self._fixture(Path(directory))
            first = prepare_reuse(
                source_worker=source,
                target_worker=target,
                source_step=10000,
                target_step=15000,
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
            )
            second = prepare_reuse(
                source_worker=source,
                target_worker=target,
                source_step=10000,
                target_step=15000,
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
            )
            self.assertTrue(first["passed"] and second["passed"])
            self.assertEqual(first["linked_file_count"], 10)
            source_file = next((source / "ss_coords").glob("*.npz"))
            target_file = target / source_file.relative_to(source)
            self.assertEqual(os.stat(source_file).st_ino, os.stat(target_file).st_ino)

    def test_tampered_coordinate_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_checkpoint, target_checkpoint = self._fixture(Path(directory))
            next((source / "ss_coords").glob("*.npz")).write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                prepare_reuse(
                    source_worker=source,
                    target_worker=target,
                    source_step=10000,
                    target_step=15000,
                    source_checkpoint=source_checkpoint,
                    target_checkpoint=target_checkpoint,
                )

    def test_ss_coords_only_does_not_reuse_target_meshes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_checkpoint, target_checkpoint = self._fixture(
                Path(directory)
            )
            result = prepare_reuse(
                source_worker=source,
                target_worker=target,
                source_step=10000,
                target_step=15000,
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
                reuse_target_meshes=False,
            )
            self.assertEqual(result["reuse_scope"], "ss_coords_only")
            self.assertEqual(result["target_mesh_count"], 0)
            self.assertEqual(result["linked_file_count"], 8)
            self.assertFalse((target / "target_mesh_cache").exists())

    def test_endpoint_annotated_report_binds_to_base_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_checkpoint, target_checkpoint = self._fixture(
                Path(directory)
            )
            ss_cache = Path(directory) / "with_vggt_ss_manifest.json"
            ss_cache.write_text('{"format":"fixture"}\n', encoding="utf-8")
            report_path = source / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            contract = endpoint_contract()
            report["run_identity"].update(
                {
                    "ss_cache_manifest": str(ss_cache.resolve()),
                    "ss_cache_manifest_sha256": sha256(ss_cache),
                    "endpoint_version": contract["version"],
                    "branch_semantics": contract["branches"],
                    "slat_support_input": "predicted_only",
                    "gt_support_used_as_slat_input": False,
                }
            )
            report["with_vggt_endpoint_contract"] = contract
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = prepare_reuse(
                source_worker=source,
                target_worker=target,
                source_step=10000,
                target_step=15000,
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["linked_file_count"], 10)

    def test_unregistered_endpoint_identity_extension_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_checkpoint, target_checkpoint = self._fixture(
                Path(directory)
            )
            report_path = source / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["run_identity"]["unexpected_endpoint_field"] = True
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "base run_identity binding differs"):
                prepare_reuse(
                    source_worker=source,
                    target_worker=target,
                    source_step=10000,
                    target_step=15000,
                    source_checkpoint=source_checkpoint,
                    target_checkpoint=target_checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
