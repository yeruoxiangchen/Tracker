from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ar_ss_flow.local_pose_lifting_flow import resolve_ss_latent_path
from pose_point_depth_mv.aggregate_proobjaverse_official_native_ss_eval import (
    validate_record_matrix,
)
from pose_point_depth_mv.materialize_proobjaverse_official_ss_targets import (
    build_rebound_lifting_manifest,
    join_source_rows,
)
from pose_point_depth_mv.native_ss_genrecon import canonical_json_sha256, sha256_file
from pose_point_depth_mv.proobjaverse_official_ss import (
    OFFICIAL_SS_EVAL_AGGREGATE,
    load_official_native_ss_deployment,
    official_domain_contract,
)


class ProObjaverseOfficialSSTests(unittest.TestCase):
    def _deployment_report(self, root: Path) -> Path:
        checkpoint = root / "step_002000.pt"
        checkpoint.write_bytes(b"checkpoint")
        report = {
            "format": OFFICIAL_SS_EVAL_AGGREGATE,
            "passed": True,
            "formal": False,
            "object_count": 1,
            "object_uids": ["dev-object"],
            "object_uid_hash": canonical_json_sha256(["dev-object"]),
            "checks": {"iou": True, "recall": True},
            "official_ss_domain_contract": official_domain_contract(
                protocol_sha256="a" * 64,
                encoder_pretrained="encoder",
                decoder_pretrained="Stable-X/trellis-vggt-v0-2",
                latent_dtype="float16",
                minimum_roundtrip_iou=0.9,
            ),
            "deployment": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_step": 2000,
                "weights": "ema",
                "cfg_strength": 3.0,
                "steps": 25,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
                "amp_dtype": "bf16",
            },
        }
        report["report_sha256"] = canonical_json_sha256(report)
        path = root / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return path

    def test_manifest_target_overrides_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = resolve_ss_latent_path(
                {"uid": "x", "ss_latent": "targets/x.npz"},
                {"uid": "x", "ss_latent": "/old/unused_placeholder.npz"},
                root,
            )
            self.assertEqual(observed, (root / "targets/x.npz").resolve())

    def test_join_requires_exact_uid_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "UID sets differ"):
            join_source_rows(
                {"samples": [{"uid": "a"}]},
                {"samples": [{"uid": "a"}, {"uid": "b"}]},
            )

    def test_rebound_manifest_freezes_domain_and_disables_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lifting_path = root / "lifting.json"
            slat_path = root / "slat.json"
            lifting_path.write_text("{}\n", encoding="utf-8")
            slat_path.write_text("{}\n", encoding="utf-8")
            domain = official_domain_contract(
                protocol_sha256="a" * 64,
                encoder_pretrained="encoder",
                decoder_pretrained="decoder",
                latent_dtype="float16",
                minimum_roundtrip_iou=0.9,
            )
            source = {
                "format": "ar_ss_flow.pose_lifting_cache.v1",
                "config": {"existing": True},
                "samples": [{"uid": "x"}],
            }
            result = build_rebound_lifting_manifest(
                source=source,
                source_manifest=lifting_path,
                source_slat_manifest=slat_path,
                output_dir=root / "out",
                rows=[{"uid": "x", "ss_latent": "/new/x.npz"}],
                domain_contract=domain,
                split="train",
            )
            binding = result["config"]["official_ss_targets"]
            self.assertFalse(result["official_gt_support_only"])
            self.assertFalse(binding["placeholder_targets_consumed"])
            self.assertTrue(binding["row_level_target_override"])
            self.assertEqual(binding["domain_contract"], domain)
            self.assertTrue(result["config_hash"])
            json.dumps(result)

    def test_passed_aggregate_loads_as_strict_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._deployment_report(Path(directory))
            payload, binding = load_official_native_ss_deployment(path)
            self.assertTrue(payload["passed"])
            self.assertEqual(binding["checkpoint_step"], 2000)
            self.assertEqual(binding["cfg_strength"], 3.0)
            self.assertEqual(binding["false_checks"], [])

    def test_changed_deployment_semantics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._deployment_report(Path(directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["deployment"]["amp_dtype"] = "fp16"
            body = dict(payload)
            body.pop("report_sha256")
            payload["report_sha256"] = canonical_json_sha256(body)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantics are invalid"):
                load_official_native_ss_deployment(path)

    def test_aggregate_record_matrix_requires_every_object_seed_once(self) -> None:
        records = [
            {"object_uid": object_uid, "seed": seed}
            for object_uid in ("a", "b")
            for seed in (42, 43)
        ]
        validate_record_matrix(
            records,
            object_uids=["a", "b"],
            seeds=[42, 43],
            label="correct",
        )
        with self.assertRaisesRegex(RuntimeError, "record matrix differs"):
            validate_record_matrix(
                records[:-1] + [records[0]],
                object_uids=["a", "b"],
                seeds=[42, 43],
                label="correct",
            )


if __name__ == "__main__":
    unittest.main()
