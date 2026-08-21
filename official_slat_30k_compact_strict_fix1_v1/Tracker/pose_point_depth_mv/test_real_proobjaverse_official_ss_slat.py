from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    END_TO_END_REPORT_FORMAT,
    END_TO_END_WORKER_FORMAT,
)
from pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat import (
    validate_cross_deployment_bridge,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.reconstruct_real_proobjaverse_official_ss_slat import (
    DEFAULT_BRIDGE_REPORT,
    DEFAULT_SLAT_CHECKPOINT,
    make_parser,
    resolve_view_selection_policy,
)


class CrossDeploymentBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ss_report = self.root / "ss_report.json"
        self.ss_checkpoint = self.root / "ss.pt"
        self.slat_checkpoint = self.root / "slat.pt"
        self.stock_freeze = self.root / "freeze.json"
        self.ss_report.write_text('{"passed":true}\n', encoding="utf-8")
        self.ss_checkpoint.write_bytes(b"ss checkpoint")
        self.slat_checkpoint.write_bytes(b"slat checkpoint")
        self.stock_freeze.write_text('{"passed":true}\n', encoding="utf-8")
        self.binding = {
            "report": str(self.ss_report),
            "report_sha256": sha256_file(self.ss_report),
            "checkpoint": str(self.ss_checkpoint),
            "checkpoint_sha256": sha256_file(self.ss_checkpoint),
            "checkpoint_step": 2000,
            "weights": "ema",
            "cfg_strength": 5.0,
            "steps": 25,
            "cfg_interval": [0.5, 1.0],
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
            "amp_dtype": "bf16",
            "false_checks": [],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_bridge(
        self,
        *,
        binding_override: dict | None = None,
        complete: bool = True,
    ) -> Path:
        binding = {**self.binding, **(binding_override or {})}
        worker = {
            "format": END_TO_END_WORKER_FORMAT,
            "complete": complete,
            "passed": False,
            "run_identity": {
                "native_ss_report_sha256": sha256_file(self.ss_report),
                "trained_slat_checkpoint_sha256": sha256_file(self.slat_checkpoint),
                "trained_slat_weights": "ema",
                "expected_trained_slat_step": 8000,
                "stock_slat_freeze_sha256": sha256_file(self.stock_freeze),
                "object_start": 16,
                "object_end": 17,
                "object_uids": ["heldout-object"],
            },
            "native_ss_binding": binding,
        }
        worker["report_sha256"] = canonical_sha256(worker)
        worker_path = self.root / "worker.json"
        worker_path.write_text(json.dumps(worker), encoding="utf-8")
        aggregate = {
            "format": END_TO_END_REPORT_FORMAT,
            "passed": False,
            "object_count": 1,
            "record_count": 3,
            "integrity": {"object_coverage_exact": True},
            "decision": {"native_ss_trained_slat_end_to_end_passed": False},
            "worker_reports": [
                {
                    "path": str(worker_path),
                    "sha256": sha256_file(worker_path),
                    "report_sha256": worker["report_sha256"],
                }
            ],
        }
        aggregate["report_sha256"] = canonical_sha256(aggregate)
        path = self.root / "aggregate.json"
        path.write_text(json.dumps(aggregate), encoding="utf-8")
        return path

    def _validate(self, bridge: Path):
        return validate_cross_deployment_bridge(
            bridge,
            native_ss_report=self.ss_report,
            native_ss_binding=self.binding,
            trained_slat_checkpoint=self.slat_checkpoint,
            trained_slat_step=8000,
            trained_slat_weights="ema",
            stock_slat_freeze=self.stock_freeze,
        )

    def test_complete_exact_binding_passes_even_when_science_gate_is_false(self) -> None:
        result = self._validate(self._write_bridge())
        self.assertTrue(result["passed"])
        self.assertFalse(result["aggregate_runtime_passed"])
        self.assertEqual(result["object_count"], 1)

    def test_changed_slat_checkpoint_is_rejected(self) -> None:
        bridge = self._write_bridge()
        self.slat_checkpoint.write_bytes(b"changed slat checkpoint")
        with self.assertRaisesRegex(RuntimeError, "artifact bindings differ"):
            self._validate(bridge)

    def test_changed_native_ss_semantic_field_is_rejected(self) -> None:
        bridge = self._write_bridge(binding_override={"cfg_strength": 3.0})
        with self.assertRaisesRegex(RuntimeError, "Native-SS semantics differ"):
            self._validate(bridge)

    def test_incomplete_worker_is_rejected(self) -> None:
        bridge = self._write_bridge(complete=False)
        with self.assertRaisesRegex(RuntimeError, "worker is incomplete"):
            self._validate(bridge)


class DeploymentDefaultTest(unittest.TestCase):
    def test_generic_entrypoint_defaults_to_step25000(self) -> None:
        args = make_parser().parse_args(
            ["--dataset_dir", "/tmp/example", "--session_id", "example"]
        )
        self.assertEqual(args.expected_slat_step, 25000)
        self.assertEqual(args.native_slat_checkpoint, DEFAULT_SLAT_CHECKPOINT)
        self.assertIn("step_025000.pt", str(DEFAULT_SLAT_CHECKPOINT))
        self.assertEqual(args.cross_deployment_bridge_report, DEFAULT_BRIDGE_REPORT)
        self.assertIn("step_025000/dev48_predicted", str(DEFAULT_BRIDGE_REPORT))
        self.assertEqual(
            resolve_view_selection_policy(args.view_selection_policy),
            "object_spherical_farthest_valid_mask",
        )

    def test_current_two_case_launcher_is_explicitly_step8000(self) -> None:
        launcher = (
            Path(__file__).parent
            / "background_jobs/run_real_official_slat_step8000_two_cases.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("step_008000.pt", launcher)
        self.assertIn("--expected_slat_step 8000", launcher)
        self.assertIn(
            "49edb3bbdbd86b10c5eea14e9c80a9996076b6fd65a459db12b130b6560bda4d",
            launcher,
        )

    def test_reusable_launcher_defaults_to_step25000_and_supports_step8000(self) -> None:
        launcher = (
            Path(__file__).parent
            / "background_jobs/run_real_official_slat_retest_one.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("MODEL_STEP=${MODEL_STEP:-25000}", launcher)
        self.assertIn("step_025000.pt", launcher)
        self.assertIn("step_008000.pt", launcher)
        self.assertIn('INPUT_PATH=${1:?', launcher)
        self.assertIn("RUN_TAG=${RUN_TAG:-spherical_v1}", launcher)
        self.assertIn('policy = "object_spherical_farthest_valid_mask"', launcher)

    def test_mobile_capture_and_server_record_pose_diversity_contract(self) -> None:
        tracker_root = Path(__file__).resolve().parents[1]
        client = (tracker_root / "ARposeTTracker.cs").read_text(encoding="utf-8")
        server = (tracker_root / "trellis_point_prior_mv/server.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("EvaluatePoseDiverseCandidate", client)
        self.assertIn("minimumPoseDiversityAngleDegrees = 10.0f", client)
        self.assertIn('"capture_view_policy"', client)
        self.assertIn("string nearestPoseAngleText", client)
        self.assertNotIn('? \\"首帧\\"', client)
        self.assertIn('"schema": "arpose_tracker_frame_metadata_v3"', server)
        self.assertIn('"pose_diverse_capture"', server)


if __name__ == "__main__":
    unittest.main()
