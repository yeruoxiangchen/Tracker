from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from official_ss_with_vggt_perf_v1.artifact_validation import (
    ENDPOINT_AGGREGATE_FORMAT,
    ENDPOINT_COMPARISONS,
    ENDPOINT_VERSION,
    ENDPOINT_WORKER_FORMAT,
    TARGET_JOIN_CONTRACT,
    VSS_AGGREGATE_FORMAT,
    canonical_sha256,
    sha256_file,
    validate_endpoint_aggregate,
    validate_endpoint_worker,
    validate_vss_aggregate,
)


def _write_report(path: Path, payload: dict) -> Path:
    value = dict(payload)
    value.pop("report_sha256", None)
    value["report_sha256"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _contract() -> dict:
    return {
        "version": ENDPOINT_VERSION,
        "branches": {"A": "a", "B": "b", "C": "c"},
        "comparisons": {
            "B_minus_A": "b-a",
            "C_minus_B": "c-b",
            "C_minus_A": "c-a",
        },
        "slat_support_input": "predicted_only",
        "gt_support_used_as_slat_input": False,
        "official_target_join_contract": dict(TARGET_JOIN_CONTRACT),
    }


class ArtifactValidationTests(unittest.TestCase):
    def _endpoint_fixture(self, root: Path) -> dict[str, Path]:
        slat = root / "step_010000.pt"
        ss_cache = root / "with_vggt_ss_manifest.json"
        vss = root / "vss_report.json"
        slat.write_bytes(b"slat-checkpoint")
        ss_cache.write_text("{}\n", encoding="utf-8")
        vss.write_text("{}\n", encoding="utf-8")
        uids = ["dev-a", "dev-b"]
        seeds = [42, 43, 44]
        ss_records = [
            {
                "object_uid": uid,
                "seed": seed,
                "same_initial_noise": True,
                "passed": True,
            }
            for uid in uids
            for seed in seeds
        ]
        mesh_records = [
            {
                "pair_id": hashlib.sha256(
                    f"{uid}|{seed}".encode("utf-8")
                ).hexdigest()[:24],
                "object_uid": uid,
                "seed": seed,
                "branch": branch,
                "passed": True,
                "surface": {"chamfer_l1": 0.1},
                "structure": {"mesh_success": True},
            }
            for uid in uids
            for seed in seeds
            for branch in ("stock", "native", "native_trained")
        ]
        worker = _write_report(
            root / "worker" / "report.json",
            {
                "format": ENDPOINT_WORKER_FORMAT,
                "complete": True,
                "passed": True,
                "formal": False,
                "run_identity": {
                    "format": ENDPOINT_WORKER_FORMAT,
                    "object_start": 16,
                    "object_end": 18,
                    "object_uids": uids,
                    "joint_seeds": [42, 43, 44],
                    "expected_trained_slat_step": 10000,
                    "trained_slat_checkpoint": str(slat.resolve()),
                    "trained_slat_checkpoint_sha256": sha256_file(slat),
                    "ss_cache_manifest": str(ss_cache.resolve()),
                    "ss_cache_manifest_sha256": sha256_file(ss_cache),
                    "native_ss_report": str(vss.resolve()),
                    "native_ss_report_sha256": sha256_file(vss),
                    "slat_support_input": "predicted_only",
                    "gt_support_used_as_slat_input": False,
                },
                "object_count": 2,
                "record_count": 6,
                "ss_records": ss_records,
                "mesh_branch_records": mesh_records,
                "with_vggt_endpoint_contract": _contract(),
                "evaluation_split": "dev",
            },
        )
        aggregate = _write_report(
            root / "aggregate" / "report.json",
            {
                "format": ENDPOINT_AGGREGATE_FORMAT,
                "passed": True,
                "formal": False,
                "object_start": 16,
                "object_end": 18,
                "object_count": 2,
                "record_count": 6,
                "joint_seeds": [42, 43, 44],
                "worker_reports": [
                    {
                        "path": str(worker.resolve()),
                        "sha256": sha256_file(worker),
                        "report_sha256": json.loads(worker.read_text())["report_sha256"],
                    }
                ],
                "integrity": {
                    "object_coverage_exact": True,
                    "ss_records_exact": True,
                    "mesh_pairs_exact": True,
                },
                "endpoint_runtime_integrity": {
                    "passed": True,
                    "complete_branch_record_matrix": True,
                    "only_registered_model_output_failures": True,
                    "all_model_outputs_passed": True,
                    "registered_model_output_failure_count": 0,
                    "registered_model_output_failures": [],
                },
                "decision": {
                    # A science-negative result is valid runtime evidence.
                    "native_ss_trained_slat_end_to_end_passed": False,
                },
                "with_vggt_endpoint_contract": _contract(),
                "evaluation_split": "dev",
                "comparisons": {key: {} for key in ENDPOINT_COMPARISONS},
                "extended_pairwise_metrics": {
                    key: {} for key in ENDPOINT_COMPARISONS
                },
            },
        )
        return {
            "slat": slat,
            "ss_cache": ss_cache,
            "vss": vss,
            "worker": worker,
            "aggregate": aggregate,
        }

    def test_science_negative_endpoint_is_valid_but_not_mislabeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            result = validate_endpoint_aggregate(
                fixture["aggregate"],
                expected_split="dev",
                expected_start=16,
                expected_end=18,
                expected_objects=2,
                expected_seeds="42,43,44",
                expected_slat_step=10000,
                expected_slat_checkpoint=fixture["slat"],
                expected_ss_cache=fixture["ss_cache"],
                expected_vss_report=fixture["vss"],
            )
        self.assertTrue(result["runtime_integrity_passed"])
        self.assertFalse(result["full_endpoint_science_passed"])

    def test_worker_step_or_checkpoint_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                validate_endpoint_worker(
                    fixture["worker"],
                    expected_split="dev",
                    expected_start=16,
                    expected_end=18,
                    expected_seeds=[42, 43, 44],
                    expected_slat_step=15000,
                    expected_slat_checkpoint=fixture["slat"],
                    expected_ss_cache=fixture["ss_cache"],
                    expected_vss_report=fixture["vss"],
                )

    def test_registered_model_output_failure_is_valid_program_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            worker = json.loads(fixture["worker"].read_text(encoding="utf-8"))
            failed = worker["mesh_branch_records"][0]
            failed["passed"] = False
            failed.pop("surface")
            failed.pop("structure")
            failed["error"] = {
                "type": "RuntimeError",
                "stage": "stock_slat_mesh_decode",
                "message": (
                    "SLat decoder input exceeds safe active-point limit: "
                    "points=90000 limit=80000"
                ),
            }
            worker["passed"] = False
            _write_report(fixture["worker"], worker)

            aggregate = json.loads(
                fixture["aggregate"].read_text(encoding="utf-8")
            )
            aggregate["passed"] = False
            aggregate["integrity"]["mesh_pairs_exact"] = False
            aggregate["worker_reports"][0].update(
                {
                    "sha256": sha256_file(fixture["worker"]),
                    "report_sha256": json.loads(
                        fixture["worker"].read_text(encoding="utf-8")
                    )["report_sha256"],
                }
            )
            aggregate["endpoint_runtime_integrity"].update(
                {
                    "all_model_outputs_passed": False,
                    "registered_model_output_failure_count": 1,
                    "registered_model_output_failures": [
                        {
                            "object_uid": failed["object_uid"],
                            "seed": failed["seed"],
                            "branch": failed["branch"],
                            "error": failed["error"],
                        }
                    ],
                }
            )
            _write_report(fixture["aggregate"], aggregate)
            result = validate_endpoint_aggregate(
                fixture["aggregate"],
                expected_split="dev",
                expected_start=16,
                expected_end=18,
                expected_objects=2,
                expected_seeds="42,43,44",
                expected_slat_step=10000,
                expected_slat_checkpoint=fixture["slat"],
                expected_ss_cache=fixture["ss_cache"],
                expected_vss_report=fixture["vss"],
            )
        self.assertTrue(result["runtime_integrity_passed"])
        self.assertFalse(result["all_model_outputs_passed"])
        self.assertEqual(result["registered_model_output_failure_count"], 1)

    def test_unregistered_model_output_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            worker = json.loads(fixture["worker"].read_text(encoding="utf-8"))
            failed = worker["mesh_branch_records"][0]
            failed["passed"] = False
            failed["error"] = {
                "type": "RuntimeError",
                "stage": "stock_slat_mesh_decode",
                "message": "CUDA out of memory",
            }
            worker["passed"] = False
            _write_report(fixture["worker"], worker)
            with self.assertRaisesRegex(RuntimeError, "unregistered failure"):
                validate_endpoint_worker(
                    fixture["worker"],
                    expected_split="dev",
                    expected_start=16,
                    expected_end=18,
                    expected_seeds=[42, 43, 44],
                    expected_slat_step=10000,
                    expected_slat_checkpoint=fixture["slat"],
                    expected_ss_cache=fixture["ss_cache"],
                    expected_vss_report=fixture["vss"],
                )

    def test_registered_cuda_decoder_topology_failure_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            worker = json.loads(fixture["worker"].read_text(encoding="utf-8"))
            failed = worker["mesh_branch_records"][0]
            failed["passed"] = False
            failed.pop("surface")
            failed.pop("structure")
            failed["error"] = {
                "type": "RuntimeError",
                "stage": "stock_slat_mesh_decode",
                "message": (
                    "SLat decoder CUDA topology failure: branch=stock "
                    "uid=dev-a seed=42: cudaErrorIllegalAddress"
                ),
            }
            worker["passed"] = False
            _write_report(fixture["worker"], worker)
            result = validate_endpoint_worker(
                fixture["worker"],
                expected_split="dev",
                expected_start=16,
                expected_end=18,
                expected_seeds=[42, 43, 44],
                expected_slat_step=10000,
                expected_slat_checkpoint=fixture["slat"],
                expected_ss_cache=fixture["ss_cache"],
                expected_vss_report=fixture["vss"],
            )
        self.assertTrue(result["runtime_passed"])
        self.assertFalse(result["all_model_outputs_passed"])
        self.assertEqual(result["registered_model_output_failure_count"], 1)

    def test_v1_target_join_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            payload = json.loads(fixture["worker"].read_text(encoding="utf-8"))
            payload["with_vggt_endpoint_contract"].pop(
                "official_target_join_contract"
            )
            _write_report(fixture["worker"], payload)
            with self.assertRaisesRegex(RuntimeError, "branch/support contract differs"):
                validate_endpoint_worker(
                    fixture["worker"],
                    expected_split="dev",
                    expected_start=16,
                    expected_end=18,
                    expected_seeds=[42, 43, 44],
                    expected_slat_step=10000,
                    expected_slat_checkpoint=fixture["slat"],
                    expected_ss_cache=fixture["ss_cache"],
                    expected_vss_report=fixture["vss"],
                )

    def test_tampered_existing_worker_report_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._endpoint_fixture(Path(temporary))
            fixture["worker"].write_text(
                fixture["worker"].read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "worker file hash differs"):
                validate_endpoint_aggregate(
                    fixture["aggregate"],
                    expected_split="dev",
                    expected_start=16,
                    expected_end=18,
                    expected_objects=2,
                    expected_seeds="42,43,44",
                    expected_slat_step=10000,
                    expected_slat_checkpoint=fixture["slat"],
                    expected_ss_cache=fixture["ss_cache"],
                    expected_vss_report=fixture["vss"],
                )

    def test_vss_science_negative_report_keeps_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "step_002000.pt"
            calibration = root / "calibration.json"
            checkpoint.write_bytes(b"vss-checkpoint")
            calibration.write_text("{}\n", encoding="utf-8")
            report = _write_report(
                root / "vss" / "report.json",
                {
                    "format": VSS_AGGREGATE_FORMAT,
                    "passed": False,
                    "formal": False,
                    "object_count": 48,
                    "record_count": 144,
                    "protocol": {"joint_seeds": [42, 43, 44]},
                    "calibration_sha256": sha256_file(calibration),
                    "checks": {
                        "correct_record_matrix_exact": True,
                        "pose_control_record_matrix_exact": True,
                        "stock_baseline_nonempty": True,
                        "disabled_stock_equivalence": True,
                        "iou_gain_mean": False,
                    },
                    "deployment": {
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "checkpoint_step": 2000,
                        "weights": "ema",
                    },
                },
            )
            result = validate_vss_aggregate(
                report,
                expected_checkpoint=checkpoint,
                expected_calibration=calibration,
            )
        self.assertTrue(result["runtime_integrity_passed"])
        self.assertFalse(result["science_passed"])

    def test_formal_launcher_retains_all_registered_ss_anchors(self) -> None:
        launcher = Path(__file__).with_name("run_train_8gpu.sh").read_text(
            encoding="utf-8"
        )
        pipeline = Path(__file__).with_name(
            "run_source_p5retry_p6_p7_p8_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--save_every 500", launcher)
        self.assertIn("REQUIRED_CHECKPOINT_STEPS=(500 1000 1500 2000)", launcher)
        self.assertIn("for checkpoint_step in 500 1000 1500 2000", pipeline)


if __name__ == "__main__":
    unittest.main()
