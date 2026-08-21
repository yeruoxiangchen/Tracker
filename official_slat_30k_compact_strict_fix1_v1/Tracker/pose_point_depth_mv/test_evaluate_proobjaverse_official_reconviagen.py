from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from pose_point_depth_mv import evaluate_proobjaverse_official_reconviagen as mod
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256


def _write_hashed(path: Path, payload: dict) -> None:
    body = dict(payload)
    body["report_sha256"] = canonical_sha256(body)
    path.write_text(json.dumps(body), encoding="utf-8")


class TerminalFailureTest(unittest.TestCase):
    def test_spconv_int32_failure_is_recordable(self) -> None:
        with mock.patch.dict(os.environ, {"SPCONV_ALGO": "native"}):
            failure = mod._classify_terminal_recon_failure(
                RuntimeError("assert faild. your data exceed int32 range.")
            )
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure["kind"], "spconv_int32_range_exceeded")
        self.assertEqual(failure["spconv_algo"], "native")
        self.assertFalse(failure["retryable"])

    def test_environment_failures_are_not_recordable(self) -> None:
        self.assertIsNone(
            mod._classify_terminal_recon_failure(
                RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            )
        )
        self.assertIsNone(
            mod._classify_terminal_recon_failure(
                RuntimeError("CUDA error: an illegal memory access was encountered")
            )
        )

    def test_resume_reuses_only_approved_failure_on_same_backend(self) -> None:
        expected = {"format": mod.RECORD_FORMAT, "object_uid": "u", "seed": 42}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            record = {
                **expected,
                "passed": False,
                "spconv_algo": "native",
                "error": {
                    "type": "RuntimeError",
                    "kind": "spconv_int32_range_exceeded",
                    "message": "your data exceed int32 range",
                    "stage": "stock_mesh_decoder_sparse_convolution",
                    "retryable": False,
                    "spconv_algo": "native",
                },
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.dict(os.environ, {"SPCONV_ALGO": "native"}):
                self.assertEqual(
                    mod._load_reusable(path, expected, resume=True), record
                )
            with mock.patch.dict(os.environ, {"SPCONV_ALGO": "implicit_gemm"}):
                with self.assertRaisesRegex(RuntimeError, "spconv_algo"):
                    mod._load_reusable(path, expected, resume=True)

    def test_resume_rejects_unknown_failed_record(self) -> None:
        expected = {"format": mod.RECORD_FORMAT, "object_uid": "u", "seed": 42}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(
                json.dumps(
                    {
                        **expected,
                        "passed": False,
                        "spconv_algo": "native",
                        "error": {
                            "kind": "cuda_oom",
                            "retryable": False,
                            "spconv_algo": "native",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"SPCONV_ALGO": "native"}):
                with self.assertRaisesRegex(RuntimeError, "terminal_failure"):
                    mod._load_reusable(path, expected, resume=True)


class AggregateFailureTest(unittest.TestCase):
    def test_aggregate_reports_failure_and_uses_complete_object_intersection(self) -> None:
        uids = [f"uid-{index:02d}" for index in range(64)]
        target_structure = {"mesh_success": True, "largest_component_ratio": 1.0}
        recon_records = []
        for index, uid in enumerate(uids):
            for seed in (42, 43, 44):
                common = {
                    "object_index": index,
                    "object_uid": uid,
                    "uid": uid,
                    "seed": seed,
                    "spconv_algo": "native",
                    "target_structure": target_structure,
                }
                if index == 21 and seed == 42:
                    recon_records.append(
                        {
                            **common,
                            "passed": False,
                            "surface": None,
                            "structure": {"mesh_success": False},
                            "error": {
                                "type": "RuntimeError",
                                "kind": "spconv_int32_range_exceeded",
                                "message": "your data exceed int32 range",
                                "stage": "stock_mesh_decoder_sparse_convolution",
                                "retryable": False,
                                "spconv_algo": "native",
                            },
                        }
                    )
                else:
                    recon_records.append(
                        {
                            **common,
                            "passed": True,
                            "surface": {
                                "chamfer_l1": 0.20,
                                "fscore_0p02": 0.70,
                                "normal_consistency": 0.80,
                            },
                            "structure": {
                                "mesh_success": True,
                                "largest_component_ratio": 0.90,
                            },
                        }
                    )

        recon_report = {
            "format": mod.WORKER_FORMAT,
            "complete": True,
            "passed": False,
            "method": mod.RECON_METHOD,
            "worker_index": 0,
            "num_workers": 1,
            "object_count": 64,
            "record_count": 192,
            "seeds": [42, 43, 44],
            "dev_split_sha256": "split",
            "cache_report_sha256": "cache",
            "target_report_sha256": "target",
            "paired_target_cache_roots": ["targets"],
            "official_protocol_sha256": "protocol",
            "pretrained": "Stable-X/trellis-vggt-v0-2",
            "sampling": {"fixed": True},
            "sampling_sha256": "sampling",
            "surface_samples": 20000,
            "predicted_meshes_saved": False,
            "render_previews_saved": False,
            "spconv_algo": "native",
            "records": recon_records,
        }

        current_rows = []
        for uid in uids[16:64]:
            for seed in (42, 43, 44):
                if uid == uids[21]:
                    current_rows.append(
                        {
                            "branch": "native_trained",
                            "object_uid": uid,
                            "seed": seed,
                            "passed": False,
                            "error": {"type": "RuntimeError", "stage": "decode"},
                        }
                    )
                else:
                    current_rows.append(
                        {
                            "branch": "native_trained",
                            "object_uid": uid,
                            "seed": seed,
                            "passed": True,
                            "target_structure": target_structure,
                            "surface": {
                                "chamfer_l1": 0.10,
                                "fscore_0p02": 0.80,
                                "normal_consistency": 0.90,
                            },
                            "structure": {
                                "mesh_success": True,
                                "largest_component_ratio": 0.95,
                            },
                        }
                    )
        current_report = {
            "format": mod.CURRENT_WORKER_FORMAT,
            "complete": True,
            "passed": False,
            "run_identity": {
                "expected_trained_slat_step": 25000,
                "trained_slat_checkpoint_sha256": "checkpoint",
                "joint_seeds": [42, 43, 44],
                "weights": "ema",
                "surface_samples": 20000,
                "save_meshes": False,
                "object_start": 16,
                "object_end": 64,
                "object_uids": uids[16:64],
            },
            "mesh_branch_records": current_rows,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recon_path = root / "recon.json"
            current_path = root / "current.json"
            output = root / "aggregate"
            _write_hashed(recon_path, recon_report)
            _write_hashed(current_path, current_report)
            args = SimpleNamespace(
                recon_reports=str(recon_path),
                current_reports=str(current_path),
                expected_current_step=25000,
                expected_current_sha256="checkpoint",
                bootstrap_samples=50,
                output_dir=str(output),
            )
            contract = {
                "rows": [{"uid": uid} for uid in uids],
                "protocol_sha256": "protocol",
            }
            with mock.patch.object(mod, "_load_contract", return_value=contract):
                mod.run_aggregate(args)
            report = json.loads((output / "report.json").read_text())

        self.assertTrue(report["runtime_integrity_passed"])
        self.assertEqual(
            report["strict_reconviagen_dev64"]["successful_record_count"], 191
        )
        self.assertEqual(
            report["strict_reconviagen_dev64"]["complete_surface_object_count"],
            63,
        )
        self.assertEqual(
            report["strict_reconviagen_heldout_dev48"][
                "complete_surface_object_count"
            ],
            47,
        )
        self.assertEqual(report["paired_complete_object_count"], 47)
        self.assertEqual(len(report["reconviagen_failed_seed_records"]), 1)
        self.assertEqual(report["excluded_incomplete_object_uids"], [uids[21]])


if __name__ == "__main__":
    unittest.main()
