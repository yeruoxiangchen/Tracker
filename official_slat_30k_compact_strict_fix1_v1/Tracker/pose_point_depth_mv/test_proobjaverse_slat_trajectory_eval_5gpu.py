#!/usr/bin/env python3
"""Unit tests for the five-GPU SLat trajectory evaluation orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.run_proobjaverse_slat_trajectory_eval_5gpu import (
    DEFAULT_STEPS,
    GT_RANGES,
    PREDICTED_RANGES,
    Job,
    build_jobs,
    validate_partition,
    validate_worker_report,
)


class TrajectoryPlanTest(unittest.TestCase):
    def test_partitions_are_exact(self) -> None:
        validate_partition(GT_RANGES, expected_start=0, expected_end=64)
        validate_partition(PREDICTED_RANGES, expected_start=16, expected_end=64)

    def test_plan_has_nine_waves_and_forty_five_workers(self) -> None:
        waves = build_jobs(
            steps=DEFAULT_STEPS,
            gpus=(3, 4, 5, 6, 7),
            checkpoint_root=Path("/checkpoints"),
            output_root=Path("/outputs"),
        )
        self.assertEqual(len(waves), 9)
        self.assertEqual(sum(len(wave) for wave in waves), 45)
        self.assertTrue(all(len(wave) == 5 for wave in waves))
        self.assertEqual(
            [(wave[0].step, wave[0].group) for wave in waves],
            [
                (step, group)
                for step in DEFAULT_STEPS
                for group in ("train64_gt", "dev64_gt", "dev48_predicted")
            ],
        )

    def test_requires_exactly_five_gpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly five"):
            build_jobs(
                steps=DEFAULT_STEPS,
                gpus=(3, 4, 5, 6),
                checkpoint_root=Path("/checkpoints"),
                output_root=Path("/outputs"),
            )


class WorkerReportTest(unittest.TestCase):
    def _job(self, root: Path, group: str) -> Job:
        return Job(
            step=15000,
            group=group,
            shard=0,
            start=16 if group == "dev48_predicted" else 0,
            end=26 if group == "dev48_predicted" else 13,
            gpu=3,
            checkpoint=Path("/checkpoints/step_015000.pt"),
            output_dir=root,
            log_path=root / "worker.log",
        )

    def test_finalized_predicted_failure_is_a_completed_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "complete": True,
                "passed": False,
                "run_identity": {
                    "expected_trained_slat_step": 15000,
                    "object_start": 16,
                    "object_end": 26,
                    "trained_slat_checkpoint": "/checkpoints/step_015000.pt",
                },
            }
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            result = validate_worker_report(self._job(root, "dev48_predicted"))
            self.assertFalse(result["passed"])
            self.assertTrue(result["complete"])

    def test_incomplete_predicted_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "complete": False,
                "passed": False,
                "run_identity": {
                    "expected_trained_slat_step": 15000,
                    "object_start": 16,
                    "object_end": 26,
                    "trained_slat_checkpoint": "/checkpoints/step_015000.pt",
                },
            }
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                validate_worker_report(self._job(root, "dev48_predicted"))

    def test_gt_report_must_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "passed": False,
                "run_config": {
                    "checkpoint_step": 15000,
                    "object_start": 0,
                    "object_end": 13,
                    "native_ss_executed": False,
                },
            }
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                validate_worker_report(self._job(root, "train64_gt"))


if __name__ == "__main__":
    unittest.main()
