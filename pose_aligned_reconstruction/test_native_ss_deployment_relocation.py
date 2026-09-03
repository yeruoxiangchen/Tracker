from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from pose_aligned_reconstruction.train_native_slat_genrecon import (
    validate_native_ss_deployment,
)


def deployment(*, report: str, checkpoint: str) -> dict:
    return {
        "report": report,
        "report_sha256": "report-sha",
        "checkpoint": checkpoint,
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_step": 2000,
        "weights": "ema",
        "cfg_strength": 3.0,
        "steps": 25,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
        "amp_dtype": "bf16",
        "false_checks": [],
    }


class NativeSSDeploymentRelocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.real = root / "real"
        self.alias = root / "alias"
        self.real.mkdir()
        self.alias.symlink_to(self.real, target_is_directory=True)
        self.report = self.real / "report.json"
        self.checkpoint = self.real / "checkpoint.pt"
        self.report.write_text("report", encoding="utf-8")
        self.checkpoint.write_text("checkpoint", encoding="utf-8")
        self.frozen = deployment(
            report=str(self.alias / self.report.name),
            checkpoint=str(self.alias / self.checkpoint.name),
        )
        self.runtime = deployment(
            report=str(self.report), checkpoint=str(self.checkpoint)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self, frozen: dict, runtime: dict) -> dict:
        return validate_native_ss_deployment(
            frozen, runtime, allow_path_relocation=True
        )

    def assert_semantic_rejected(self, key: str, value) -> None:
        runtime = copy.deepcopy(self.runtime)
        runtime[key] = value
        with self.assertRaisesRegex(RuntimeError, "outside approved path relocation"):
            self.validate(self.frozen, runtime)

    def test_exact_deployment_needs_no_override(self) -> None:
        result = validate_native_ss_deployment(
            self.frozen, copy.deepcopy(self.frozen), allow_path_relocation=False
        )
        self.assertFalse(result["path_relocated"])
        self.assertEqual(result["relocations"], {})

    def test_mismatch_without_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "SLAT cache and training bind different Native SS deployments$"
        ):
            validate_native_ss_deployment(
                self.frozen, self.runtime, allow_path_relocation=False
            )

    def test_report_path_relocation_to_same_file_passes(self) -> None:
        runtime = copy.deepcopy(self.frozen)
        runtime["report"] = str(self.report)
        result = self.validate(self.frozen, runtime)
        self.assertEqual(set(result["relocations"]), {"report"})

    def test_checkpoint_path_relocation_to_same_file_passes(self) -> None:
        runtime = copy.deepcopy(self.frozen)
        runtime["checkpoint"] = str(self.checkpoint)
        result = self.validate(self.frozen, runtime)
        self.assertEqual(set(result["relocations"]), {"checkpoint"})

    def test_report_path_to_different_file_is_rejected(self) -> None:
        other = self.real / "other-report.json"
        other.write_text("other", encoding="utf-8")
        runtime = copy.deepcopy(self.runtime)
        runtime["report"] = str(other)
        with self.assertRaisesRegex(RuntimeError, "different files: report"):
            self.validate(self.frozen, runtime)

    def test_checkpoint_path_to_different_file_is_rejected(self) -> None:
        other = self.real / "other-checkpoint.pt"
        other.write_text("other", encoding="utf-8")
        runtime = copy.deepcopy(self.runtime)
        runtime["checkpoint"] = str(other)
        with self.assertRaisesRegex(RuntimeError, "different files: checkpoint"):
            self.validate(self.frozen, runtime)

    def test_amp_dtype_change_is_rejected(self) -> None:
        self.assert_semantic_rejected("amp_dtype", "fp16")

    def test_false_checks_change_is_rejected(self) -> None:
        self.assert_semantic_rejected("false_checks", ["quality_gate"])

    def test_cfg_strength_change_is_rejected(self) -> None:
        self.assert_semantic_rejected("cfg_strength", 5.0)

    def test_report_sha256_change_is_rejected(self) -> None:
        self.assert_semantic_rejected("report_sha256", "changed")

    def test_checkpoint_sha256_change_is_rejected(self) -> None:
        self.assert_semantic_rejected("checkpoint_sha256", "changed")

    def test_unknown_semantic_field_add_or_delete_is_rejected(self) -> None:
        added = copy.deepcopy(self.runtime)
        added["future_semantic"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "outside approved path relocation"):
            self.validate(self.frozen, added)
        deleted = copy.deepcopy(self.runtime)
        del deleted["amp_dtype"]
        with self.assertRaisesRegex(RuntimeError, "outside approved path relocation"):
            self.validate(self.frozen, deleted)


if __name__ == "__main__":
    unittest.main()
