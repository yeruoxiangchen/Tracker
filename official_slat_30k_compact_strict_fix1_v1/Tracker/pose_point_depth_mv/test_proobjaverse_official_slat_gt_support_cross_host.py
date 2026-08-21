#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support_cross_host import (
    validate_checkpoint_native_ss_binding_relocation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointNativeSSBindingRelocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.report = self.root / "report.json"
        self.checkpoint = self.root / "step_002000.pt"
        self.report.write_bytes(b"fixed report\n")
        self.checkpoint.write_bytes(b"fixed checkpoint\n")
        self.runtime = {
            "report": str(self.report),
            "report_sha256": _sha256(self.report),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_step": 2000,
            "weights": "ema",
            "cfg_strength": 3.0,
            "steps": 25,
            "cfg_interval": [0.5, 1.0],
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        }
        self.saved = {
            **self.runtime,
            "report": str(self.root / "missing-host" / "report.json"),
            "checkpoint": str(self.root / "missing-host" / "step_002000.pt"),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_binding_needs_no_override(self) -> None:
        result = validate_checkpoint_native_ss_binding_relocation(
            self.runtime, self.runtime, allow_path_relocation=False
        )
        self.assertFalse(result["path_relocated"])

    def test_mismatch_without_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "upstream Native SS differs"):
            validate_checkpoint_native_ss_binding_relocation(
                self.saved, self.runtime, allow_path_relocation=False
            )

    def test_two_content_addressed_paths_can_relocate(self) -> None:
        result = validate_checkpoint_native_ss_binding_relocation(
            self.saved, self.runtime, allow_path_relocation=True
        )
        self.assertTrue(result["path_relocated"])
        self.assertEqual(set(result["relocations"]), {"report", "checkpoint"})

    def test_untraversable_foreign_host_path_can_relocate_by_hash(self) -> None:
        blocker = self.root / "foreign-root"
        blocker.write_bytes(b"not a directory\n")
        saved = {
            **self.saved,
            "report": str(blocker / "data" / "report.json"),
        }
        result = validate_checkpoint_native_ss_binding_relocation(
            saved, self.runtime, allow_path_relocation=True
        )
        self.assertTrue(result["path_relocated"])
        self.assertIsNotNone(
            result["relocations"]["report"]["saved_path_unavailable_reason"]
        )

    def test_semantic_field_change_is_rejected(self) -> None:
        changed = {**self.runtime, "cfg_strength": 5.0}
        with self.assertRaisesRegex(RuntimeError, "outside approved path relocation"):
            validate_checkpoint_native_ss_binding_relocation(
                self.saved, changed, allow_path_relocation=True
            )

    def test_runtime_content_change_is_rejected(self) -> None:
        self.report.write_bytes(b"changed report\n")
        with self.assertRaisesRegex(RuntimeError, "content SHA256 changed"):
            validate_checkpoint_native_ss_binding_relocation(
                self.saved, self.runtime, allow_path_relocation=True
            )

    def test_unknown_field_change_is_rejected(self) -> None:
        changed = {**self.runtime, "future_semantic_field": "new"}
        with self.assertRaisesRegex(RuntimeError, "outside approved path relocation"):
            validate_checkpoint_native_ss_binding_relocation(
                self.saved, changed, allow_path_relocation=True
            )

    def test_hash_binding_change_is_rejected(self) -> None:
        changed = {**self.runtime, "checkpoint_sha256": "0" * 64}
        with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256 changed"):
            validate_checkpoint_native_ss_binding_relocation(
                self.saved, changed, allow_path_relocation=True
            )


if __name__ == "__main__":
    unittest.main()
