from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from pose_aligned_reconstruction.train_native_slat_genrecon import (
    APPROVED_RESUME_DATA_PATH_FIELDS,
    checkpoint_args,
    make_parser,
    validate_args,
    validate_resume_data_identity,
)


def identity(paths: dict[tuple[str, ...], str]) -> dict:
    value = {
        "cache_manifest": paths[("cache_manifest",)],
        "cache_manifest_sha256": "cache-sha",
        "lifting_cache_manifest": paths[("lifting_cache_manifest",)],
        "lifting_cache_manifest_sha256": "lifting-sha",
        "config_hash": "config-sha",
        "sample_count": 2000,
        "object_count": 2000,
        "object_uids": ["a", "b"],
        "object_uid_hash": "uids-sha",
        "native_ss": {
            "report": paths[("native_ss", "report")],
            "report_sha256": "report-sha",
            "checkpoint": paths[("native_ss", "checkpoint")],
            "checkpoint_sha256": "checkpoint-sha",
            "checkpoint_step": 2000,
            "weights": "ema",
            "cfg_strength": 3.0,
            "steps": 25,
            "cfg_interval": [0.5, 1.0],
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
        "stock_slat_freeze_sha256": "freeze-sha",
        "target_decoder_audit": {
            "path": paths[("target_decoder_audit", "path")],
            "sha256": "audit-sha",
            "format": "audit.v1",
            "summary": {"object_count": 32},
            "thresholds": {"minimum": 1.0},
        },
    }
    return value


class ResumeDataIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.real = root / "real"
        self.alias = root / "alias"
        self.real.mkdir()
        self.alias.symlink_to(self.real, target_is_directory=True)
        self.saved_paths = {}
        self.current_paths = {}
        for index, field in enumerate(APPROVED_RESUME_DATA_PATH_FIELDS):
            real = self.real / f"file-{index}"
            real.write_text(str(index), encoding="utf-8")
            self.saved_paths[field] = str(self.alias / real.name)
            self.current_paths[field] = str(real)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_identity_needs_no_override(self) -> None:
        saved = identity(self.saved_paths)
        result = validate_resume_data_identity(
            saved, copy.deepcopy(saved), allow_path_relocation=False
        )
        self.assertFalse(result["applied"])

    def test_mismatch_without_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "resume data identity differs$"):
            validate_resume_data_identity(
                identity(self.saved_paths),
                identity(self.current_paths),
                allow_path_relocation=False,
            )

    def test_only_approved_paths_may_relocate(self) -> None:
        saved = identity(self.saved_paths)
        current = identity(self.current_paths)
        saved_before = copy.deepcopy(saved)
        current_before = copy.deepcopy(current)
        result = validate_resume_data_identity(
            saved, current, allow_path_relocation=True
        )
        self.assertTrue(result["applied"])
        self.assertEqual(len(result["approved_fields"]), 5)
        self.assertTrue(result["all_non_path_fields_exact"])
        self.assertEqual(saved, saved_before)
        self.assertEqual(current, current_before)

    def test_different_path_target_is_rejected(self) -> None:
        current = identity(self.current_paths)
        other = self.real / "different"
        other.write_text("different", encoding="utf-8")
        current["cache_manifest"] = str(other)
        with self.assertRaisesRegex(
            RuntimeError, "resume data path relocation points to different files"
        ):
            validate_resume_data_identity(
                identity(self.saved_paths), current, allow_path_relocation=True
            )

    def test_non_path_difference_is_rejected(self) -> None:
        current = identity(self.current_paths)
        current["native_ss"]["checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(
            RuntimeError,
            "resume data identity differs outside approved path relocation",
        ):
            validate_resume_data_identity(
                identity(self.saved_paths), current, allow_path_relocation=True
            )

    def test_override_requires_resume_and_is_not_checkpoint_bound(self) -> None:
        parser = make_parser()
        required = [
            "--cache_manifest", "cache",
            "--lifting_cache_manifest", "lifting",
            "--target_decoder_audit", "audit",
            "--native_ss_report", "ss",
            "--stock_slat_freeze", "freeze",
            "--output_dir", "output",
            "--allow_resume_data_path_relocation",
        ]
        args = parser.parse_args(required)
        with self.assertRaisesRegex(ValueError, "require --resume"):
            validate_args(args)
        args.resume = "checkpoint.pt"
        validate_args(args)
        self.assertNotIn("allow_resume_data_path_relocation", checkpoint_args(args))


if __name__ == "__main__":
    unittest.main()
