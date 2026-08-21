#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION

from official_slat_with_vggt_perf_v1 import runtime
from official_slat_with_vggt_perf_v1.dataset import (
    DirectSLatWithoutLegacyCondition,
    StrictWithVGGTNativeConditionSLatDataset,
)
from official_slat_with_vggt_perf_v1 import train
from official_slat_with_vggt_perf_v1 import train_proobjaverse_official


class RuntimeSourceTest(unittest.TestCase):
    def test_exact_runtime_sources_and_ddp_policy(self) -> None:
        identity = runtime.validate_runtime_sources()
        self.assertEqual(identity["version"], runtime.RUNTIME_VERSION)
        self.assertFalse(identity["scientific_math_changed"])
        strict_train, strict_projection, loaded = runtime.load_strict_modules()
        self.assertEqual(identity, loaded)
        self.assertIsNone(strict_train.strict_perf_ddp_kwargs()["device_ids"])
        self.assertTrue(
            callable(strict_projection.validate_strict_cpu_lifting_sample)
        )

    def test_any_locked_source_hash_drift_is_rejected(self) -> None:
        real_sha256 = runtime._sha256
        first = next(iter(runtime.SCIENTIFIC_SOURCE_SHA256))

        def changed(path: Path) -> str:
            if str(path).endswith(first):
                return "0" * 64
            return real_sha256(path)

        with mock.patch.object(runtime, "_sha256", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "scientific source changed"):
                runtime.validate_runtime_sources()


class LeanDatasetTest(unittest.TestCase):
    def _dataset(self, root: Path, *, finite: bool = True):
        uid = "unit-uid"
        object_uid = "unit-object"
        target = root / "target.npz"
        np.savez(
            target,
            coords=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
            feats=np.ones((2, 8), dtype=np.float32),
        )
        support_path = root / "support.pt"
        physical_path = root / "physical.pt"
        condition_path = root / "condition-must-not-be-opened.pt"
        common = {
            "format": DIRECT_SLAT_CACHE_VERSION,
            "uid": uid,
            "object_uid": object_uid,
            "config_hash": "config-hash",
        }
        torch.save(
            {
                **common,
                "seed": 42,
                "corrected_ss": torch.zeros(1, 8, 16, 16, 16),
                "occupancy_logits64": torch.zeros(1, 1, 64, 64, 64),
                "corrected_coords64": torch.zeros(1, 3, dtype=torch.int32),
            },
            support_path,
        )
        torch.save({**common, "unused_native_placeholder": True}, physical_path)
        dataset = object.__new__(DirectSLatWithoutLegacyCondition)
        dataset.rows = [
            {
                "uid": uid,
                "object_uid": object_uid,
                "support_seed": 42,
                "target_file": str(target),
                "support_file": str(support_path),
                "physical_file": str(physical_path),
                "condition_file": str(condition_path),
            }
        ]
        dataset.root = root
        dataset.config = {"condition_arch": "native_ss_genrecon_v2"}
        dataset.config_hash = "config-hash"
        dataset.check_tensor_finite = finite
        return dataset, condition_path

    def test_skips_legacy_condition_and_keeps_training_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, forbidden = self._dataset(Path(temporary))
            loaded: list[Path] = []
            real_load = torch.load

            def recording_load(path, *args, **kwargs):
                loaded.append(Path(path))
                self.assertIs(kwargs.get("weights_only"), False)
                return real_load(path, *args, **kwargs)

            with mock.patch.object(torch, "load", side_effect=recording_load):
                sample = dataset[0]
        self.assertNotIn(forbidden, loaded)
        self.assertNotIn("condition", sample)
        self.assertFalse(sample["legacy_condition_file_loaded"])
        self.assertEqual(sample["runtime_condition_source"], "with_vggt_sidecar_only")
        self.assertEqual(tuple(sample["target_feats"].shape), (2, 8))

    def test_nonfinite_scientific_tensor_remains_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = self._dataset(root)
            target = Path(dataset.rows[0]["target_file"])
            feats = np.ones((2, 8), dtype=np.float32)
            feats[0, 0] = np.nan
            np.savez(
                target,
                coords=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
                feats=feats,
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                dataset[0]


class RuntimeCompositionTest(unittest.TestCase):
    def test_configure_runtime_installs_only_performance_bindings(self) -> None:
        strict_train, strict_projection, _ = runtime.load_strict_modules()
        configured, identity = train.configure_runtime()
        self.assertIs(configured, strict_train)
        self.assertEqual(identity["version"], runtime.RUNTIME_VERSION)
        self.assertIs(
            strict_train.NativeConditionSLatDataset,
            StrictWithVGGTNativeConditionSLatDataset,
        )
        self.assertIs(
            train._v2_model.project_sparse_frustum_dino,
            strict_projection.project_sparse_frustum_dino,
        )
        self.assertIs(train._science_arm._train, strict_train)

    def test_audited_cache_profile_is_deliberately_forbidden(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["trainer", "--skip_redundant_cache_finite_checks"],
        ):
            with self.assertRaisesRegex(ValueError, "retains per-sample finite"):
                train.main()

    def test_official_entry_installs_official_decoder_validator(self) -> None:
        with mock.patch.object(
            train_proobjaverse_official._arm, "main"
        ) as arm_main:
            train_proobjaverse_official.main()
        arm_main.assert_called_once_with(
            decoder_validator=(
                train_proobjaverse_official.validate_official_decoder_audit
            )
        )


if __name__ == "__main__":
    unittest.main()
