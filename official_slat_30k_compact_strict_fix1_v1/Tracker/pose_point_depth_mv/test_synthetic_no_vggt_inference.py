from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic import (
    EXPECTED_OBJECT_COUNT,
    EXPECTED_SAMPLE_COUNT,
    validate_synthetic_deployment,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


class SyntheticNoVggtInferenceTest(unittest.TestCase):
    def _headers(self, ss_path: Path) -> tuple[dict, dict]:
        identity = {
            "manifest": "/frozen/synthetic.json",
            "manifest_sha256": "manifest-sha",
            "config_hash": "config-sha",
            "object_count": EXPECTED_OBJECT_COUNT,
            "sample_count": EXPECTED_SAMPLE_COUNT,
        }
        ss = {
            "format": NATIVE_SS_NO_VGGT_VERSION,
            "data_identity": identity,
            "model_summary": {},
        }
        slat = {
            "format": NATIVE_SLAT_NO_VGGT_VERSION,
            "data_identity": identity,
            "model_summary": {
                "upstream_native_ss": {
                    "checkpoint_sha256": sha256_file(ss_path),
                    "checkpoint_step": 2000,
                    "weights": "ema",
                }
            },
        }
        return ss, slat

    def test_synthetic_profile_validates_direct_lineage_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ss_path = root / "ss.pt"
            slat_path = root / "slat.pt"
            freeze_path = root / "freeze.json"
            ss_path.write_bytes(b"ss")
            slat_path.write_bytes(b"slat")
            freeze_path.write_text("{}", encoding="utf-8")
            ss, slat = self._headers(ss_path)
            with (
                mock.patch(
                    "pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic.torch.load",
                    side_effect=[ss, slat],
                ),
                mock.patch(
                    "pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic.load_stock_slat_freeze",
                    return_value={},
                ),
                mock.patch(
                    "pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic.validate_native_ss_no_vggt_checkpoint"
                ) as validate_ss,
                mock.patch(
                    "pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic.validate_native_slat_no_vggt_checkpoint"
                ) as validate_slat,
            ):
                upstream, lineage = validate_synthetic_deployment(
                    pretrained="Stable-X/trellis-vggt-v0-2",
                    ss_path=ss_path,
                    slat_path=slat_path,
                    stock_freeze_path=freeze_path,
                )
            self.assertEqual(upstream["weights"], "ema")
            self.assertEqual(lineage["object_count"], EXPECTED_OBJECT_COUNT)
            self.assertEqual(lineage["sequence_count"], EXPECTED_SAMPLE_COUNT)
            self.assertFalse(lineage["real_full_migration_contract_consumed"])
            validate_ss.assert_called_once()
            validate_slat.assert_called_once()

    def test_synthetic_profile_rejects_real_migration_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ss_path = root / "ss.pt"
            slat_path = root / "slat.pt"
            freeze_path = root / "freeze.json"
            ss_path.write_bytes(b"ss")
            slat_path.write_bytes(b"slat")
            freeze_path.write_text("{}", encoding="utf-8")
            ss, slat = self._headers(ss_path)
            ss["model_summary"]["migration_contract"] = {"stage": "ss"}
            with mock.patch(
                "pose_point_depth_mv.infer_omni_real_native_no_vggt_synthetic.torch.load",
                side_effect=[ss, slat],
            ):
                with self.assertRaisesRegex(RuntimeError, "real migration contract"):
                    validate_synthetic_deployment(
                        pretrained="Stable-X/trellis-vggt-v0-2",
                        ss_path=ss_path,
                        slat_path=slat_path,
                        stock_freeze_path=freeze_path,
                    )


if __name__ == "__main__":
    unittest.main()
