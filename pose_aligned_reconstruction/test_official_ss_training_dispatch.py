from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pose_aligned_reconstruction.proobjaverse_official_ss_compact import (
    OFFICIAL_SS_COMPACT_MANIFEST_FORMAT,
)
from pose_aligned_reconstruction.train_proobjaverse_official_native_ss_no_vggt import (
    load_official_training_dataset,
)


class OfficialSSTrainingDispatchTest(unittest.TestCase):
    def _manifest(self, payload: dict[str, object]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_compact_paper_manifest_uses_local_compact_adapter(self) -> None:
        path = self._manifest({"format": OFFICIAL_SS_COMPACT_MANIFEST_FORMAT})
        sentinel = object()
        with mock.patch(
            "pose_aligned_reconstruction.proobjaverse_official_ss_compact."
            "CompactOfficialSSDataset",
            return_value=sentinel,
        ) as loader:
            observed = load_official_training_dataset(str(path), indices="2-4")
        self.assertIs(observed, sentinel)
        loader.assert_called_once_with(str(path), indices="2-4")

    def test_legacy_manifest_keeps_shared_loader(self) -> None:
        path = self._manifest({"format": "legacy-format"})
        sentinel = object()
        with mock.patch(
            "ar_ss_flow.local_pose_lifting_flow.PoseLiftingCacheDataset",
            return_value=sentinel,
        ) as loader:
            observed = load_official_training_dataset(str(path), indices="all")
        self.assertIs(observed, sentinel)
        loader.assert_called_once_with(str(path), indices="all")


if __name__ == "__main__":
    unittest.main()
