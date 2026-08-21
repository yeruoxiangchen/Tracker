from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from official_ss_with_vggt_perf_v1.reuse_partial_endpoint_ss_coords import (
    SOURCE_FORMAT,
    TARGET_FORMAT,
    reuse_partial_ss_coords,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_id(uid: str, seed: int) -> str:
    return hashlib.sha256(f"{uid}|{seed}".encode()).hexdigest()[:24]


def _write_pair(root: Path, uid: str, seed: int, *, offset: int = 0) -> None:
    coords = root / "ss_coords"
    coords.mkdir(parents=True, exist_ok=True)
    stem = _pair_id(uid, seed)
    npz = coords / f"{stem}.npz"
    np.savez_compressed(
        npz,
        stock=np.asarray([[0, offset, 0, 0]], dtype=np.int32),
        native=np.asarray([[0, offset + 1, 0, 0]], dtype=np.int32),
    )
    audit = {
        "object_uid": uid,
        "seed": seed,
        "same_initial_noise": True,
        "stock_count": 1,
        "native_count": 1,
        "passed": True,
        "coords_npz_sha256": _sha(npz),
    }
    (coords / f"{stem}.json").write_text(json.dumps(audit), encoding="utf-8")


class PartialSSCoordinateReuseTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source, target = root / "source", root / "target"
        source.mkdir()
        target.mkdir()
        stable = {
            "object_start": 16,
            "object_end": 18,
            "object_uids": ["a", "b"],
            "joint_seeds": [42, 43],
            "trained_slat_checkpoint_sha256": "c" * 64,
            "native_ss_report_sha256": "d" * 64,
        }
        (source / "run_identity.json").write_text(
            json.dumps({"format": SOURCE_FORMAT, **stable}), encoding="utf-8"
        )
        (target / "run_identity.json").write_text(
            json.dumps({"format": TARGET_FORMAT, **stable}), encoding="utf-8"
        )
        _write_pair(source, "a", 42)
        _write_pair(source, "a", 43)
        return source, target

    def test_links_missing_pairs_and_keeps_valid_target_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, target = self._fixture(Path(temporary))
            _write_pair(target, "a", 42, offset=10)
            result = reuse_partial_ss_coords(source, target)
            linked_stem = _pair_id("a", 43)
            self.assertEqual(result["linked_pair_count"], 1)
            self.assertEqual(result["kept_complete_target_pair_count"], 1)
            self.assertEqual(
                (source / "ss_coords" / f"{linked_stem}.npz").stat().st_ino,
                (target / "ss_coords" / f"{linked_stem}.npz").stat().st_ino,
            )

    def test_identity_mismatch_fails_before_linking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, target = self._fixture(Path(temporary))
            identity = json.loads((target / "run_identity.json").read_text())
            identity["joint_seeds"] = [42]
            (target / "run_identity.json").write_text(json.dumps(identity))
            with self.assertRaisesRegex(RuntimeError, "run identities differ"):
                reuse_partial_ss_coords(source, target)

    def test_tampered_npz_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, target = self._fixture(Path(temporary))
            next((source / "ss_coords").glob("*.npz")).write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                reuse_partial_ss_coords(source, target)


if __name__ == "__main__":
    unittest.main()
