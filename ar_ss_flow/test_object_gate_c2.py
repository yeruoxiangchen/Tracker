from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ar_ss_flow.object_gate_c2 import (
    HYPOTHESES,
    SelfReferenceObjectGateTable,
    apply_object_gate_exact,
    deterministic_permuted_gates,
    stable_sigmoid,
)


class ObjectGateC2Test(unittest.TestCase):
    def test_sigmoid_is_monotonic(self) -> None:
        values = stable_sigmoid(np.asarray([-1.0, 0.0, 1.0]), 0.5)
        self.assertTrue(np.all(np.diff(values) > 0.0))
        self.assertAlmostEqual(float(values[1]), 0.5)

    def test_exact_zero_gate_returns_same_tensor(self) -> None:
        stock = torch.randn(1, 8, 2, 2, 2)
        delta = torch.randn_like(stock)
        output, applied = apply_object_gate_exact(stock, delta, 0.0)
        self.assertEqual(output.data_ptr(), stock.data_ptr())
        self.assertEqual(float(applied.abs().max()), 0.0)

    def test_gate_scales_delta(self) -> None:
        stock = torch.zeros(1, 1, 1, 1, 1)
        delta = torch.ones_like(stock)
        output, applied = apply_object_gate_exact(stock, delta, 0.25)
        self.assertAlmostEqual(float(output.item()), 0.25)
        self.assertAlmostEqual(float(applied.item()), 0.25)

    def test_permutation_preserves_multiset(self) -> None:
        gates = {"a": 0.1, "b": 0.2, "c": 0.8}
        output = deterministic_permuted_gates(gates, gates, seed=42)
        self.assertEqual(sorted(output.values()), sorted(gates.values()))
        self.assertNotEqual(output, gates)

    def test_load_gate_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "protocol": {
                    "hypotheses": list(HYPOTHESES),
                    "visual_only_pairwise": True,
                    "geometry_pair_scale_forced_zero": True,
                },
                "object_uids": {"0": "object-a"},
            }
            calibration = {
                "config": {
                    "name": "mean__ref_mean",
                    "statistic": {"name": "mean", "kind": "mean", "fraction": None},
                    "reference_reducer": "mean",
                },
                "temperature": 0.1,
                "minimum_valid_voxels": 1,
            }
            confidence = np.zeros((1, len(HYPOTHESES), 4, 8), dtype=np.float32)
            confidence[:, 0, 0] = 0.8
            confidence[:, 0, 1:] = 0.4
            for index in range(1, len(HYPOTHESES)):
                confidence[:, index, 0] = 0.3
                confidence[:, index, 1:] = 0.5
            support = np.ones((1, len(HYPOTHESES), 8), dtype=np.uint8)
            np.savez_compressed(
                root / "samples.npz",
                selfref_object_index=np.asarray([0], dtype=np.int32),
                selfref_confidence=confidence,
                selfref_common_support=support,
            )
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "calibration.json").write_text(
                json.dumps(calibration), encoding="utf-8"
            )
            table = SelfReferenceObjectGateTable.load(
                report_path=root / "report.json",
                samples_path=root / "samples.npz",
                calibration_path=root / "calibration.json",
            )
            self.assertIn("object-a", table)
            self.assertGreater(
                table.gate("object-a", "correct"),
                table.gate("object-a", "pose_cyclic1"),
            )


if __name__ == "__main__":
    unittest.main()
