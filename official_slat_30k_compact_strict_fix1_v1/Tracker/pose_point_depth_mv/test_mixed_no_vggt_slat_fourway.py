from __future__ import annotations

import unittest

import torch
from torch import nn

from pose_point_depth_mv.evaluate_mixed_no_vggt_slat_fourway import (
    decode_mesh_fp32,
    paired_improvements,
    select_balanced_objects,
    summarize_records,
    zero_every_block_condition,
)
from pose_point_depth_mv.summarize_mixed_slat_stock_context_2x2 import (
    report_context_mode,
    summarize_2x2,
)


class MixedNoVGGTSlatFourwayTests(unittest.TestCase):
    @staticmethod
    def _two_by_two_report(mode: str, correct_chamfer: float) -> dict:
        records = []
        for domain in ("synthetic", "real"):
            for branch, chamfer in (
                ("lora_only", 0.20),
                ("correct", correct_chamfer),
            ):
                records.append(
                    {
                        "domain": domain,
                        "object_uid": f"{domain}-object",
                        "branch": branch,
                        "surface": {
                            "chamfer_l1": chamfer,
                            "chamfer_l2": chamfer**2,
                            "fscore_0p01": 1.0 - chamfer,
                            "fscore_0p02": 1.0 - chamfer,
                            "fscore_0p05": 1.0 - chamfer,
                            "normal_consistency": 1.0 - chamfer,
                        },
                        "structure": {
                            "largest_component_ratio": 1.0 - chamfer,
                            "component_count": 2 if branch == "lora_only" else 1,
                        },
                    }
                )
        return {
            "passed": True,
            "formal": False,
            "training_overlap": True,
            "object_count": 2,
            "objects_per_domain": 1,
            "same_native_ss_coordinates": True,
            "same_initial_noise": True,
            "sampling": {"steps": 25},
            "stock_context_views": mode,
            "run_config": {
                "checkpoint_step": 400,
                "cache_identity": {"hash": "same"},
                "selection": {"rows": ["same"]},
                "noise_seed": 42,
                "surface_samples": 20000,
            },
            "records": records,
        }

    def test_2x2_interaction_detects_larger_first_view_posed_gain(self) -> None:
        all_report = self._two_by_two_report("all", correct_chamfer=0.18)
        first_report = self._two_by_two_report("first", correct_chamfer=0.10)
        result = summarize_2x2(
            all_report, first_report, bootstrap_samples=20
        )
        interaction = result["mixed_macro_1to1"][
            "interaction_first_minus_all"
        ]["metrics"]["chamfer_l1"]
        self.assertAlmostEqual(interaction["mean"], 0.08)
        self.assertEqual(report_context_mode({"run_config": {}}), "all")

    def test_2x2_rejects_mismatched_training_steps(self) -> None:
        all_report = self._two_by_two_report("all", correct_chamfer=0.18)
        first_report = self._two_by_two_report("first", correct_chamfer=0.10)
        first_report["run_config"]["checkpoint_step"] = 800
        with self.assertRaisesRegex(RuntimeError, "checkpoint_step"):
            summarize_2x2(all_report, first_report, bootstrap_samples=20)

    def test_mesh_decode_forces_fp32_and_disables_autocast(self) -> None:
        class Decoder(nn.Module):
            def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
                self.seen_dtype = value.dtype
                self.saw_autocast = torch.is_autocast_enabled("cpu")
                return (value + 1.0,)

        decoder = Decoder()
        latent = torch.ones(2, 3, dtype=torch.bfloat16)
        mean = torch.zeros(1, dtype=torch.float32)
        std = torch.ones(1, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            decoded = decode_mesh_fp32(decoder, latent, std=std, mean=mean)
        self.assertEqual(decoder.seen_dtype, torch.float32)
        self.assertFalse(decoder.saw_autocast)
        self.assertEqual(decoded.dtype, torch.float32)

    def test_balanced_selection_is_deterministic_and_prefers_more_views(self) -> None:
        rows = []
        for domain in ("synthetic", "real"):
            for object_index in range(4):
                for views in (2, 8):
                    rows.append(
                        {
                            "_mixed_domain": domain,
                            "object_uid": f"{domain}-{object_index}",
                            "uid": f"{domain}-{object_index}-v{views}",
                            "support_seed": 42,
                            "view_count": views,
                        }
                    )
        first = select_balanced_objects(rows, objects_per_domain=3, seed=7)
        second = select_balanced_objects(rows, objects_per_domain=3, seed=7)
        self.assertEqual(first, second)
        self.assertEqual([row["domain"] for row in first], ["synthetic", "real"] * 3)
        self.assertTrue(all(row["view_count"] == 8 for row in first))
        self.assertEqual(len({row["object_uid"] for row in first}), 6)

    def test_zero_hook_removes_weight_and_bias_then_restores_module(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block_condition = nn.Linear(2, 3)

        model = Model()
        nn.init.constant_(model.block_condition.weight, 2.0)
        nn.init.constant_(model.block_condition.bias, 5.0)
        inputs = torch.ones(4, 2)
        original = model.block_condition(inputs)
        with zero_every_block_condition(model):
            suppressed = model.block_condition(inputs)
        restored = model.block_condition(inputs)
        self.assertEqual(int(torch.count_nonzero(suppressed)), 0)
        torch.testing.assert_close(restored, original)

    def test_paired_metrics_use_positive_is_left_better(self) -> None:
        left = {
            "surface": {
                "chamfer_l1": 0.1,
                "chamfer_l2": 0.01,
                "fscore_0p01": 0.3,
                "fscore_0p02": 0.5,
                "fscore_0p05": 0.7,
                "normal_consistency": 0.8,
            },
            "structure": {"largest_component_ratio": 0.9, "component_count": 2},
        }
        right = {
            "surface": {
                "chamfer_l1": 0.2,
                "chamfer_l2": 0.04,
                "fscore_0p01": 0.2,
                "fscore_0p02": 0.4,
                "fscore_0p05": 0.6,
                "normal_consistency": 0.7,
            },
            "structure": {"largest_component_ratio": 0.8, "component_count": 4},
        }
        deltas = paired_improvements(left, right)
        self.assertAlmostEqual(deltas["chamfer_l1"], 0.1)
        self.assertAlmostEqual(deltas["fscore_0p02"], 0.1)
        self.assertAlmostEqual(deltas["normal_consistency"], 0.1)
        self.assertAlmostEqual(deltas["largest_component_ratio"], 0.1)
        self.assertAlmostEqual(deltas["component_count"], 2.0)

    def test_summary_keeps_domains_and_one_to_one_macro_separate(self) -> None:
        records = []
        for domain in ("synthetic", "real"):
            for branch_index, branch in enumerate(
                ("stock", "lora_only", "pose_cyclic1", "correct")
            ):
                records.append(
                    {
                        "domain": domain,
                        "object_uid": f"{domain}-object",
                        "branch": branch,
                        "surface": {
                            "chamfer_l1": 0.4 - branch_index * 0.05,
                            "chamfer_l2": 0.2 - branch_index * 0.02,
                            "fscore_0p01": 0.1 + branch_index * 0.02,
                            "fscore_0p02": 0.2 + branch_index * 0.03,
                            "fscore_0p05": 0.3 + branch_index * 0.04,
                            "normal_consistency": 0.5 + branch_index * 0.05,
                        },
                        "structure": {
                            "largest_component_ratio": 0.7 + branch_index * 0.05,
                            "component_count": 8 - branch_index,
                        },
                    }
                )
        summary = summarize_records(records, bootstrap_samples=20)
        self.assertEqual(
            set(summary["comparisons"]),
            {"synthetic", "real", "mixed_macro_1to1"},
        )
        pose = summary["comparisons"]["mixed_macro_1to1"][
            "correct_pose_specificity"
        ]["metrics"]
        self.assertGreater(pose["chamfer_l1"]["mean"], 0.0)
        self.assertGreater(pose["fscore_0p02"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
