#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
import json
from unittest import mock

import torch

from pose_point_depth_mv.native_slat_condition_only import (
    NATIVE_SLAT_BASELINE,
    NATIVE_SLAT_CONDITION_ONLY_CFG,
    NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
    NATIVE_SLAT_CONDITION_ONLY_TRAINING,
)
from pose_point_depth_mv.native_slat_condition_only_no_vggt import (
    NO_VGGT_CONDITION_ONLY_CONTRACT,
)
from pose_point_depth_mv.native_slat_condition_only_objective_v2 import (
    NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
    validate_native_slat_condition_only_objective_v2_checkpoint,
)
from pose_point_depth_mv.native_slat_decoder_geometry import (
    DECODER_GEOMETRY_LOSS_VERSION,
    decoder_field_distance,
    stock_relative_trust_loss,
)
from pose_point_depth_mv.train_native_slat_condition_only import (
    deterministic_geometry_event,
    distributed_timestep_bins,
    sample_condition_only_t,
)
from pose_point_depth_mv import summarize_native_slat_objective_v2_ablation as summary_cli


class NativeSLatObjectiveV2Tests(unittest.TestCase):
    def _checkpoint(self) -> dict:
        upstream = {"report_sha256": "ss"}
        arguments = {
            "t_schedule": "uniform",
            "t_uniform_probability": 0.75,
            "separate_t_rng": True,
            "decoder_geometry_weight": 0.10,
            "stock_flow_trust_weight": 0.01,
            "stock_flow_required_improvement": 0.01,
            "stock_trust_weight": 0.05,
            "stock_required_improvement": 0.01,
            "geometry_event_probability": 0.125,
            "geometry_t_max": 0.5,
        }
        return {
            "format": NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
            "step": 10,
            "args": arguments,
            "model_summary": {
                "format": NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
                "protocol_version": NATIVE_SLAT_CONDITION_ONLY_OBJECTIVE_V2_VERSION,
                "pretrained": "unit-test",
                "baseline": NATIVE_SLAT_BASELINE,
                "projection": NATIVE_SLAT_CONDITION_ONLY_PROJECTION,
                "training_semantics": NATIVE_SLAT_CONDITION_ONLY_TRAINING,
                "cfg_semantics": NATIVE_SLAT_CONDITION_ONLY_CFG,
                "condition_scale_policy": "learned_projection_only",
                "post_cfg_cap": False,
                "direct_slat_residual_dependency": False,
                "flow_lora": {
                    "present": False,
                    "module_count": 0,
                    "parameter_count": 0,
                    "construction": "PEFT is not imported or installed",
                },
                "context_view_fusion": {"trainable": False},
                "stock_slat_freeze": {"freeze_sha256": "stock"},
                "upstream_native_ss": upstream,
                "input_context_contract": NO_VGGT_CONDITION_ONLY_CONTRACT,
                "vggt_model_executed": False,
                "t_schedule": {
                    "name": "uniform",
                    "uniform_probability": 0.75,
                    "rng": "separate deterministic rank/micro-step generator",
                },
                "training_objective": {
                    "version": DECODER_GEOMETRY_LOSS_VERSION,
                    "decoder_geometry_weight": 0.10,
                    "stock_flow_trust_weight": 0.01,
                    "stock_flow_required_improvement": 0.01,
                    "stock_trust_weight": 0.05,
                    "stock_required_improvement": 0.01,
                    "geometry_event_probability": 0.125,
                    "geometry_t_max": 0.5,
                },
            },
            "model_trainable_state": {"aggregator.x": torch.ones(1)},
            "ema_trainable_state": {"aggregator.x": torch.ones(1)},
            "ema": {"updates": 10, "target_decay": 0.9995},
        }

    def test_decoder_field_distance_is_exactly_zero_at_target(self) -> None:
        target = {
            "sdf": torch.tensor([[-0.2], [0.0], [0.3]]),
            "deform": torch.tensor([[0.1, -0.1], [0.0, 0.2], [0.3, 0.4]]),
            "weights": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        }
        result = decoder_field_distance(target, target)
        self.assertEqual(float(result["total"]), 0.0)
        wrong = {name: value.clone() for name, value in target.items()}
        wrong["sdf"] = -wrong["sdf"] + 0.05
        self.assertGreater(float(decoder_field_distance(wrong, target)["total"]), 0.0)

    def test_stock_trust_is_relative_and_has_margin(self) -> None:
        loss, relative = stock_relative_trust_loss(
            torch.tensor(0.5), torch.tensor(0.5), required_improvement=0.01
        )
        self.assertAlmostEqual(float(relative), 0.0, places=7)
        self.assertAlmostEqual(float(loss), 0.01, places=7)
        better, relative = stock_relative_trust_loss(
            torch.tensor(0.4), torch.tensor(0.5), required_improvement=0.01
        )
        self.assertEqual(float(better), 0.0)
        self.assertLess(float(relative), -0.01)

    def test_uniform_timestep_and_bins(self) -> None:
        args = SimpleNamespace(
            t_schedule="uniform",
            t_uniform_probability=0.75,
            t_logit_mean=1.0,
            t_logit_std=1.0,
            separate_t_rng=True,
        )
        torch.manual_seed(7)
        state = torch.get_rng_state().clone()
        values = [
            sample_condition_only_t(
                args, torch.device("cpu"), seed=42, rank=0, micro_step=index
            )
            for index in range(8)
        ]
        self.assertTrue(all(0.0 <= value < 1.0 for value in values))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        bins = distributed_timestep_bins(
            [
                {"t": 0.1, "flow_loss": 1.0, "stock_loss": 2.0, "gain": 1.0},
                {"t": 0.9, "flow_loss": 3.0, "stock_loss": 2.0, "gain": -1.0},
            ],
            device=torch.device("cpu"),
            world_size=1,
        )
        self.assertEqual(bins[0]["count"], 1.0)
        self.assertEqual(bins[4]["count"], 1.0)

    def test_geometry_event_is_resume_stable(self) -> None:
        first = [
            deterministic_geometry_event(
                seed=42, rank=1, micro_step=step, probability=0.25
            )
            for step in range(20)
        ]
        second = [
            deterministic_geometry_event(
                seed=42, rank=1, micro_step=step, probability=0.25
            )
            for step in range(20)
        ]
        self.assertEqual(first, second)
        self.assertGreater(sum(first), 0)
        self.assertLess(sum(first), len(first))

    def test_checkpoint_binds_schedule_and_geometry(self) -> None:
        checkpoint = self._checkpoint()
        validate_native_slat_condition_only_objective_v2_checkpoint(
            checkpoint,
            pretrained="unit-test",
            stock_slat_freeze={"freeze_sha256": "stock"},
            upstream_native_ss={"report_sha256": "ss"},
        )
        checkpoint["model_summary"]["training_objective"][
            "decoder_geometry_weight"
        ] = 0.2
        with self.assertRaisesRegex(ValueError, "decoder_geometry_weight"):
            validate_native_slat_condition_only_objective_v2_checkpoint(
                checkpoint,
                pretrained="unit-test",
                stock_slat_freeze={"freeze_sha256": "stock"},
                upstream_native_ss={"report_sha256": "ss"},
            )

    def test_three_arm_summary_uses_paired_full_mesh(self) -> None:
        def report(chamfer: float, fscore: float) -> dict:
            rows = []
            for index in range(2):
                rows.append(
                    {
                        "object_uid": f"object-{index}",
                        "uid": f"uid-{index}",
                        "seed": 42,
                        "branches": {
                            "full": {
                                "surface": {
                                    "chamfer_l1": chamfer,
                                    "fscore_0p02": fscore,
                                    "normal_consistency": 0.5,
                                },
                                "structure": {"largest_component_ratio": 0.9},
                            }
                        },
                    }
                )
            stock_summary = {
                "chamfer_l1_improvement": {
                    "mean": 0.1,
                    "median": 0.1,
                    "positive_rate": 1.0,
                },
                "fscore_0p02_delta": {"mean": 0.1},
            }
            return {
                "format": summary_cli.EXPECTED_REPORT,
                "passed": True,
                "object_count": 2,
                "record_count": 2,
                "run_config": {"checkpoint": f"checkpoint-{chamfer}"},
                "summary": stock_summary,
                "records": rows,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, value in (
                ("logit", report(0.4, 0.4)),
                ("uniform", report(0.3, 0.5)),
                ("geometry", report(0.2, 0.6)),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            output = root / "summary.json"
            argv = [
                "summary",
                "--logit_report",
                str(paths[0]),
                "--uniform_report",
                str(paths[1]),
                "--geometry_report",
                str(paths[2]),
                "--output",
                str(output),
                "--expected_objects",
                "2",
                "--bootstrap_samples",
                "20",
            ]
            with mock.patch("sys.argv", argv):
                summary_cli.main()
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(
                value["decision"]["geometry_objective_increment_gate"]
            )


if __name__ == "__main__":
    unittest.main()
