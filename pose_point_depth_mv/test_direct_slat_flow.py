from __future__ import annotations

import argparse
import unittest

import torch

from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_TRAINING_SEMANTICS_V3,
    DIRECT_SLAT_TRAINING_SEMANTICS_V4,
    DIRECT_SLAT_TRAINING_SEMANTICS_V5,
    SLAT_DELTA_BOUND_SMOOTH,
    SLAT_ROLLOUT_COMPONENT_ADAPTER_ONLY,
    SLAT_ROLLOUT_COMPONENT_LORA_ONLY,
    SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
    SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY,
    SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
    SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET,
    DirectSupportSLATFlowModel,
    PostCFGSupportSLATRolloutFlow,
    SS2SLATSupportAdapter,
    bounded_slat_flow_delta,
    combine_slat_lora_support_budgets,
    correct_over_wrong_support_rank_loss,
    detached_sparse_euler_step,
    native_flow_timestep_sequence,
    deterministic_probability_event,
    deterministic_probability_partition,
    deterministic_wrong_support_index,
    gather_slat_support_evidence,
    resolve_slat_guided_delta_policy,
    resolve_slat_delta_bound_mode,
    resolve_slat_support_interval_policy,
    stock_relative_residual_excess_loss,
    straight_through_exact_reference,
    validate_sparse_target_alignment,
)
from pose_point_depth_mv.eval_direct_slat_flow import select_evaluation_indices
from pose_point_depth_mv.train_direct_slat_flow import validate_args


class _FakeSparse:
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor):
        self.feats = feats
        self.coords = coords

    def replace(self, feats: torch.Tensor):
        return _FakeSparse(feats, self.coords)


class _FakeGuidedModel:
    def __init__(self):
        self.last_rollout_component = None
        self.grad_enabled_calls = []

    def stock_prediction(self, x, t, condition):
        del t
        value = 2.0 if condition == "positive" else 1.0
        return x.replace(torch.full_like(x.feats, value))

    def conditioned_prediction(self, x, t, condition, **kwargs):
        del t, condition, kwargs
        raw = x.replace(torch.full_like(x.feats, 4.0))
        return raw, {"raw_flow_delta_rms": torch.tensor(2.0)}

    def post_cfg_conditioned_prediction(
        self, x, t, positive_condition, negative_condition, **kwargs
    ):
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        self.last_rollout_component = kwargs.get("rollout_component")
        stock_positive = self.stock_prediction(x, t, positive_condition)
        cfg_active = bool(kwargs["cfg_active"])
        support_active = bool(kwargs.get("support_active", True))
        if cfg_active:
            stock_negative = self.stock_prediction(x, t, negative_condition)
            stock = stock_positive.replace(
                float(kwargs["cfg_strength"]) * stock_positive.feats
                + (1.0 - float(kwargs["cfg_strength"])) * stock_negative.feats
            )
        else:
            stock_negative = None
            stock = stock_positive
        if not support_active:
            zero = stock.feats.new_zeros(())
            return stock, stock, {
                "stock_velocity_rms": stock.feats.square().mean().sqrt(),
                "raw_flow_delta_rms": zero,
                "effective_flow_delta_rms": zero,
                "delta_clip_scale": zero.new_tensor(1.0),
                "delta_clip_activated": zero,
                "raw_flow_delta_abs_max": zero,
                "effective_flow_delta_abs_max": zero,
                "positive_raw_flow_delta_rms": zero,
            }
        raw_positive, _ = self.conditioned_prediction(
            x, t, positive_condition
        )
        raw = (
            raw_positive.replace(
                float(kwargs["cfg_strength"]) * raw_positive.feats
                + (1.0 - float(kwargs["cfg_strength"])) * stock_negative.feats
            )
            if stock_negative is not None
            else raw_positive
        )
        effective, stats = bounded_slat_flow_delta(
            stock.feats,
            raw.feats,
            raw.coords,
            delta_scale=float(kwargs.get("slat_delta_scale", 1.0)),
            delta_rms_ratio_cap=kwargs.get("slat_delta_rms_ratio_cap"),
            bound_mode=kwargs.get("slat_delta_bound_mode", "hard_clip_v1"),
        )
        stats["positive_raw_flow_delta_rms"] = torch.tensor(2.0)
        return raw.replace(effective), stock, stats


class DirectSLatCoordinateTests(unittest.TestCase):
    def test_asymmetric_xyz_batch_mapping(self) -> None:
        corrected = torch.zeros((2, 8, 16, 16, 16), dtype=torch.float32)
        occupancy = torch.zeros((2, 1, 64, 64, 64), dtype=torch.float32)
        physical = torch.zeros((2, 16**3, 4), dtype=torch.float32)
        for batch in range(2):
            for x, y, z in ((0, 1, 2), (3, 5, 7), (15, 14, 13)):
                corrected[batch, :, x, y, z] = torch.arange(8) + (
                    batch * 10000 + x * 100 + y * 10 + z
                )
            physical[batch, :, 0] = torch.arange(16**3) + batch * 10000
        grid = torch.arange(64, dtype=torch.float32)
        occupancy[:, 0] = (
            grid[:, None, None] * 10000
            + grid[None, :, None] * 100
            + grid[None, None, :]
        )
        occupancy[1] += 1_000_000
        coords32 = torch.tensor(
            [
                [0, 1, 2, 4],
                [0, 7, 11, 15],
                [1, 31, 29, 27],
            ],
            dtype=torch.int32,
        )
        evidence, audit = gather_slat_support_evidence(
            coords32, corrected, occupancy, physical
        )
        expected16 = coords32[:, 1:].long() // 2
        self.assertTrue(torch.equal(audit["xyz16"], expected16))
        expected_flat = (
            expected16[:, 0] * 256 + expected16[:, 1] * 16 + expected16[:, 2]
        )
        self.assertTrue(torch.equal(audit["flat16"], expected_flat))
        self.assertEqual(tuple(evidence.shape), (3, 8 + 2 + 4))
        self.assertEqual(float(evidence[0, 10].item()), float(expected_flat[0]))
        self.assertEqual(
            float(evidence[2, 10].item()), float(expected_flat[2] + 10000)
        )
        children = audit["children64"][0]
        expected_children = torch.tensor(
            [
                [2, 4, 8],
                [2, 4, 9],
                [2, 5, 8],
                [2, 5, 9],
                [3, 4, 8],
                [3, 4, 9],
                [3, 5, 8],
                [3, 5, 9],
            ]
        )
        self.assertTrue(torch.equal(children.cpu(), expected_children))

    def test_invalid_coordinates_and_duplicate_targets_fail(self) -> None:
        feats = torch.zeros((2, 8))
        duplicate = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_sparse_target_alignment(duplicate, feats)
        invalid = torch.tensor([[0, 0, 0, 64]])
        with self.assertRaisesRegex(ValueError, "leave"):
            validate_sparse_target_alignment(invalid, torch.zeros((1, 8)))


class DirectSLatAdapterTests(unittest.TestCase):
    def test_zero_initialised_output_is_exact_zero(self) -> None:
        torch.manual_seed(7)
        adapter = SS2SLATSupportAdapter(
            physical_channels=4, hidden_dim=16, flow_channels=12
        )
        evidence = torch.randn(11, 8 + 2 + 4)
        output = adapter(evidence)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_straight_through_anchor_is_exact_and_keeps_gradient(self) -> None:
        value = torch.tensor([0.007195, -3.0], requires_grad=True)
        reference = torch.tensor([1.25, 8.0])
        anchored = straight_through_exact_reference(value, reference)
        self.assertTrue(torch.equal(anchored, reference))
        anchored.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_v5_post_cfg_zero_init_and_missing_support_are_exact_stock(
        self,
    ) -> None:
        class ZeroInitToyModel(DirectSupportSLATFlowModel):
            def __init__(self) -> None:
                torch.nn.Module.__init__(self)
                self.flow = torch.nn.Module()
                self.flow.proj = torch.nn.Module()
                self.flow.proj.lora_B = torch.nn.Linear(
                    1, 1, bias=False
                )
                torch.nn.init.zeros_(self.flow.proj.lora_B.weight)
                self.support_adapter = torch.nn.Identity()

            def stock_prediction(self, x, t, cond):
                del t
                value = 2.0 if cond == "positive" else 1.0
                return x.replace(torch.full_like(x.feats, value))

            def conditioned_prediction(
                self,
                x,
                t,
                cond,
                *,
                stock_velocity=None,
                corrected_ss=None,
                occupancy_logits64=None,
                physical_tokens16=None,
                support_scale=1.0,
                **kwargs,
            ):
                del kwargs
                stock = (
                    self.stock_prediction(x, t, cond)
                    if stock_velocity is None
                    else stock_velocity
                )
                support_present = all(
                    torch.is_tensor(value)
                    for value in (
                        corrected_ss,
                        occupancy_logits64,
                        physical_tokens16,
                    )
                )
                zero = stock.feats.new_zeros(())
                stats = {
                    "raw_flow_delta_rms": zero,
                    "raw_flow_delta_abs_max": zero,
                    "zero_init_stock_anchor": zero.new_tensor(
                        float(support_present and float(support_scale) != 0.0)
                    ),
                }
                if support_present and float(support_scale) != 0.0:
                    raw_joint = (
                        stock.feats
                        + self.flow.proj.lora_B.weight.sum()
                    )
                    return stock.replace(
                        straight_through_exact_reference(
                            raw_joint,
                            stock.feats,
                        )
                    ), stats
                return stock, stats

            def lora_only_prediction(self, x, t, cond):
                stock = self.stock_prediction(x, t, cond)
                # Mimic the small numerical discrepancy observed between
                # lora_disabled and zero-LoRA BF16/CFG value paths while
                # preserving a live gradient to LoRA-B.
                discrepancy = (
                    self.flow.proj.lora_B.weight.sum()
                    + stock.feats.new_tensor(1.0e-4)
                )
                return stock.replace(stock.feats + discrepancy)

        model = ZeroInitToyModel()
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
        )
        x = _FakeSparse(torch.zeros((2, 1)), coords)
        t = torch.tensor([750.0])
        kwargs = {
            "stock_positive_velocity": model.stock_prediction(
                x, t, "positive"
            ),
            "stock_negative_velocity": model.stock_prediction(
                x, t, "negative"
            ),
            "cfg_strength": 5.0,
            "cfg_active": True,
            "slat_delta_scale": 1.0,
            "slat_delta_rms_ratio_cap": 0.10,
            "slat_delta_bound_mode": SLAT_DELTA_BOUND_SMOOTH,
            "slat_residual_combination_policy": (
                SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
            ),
            "slat_lora_delta_scale": 1.0,
            "slat_lora_delta_rms_ratio_cap": 0.03,
            "slat_support_delta_scale": 1.0,
            "slat_support_delta_rms_ratio_cap": 0.07,
        }
        missing, missing_stock, _ = model.post_cfg_conditioned_prediction(
            x,
            t,
            "positive",
            "negative",
            corrected_ss=None,
            occupancy_logits64=None,
            physical_tokens16=None,
            **kwargs,
        )
        self.assertTrue(torch.equal(missing.feats, missing_stock.feats))

        support = torch.zeros((1,))
        enabled, enabled_stock, stats = (
            model.post_cfg_conditioned_prediction(
                x,
                t,
                "positive",
                "negative",
                corrected_ss=support,
                occupancy_logits64=support,
                physical_tokens16=support,
                **kwargs,
            )
        )
        self.assertTrue(torch.equal(enabled.feats, enabled_stock.feats))
        self.assertEqual(
            float(stats["post_cfg_zero_init_stock_anchor"].item()),
            1.0,
        )
        enabled.feats.sum().backward()
        self.assertIsNotNone(model.flow.proj.lora_B.weight.grad)
        self.assertGreater(
            float(model.flow.proj.lora_B.weight.grad.abs().sum().item()),
            0.0,
        )


class DirectSLatBoundedDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coords = torch.tensor(
            [
                [2, 0, 0, 0],
                [2, 1, 0, 0],
                [7, 0, 1, 0],
                [7, 0, 0, 1],
            ],
            dtype=torch.int32,
        )
        self.stock = torch.tensor(
            [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]
        )
        self.raw = torch.tensor(
            [[6.0, 6.0], [6.0, 6.0], [1.25, 1.25], [1.25, 1.25]]
        )

    def test_legacy_policy_returns_raw_full_exactly(self) -> None:
        effective, stats = bounded_slat_flow_delta(
            self.stock,
            self.raw,
            self.coords,
        )
        self.assertIs(effective, self.raw)
        self.assertEqual(float(stats["delta_clip_activated"].item()), 0.0)
        self.assertEqual(float(stats["delta_clip_scale"].item()), 1.0)
        self.assertTrue(
            torch.equal(
                stats["raw_flow_delta_rms"],
                stats["effective_flow_delta_rms"],
            )
        )

    def test_cap_is_applied_independently_per_sparse_batch(self) -> None:
        effective, stats = bounded_slat_flow_delta(
            self.stock,
            self.raw,
            self.coords,
            delta_scale=0.5,
            delta_rms_ratio_cap=0.5,
        )
        expected = torch.tensor(
            [[2.5, 2.5], [2.5, 2.5], [1.125, 1.125], [1.125, 1.125]]
        )
        self.assertTrue(torch.allclose(effective, expected))
        self.assertTrue(
            torch.allclose(
                stats["delta_clip_scale_per_batch"],
                torch.tensor([0.25, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                stats["delta_clip_activated_per_batch"],
                torch.tensor([1.0, 0.0]),
            )
        )
        self.assertTrue(torch.equal(stats["batch_ids"], torch.tensor([2, 7])))

    def test_zero_scale_is_exact_stock(self) -> None:
        effective, stats = bounded_slat_flow_delta(
            self.stock,
            self.raw,
            self.coords,
            delta_scale=0.0,
            delta_rms_ratio_cap=0.5,
        )
        self.assertIs(effective, self.stock)
        self.assertEqual(float(stats["effective_flow_delta_rms"].item()), 0.0)
        self.assertEqual(float(stats["effective_flow_delta_abs_max"].item()), 0.0)

    def test_unclipped_scale_keeps_gradient(self) -> None:
        raw = self.raw.clone().requires_grad_(True)
        effective, _ = bounded_slat_flow_delta(
            self.stock,
            raw,
            self.coords,
            delta_scale=0.5,
        )
        effective.sum().backward()
        self.assertTrue(torch.allclose(raw.grad, torch.full_like(raw, 0.5)))

    def test_clipped_scale_keeps_detached_scaled_gradient(self) -> None:
        raw = self.raw.clone().requires_grad_(True)
        effective, _ = bounded_slat_flow_delta(
            self.stock,
            raw,
            self.coords,
            delta_scale=0.5,
            delta_rms_ratio_cap=0.5,
        )
        effective.sum().backward()
        expected = torch.tensor(
            [[0.125, 0.125], [0.125, 0.125], [0.5, 0.5], [0.5, 0.5]]
        )
        self.assertTrue(torch.allclose(raw.grad, expected))

    def test_smooth_bound_is_bounded_and_keeps_radial_gradient(self) -> None:
        raw = self.raw.clone().requires_grad_(True)
        effective, stats = bounded_slat_flow_delta(
            self.stock,
            raw,
            self.coords,
            delta_scale=1.0,
            delta_rms_ratio_cap=0.5,
            bound_mode=SLAT_DELTA_BOUND_SMOOTH,
        )
        per_batch = stats["effective_flow_delta_rms_per_batch"]
        allowed = 0.5 * stats["stock_velocity_rms_per_batch"]
        self.assertTrue(bool(torch.all(per_batch <= allowed).item()))
        effective.sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all().item()))
        self.assertGreater(float(raw.grad[:2].abs().sum().item()), 0.0)
        self.assertFalse(
            torch.allclose(raw.grad[:2], torch.full_like(raw.grad[:2], 0.25))
        )

    def test_zero_raw_residual_excess_has_finite_zero_gradient(self) -> None:
        raw = self.stock.clone().requires_grad_(True)
        _, stats = bounded_slat_flow_delta(
            self.stock,
            raw,
            self.coords,
            delta_rms_ratio_cap=0.1,
        )
        loss, ratio, _ = stock_relative_residual_excess_loss(
            stock_rms_per_batch=stats["stock_velocity_rms_per_batch"],
            raw_delta_rms_per_batch=stats["raw_flow_delta_rms_per_batch"],
            ratio_cap=0.1,
        )
        loss.backward()
        self.assertEqual(float(loss.item()), 0.0)
        self.assertTrue(torch.equal(ratio, torch.zeros_like(ratio)))
        self.assertTrue(torch.equal(raw.grad, torch.zeros_like(raw)))
        self.assertTrue(bool(torch.isfinite(raw.grad).all().item()))

    def test_invalid_policy_and_coordinates_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bounded_slat_flow_delta(
                self.stock,
                self.raw,
                self.coords,
                delta_scale=-0.1,
            )
        bad_coords = self.coords.clone()
        bad_coords[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "negative batch"):
            bounded_slat_flow_delta(
                self.stock,
                self.raw,
                bad_coords,
            )


class DirectSLatV2SemanticsTests(unittest.TestCase):
    @staticmethod
    def _v2_args(**overrides):
        values = {
            "max_steps": 4,
            "save_every": 4,
            "log_every": 1,
            "grad_accum": 1,
            "grad_clip": 1.0,
            "amp_init_scale": 8192.0,
            "lora_rank": 16,
            "lora_alpha": 32,
            "adapter_hidden_dim": 128,
            "max_slat_points": 40960,
            "support_scale": 1.0,
            "slat_delta_scale": 1.0,
            "slat_delta_rms_ratio_cap": 0.1,
            "delta_norm_weight": 1.0e-4,
            "training_semantics": "bounded_mechanism_v2",
            "slat_guided_delta_policy": "post_cfg_v2",
            "slat_delta_bound_mode": "hard_clip_v1",
            "support_interval_policy": "all_steps_v1",
            "slat_residual_combination_policy": "joint_total_v1",
            "slat_lora_delta_scale": 1.0,
            "slat_lora_delta_rms_ratio_cap": -1.0,
            "slat_support_delta_scale": 1.0,
            "slat_support_delta_rms_ratio_cap": -1.0,
            "raw_delta_excess_weight": 0.01,
            "wrong_support_rank_weight": 0.1,
            "wrong_support_margin": 0.001,
            "wrong_support_probability": 0.5,
            "support_dropout_weight": 0.1,
            "support_dropout_probability": 0.25,
            "wrong_support_stock_weight": 0.0,
            "rollout_consistency_weight": 0.0,
            "rollout_probability": 0.0,
            "rollout_step_size": 0.0,
            "rollout_horizons": "1,2,4",
            "rollout_supervision_policy": (
                SLAT_ROLLOUT_SUPERVISION_ALL_VISITED
            ),
            "rollout_schedule_steps": 25,
            "rollout_rescale_t": 3.0,
            "endpoint_x0_weight": 0.0,
            "rollout_endpoint_rank_weight": 0.0,
            "rollout_endpoint_rank_margin": 0.0,
            "train_cfg_strength": 1.0,
            "train_cfg_interval": (0.5, 1.0),
            "t_schedule": "uniform",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_v2_training_arguments_require_finite_positive_mechanism(self) -> None:
        validate_args(self._v2_args())
        with self.assertRaisesRegex(ValueError, "positive residual cap"):
            validate_args(self._v2_args(slat_delta_rms_ratio_cap=0.0))
        with self.assertRaisesRegex(ValueError, "support_scale > 0"):
            validate_args(self._v2_args(support_scale=-1.0))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_args(self._v2_args(wrong_support_rank_weight=float("nan")))

    def test_v3_requires_post_cfg_rollout_and_wrong_to_stock(self) -> None:
        args = self._v2_args(
            training_semantics=DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            wrong_support_rank_weight=0.0,
            wrong_support_margin=0.0,
            wrong_support_stock_weight=0.25,
            rollout_consistency_weight=0.25,
            wrong_support_probability=0.25,
            support_dropout_probability=0.25,
            rollout_probability=0.25,
            rollout_step_size=0.05,
            train_cfg_strength=5.0,
        )
        validate_args(args)
        with self.assertRaisesRegex(ValueError, "wrong_support_stock_weight"):
            validate_args(
                self._v2_args(
                    training_semantics=DIRECT_SLAT_TRAINING_SEMANTICS_V3,
                    wrong_support_rank_weight=0.0,
                    wrong_support_margin=0.0,
                    wrong_support_stock_weight=0.0,
                    rollout_consistency_weight=0.25,
                    wrong_support_probability=0.25,
                    support_dropout_probability=0.25,
                    rollout_probability=0.25,
                    rollout_step_size=0.05,
                    train_cfg_strength=5.0,
                )
            )
        with self.assertRaisesRegex(ValueError, "train_cfg_strength > 1"):
            validate_args(
                self._v2_args(
                    training_semantics=DIRECT_SLAT_TRAINING_SEMANTICS_V3,
                    wrong_support_rank_weight=0.0,
                    wrong_support_margin=0.0,
                    wrong_support_stock_weight=0.25,
                    rollout_consistency_weight=0.25,
                    wrong_support_probability=0.25,
                    support_dropout_probability=0.25,
                    rollout_probability=0.25,
                    rollout_step_size=0.05,
                    train_cfg_strength=1.0,
                )
            )

    def test_rollout_policy_resolution_preserves_legacy_and_binds_v2(self) -> None:
        self.assertEqual(
            resolve_slat_guided_delta_policy({}),
            "positive_branch_v1",
        )
        self.assertEqual(
            resolve_slat_guided_delta_policy({}, "post_cfg_v2"),
            "post_cfg_v2",
        )
        saved_v2 = {
            "training_semantics": "bounded_mechanism_v2",
            "slat_guided_delta_policy": "post_cfg_v2",
        }
        self.assertEqual(resolve_slat_guided_delta_policy(saved_v2), "post_cfg_v2")
        with self.assertRaisesRegex(ValueError, "immutable"):
            resolve_slat_guided_delta_policy(saved_v2, "positive_branch_v1")
        saved_v3 = {
            "training_semantics": DIRECT_SLAT_TRAINING_SEMANTICS_V3,
            "slat_guided_delta_policy": "post_cfg_v2",
        }
        with self.assertRaisesRegex(ValueError, "immutable"):
            resolve_slat_guided_delta_policy(saved_v3, "positive_branch_v1")

        saved_v4 = {
            "training_semantics": DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            "slat_delta_bound_mode": SLAT_DELTA_BOUND_SMOOTH,
            "support_interval_policy": SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
        }
        self.assertEqual(
            resolve_slat_delta_bound_mode(saved_v4),
            SLAT_DELTA_BOUND_SMOOTH,
        )
        self.assertEqual(
            resolve_slat_support_interval_policy(saved_v4),
            SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            resolve_slat_delta_bound_mode(saved_v4, "hard_clip_v1")
        with self.assertRaisesRegex(ValueError, "immutable"):
            resolve_slat_support_interval_policy(saved_v4, "all_steps_v1")

    def test_detached_rollout_matches_native_euler_direction(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        x = _FakeSparse(torch.tensor([[3.0]], requires_grad=True), coords)
        velocity = _FakeSparse(torch.tensor([[4.0]], requires_grad=True), coords)
        previous, previous_t, applied = detached_sparse_euler_step(
            x,
            velocity,
            t_value=0.75,
            step_size=0.05,
        )
        self.assertTrue(torch.allclose(previous.feats, torch.tensor([[2.8]])))
        self.assertAlmostEqual(previous_t, 0.70)
        self.assertAlmostEqual(applied, 0.05)
        self.assertFalse(previous.feats.requires_grad)

    def test_v5_terminal_rollout_keeps_only_one_supervision_graph(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        x = _FakeSparse(torch.zeros((1, 1), requires_grad=True), coords)
        model = _FakeGuidedModel()
        _, stats, _, _ = DirectSupportSLATFlowModel.forward(
            model,
            x,
            torch.tensor([800.0]),
            "positive",
            corrected_ss=torch.tensor(0),
            occupancy_logits64=torch.tensor(0),
            physical_tokens16=torch.tensor(0),
            negative_condition="negative",
            post_cfg_strength=5.0,
            post_cfg_active=True,
            rollout_previous_t_values=(0.7, 0.6, 0.5),
            rollout_cfg_active_values=(True, True, True),
            rollout_support_active_values=(True, True, True),
            rollout_supervision_policy=(
                SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY
            ),
        )
        # One main-state graph, two graph-free trajectory calls, then one
        # terminal supervision graph.  The numerical horizon remains three.
        self.assertEqual(
            model.grad_enabled_calls,
            [True, False, False, True],
        )
        self.assertEqual(len(stats["_rollout_predictions"]), 1)
        self.assertEqual(len(stats["_rollout_states"]), 1)
        self.assertEqual(stats["_rollout_t_values"], (0.5,))
        self.assertEqual(
            stats["_rollout_generated_t_values"],
            (0.7, 0.6, 0.5),
        )
        self.assertEqual(int(stats["rollout_horizon"].item()), 3)
        self.assertEqual(
            int(stats["rollout_supervised_state_count"].item()),
            1,
        )

        legacy_model = _FakeGuidedModel()
        _, legacy_stats, _, _ = DirectSupportSLATFlowModel.forward(
            legacy_model,
            x,
            torch.tensor([800.0]),
            "positive",
            corrected_ss=torch.tensor(0),
            occupancy_logits64=torch.tensor(0),
            physical_tokens16=torch.tensor(0),
            negative_condition="negative",
            post_cfg_strength=5.0,
            post_cfg_active=True,
            rollout_previous_t_values=(0.7, 0.6, 0.5),
            rollout_cfg_active_values=(True, True, True),
            rollout_support_active_values=(True, True, True),
            rollout_supervision_policy=SLAT_ROLLOUT_SUPERVISION_ALL_VISITED,
        )
        self.assertEqual(
            legacy_model.grad_enabled_calls,
            [True, True, True, True],
        )
        self.assertEqual(len(legacy_stats["_rollout_predictions"]), 3)
        self.assertEqual(
            int(legacy_stats["rollout_supervised_state_count"].item()),
            3,
        )

    def test_v4_requires_native_schedule_smooth_bound_and_endpoint(self) -> None:
        args = self._v2_args(
            training_semantics=DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            slat_delta_bound_mode=SLAT_DELTA_BOUND_SMOOTH,
            support_interval_policy=SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
            t_schedule="native_schedule",
            wrong_support_rank_weight=0.0,
            wrong_support_margin=0.0,
            wrong_support_probability=0.2,
            wrong_support_stock_weight=0.25,
            support_dropout_probability=0.2,
            rollout_probability=0.4,
            rollout_consistency_weight=0.25,
            endpoint_x0_weight=0.1,
            train_cfg_strength=5.0,
        )
        validate_args(args)
        with self.assertRaisesRegex(ValueError, "smooth_rms_v2"):
            validate_args(
                argparse.Namespace(**{
                    **vars(args),
                    "slat_delta_bound_mode": "hard_clip_v1",
                })
            )
        schedule = native_flow_timestep_sequence(steps=25, rescale_t=3.0)
        self.assertEqual(len(schedule), 26)
        self.assertEqual(schedule[0], 1.0)
        self.assertEqual(schedule[-1], 0.0)
        self.assertTrue(
            all(left > right for left, right in zip(schedule, schedule[1:]))
        )

    def test_v5_requires_branch_budgets_and_endpoint_rank(self) -> None:
        args = self._v2_args(
            training_semantics=DIRECT_SLAT_TRAINING_SEMANTICS_V5,
            slat_delta_bound_mode=SLAT_DELTA_BOUND_SMOOTH,
            support_interval_policy=SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
            slat_residual_combination_policy=(
                SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
            ),
            slat_lora_delta_rms_ratio_cap=0.03,
            slat_support_delta_rms_ratio_cap=0.07,
            t_schedule="native_schedule",
            rollout_horizons="2,4,8",
            wrong_support_rank_weight=0.0,
            wrong_support_margin=0.0,
            wrong_support_probability=0.2,
            wrong_support_stock_weight=0.25,
            support_dropout_probability=0.2,
            rollout_probability=0.4,
            rollout_consistency_weight=0.25,
            endpoint_x0_weight=0.1,
            rollout_endpoint_rank_weight=0.2,
            rollout_endpoint_rank_margin=0.0005,
            train_cfg_strength=5.0,
        )
        validate_args(args)
        validate_args(
            argparse.Namespace(
                **{
                    **vars(args),
                    "rollout_supervision_policy": (
                        SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY
                    ),
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "branch caps exceed"):
            validate_args(
                argparse.Namespace(
                    **{
                        **vars(args),
                        "slat_lora_delta_rms_ratio_cap": 0.06,
                        "slat_support_delta_rms_ratio_cap": 0.06,
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "versioned"):
            validate_args(
                argparse.Namespace(
                    **{
                        **vars(args),
                        "training_semantics": DIRECT_SLAT_TRAINING_SEMANTICS_V4,
                        "rollout_supervision_policy": (
                            SLAT_ROLLOUT_SUPERVISION_TERMINAL_CONSTANT_MEMORY
                        ),
                    }
                )
            )

    def test_v5_branch_budget_composes_lora_and_support_increment(self) -> None:
        stock = torch.full((2, 1), 10.0)
        raw_lora = torch.full((2, 1), 14.0)
        raw_joint = torch.full((2, 1), 20.0)
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
        )
        effective, stats = combine_slat_lora_support_budgets(
            stock,
            raw_lora,
            raw_joint,
            coords,
            lora_delta_scale=1.0,
            lora_delta_rms_ratio_cap=0.2,
            support_delta_scale=1.0,
            support_delta_rms_ratio_cap=0.3,
            total_delta_scale=1.0,
            total_delta_rms_ratio_cap=0.5,
            bound_mode="hard_clip_v1",
        )
        self.assertTrue(torch.allclose(effective, torch.full((2, 1), 15.0)))
        self.assertAlmostEqual(
            float(stats["lora_branch_effective_flow_delta_rms"].item()),
            2.0,
        )
        self.assertAlmostEqual(
            float(stats["support_branch_effective_flow_delta_rms"].item()),
            3.0,
        )

    def test_v3_auxiliary_partition_is_deterministic_and_exclusive(self) -> None:
        choices = [
            deterministic_probability_partition(
                seed=seed,
                probabilities=(0.25, 0.25, 0.25),
            )
            for seed in range(20)
        ]
        self.assertEqual(
            choices,
            [
                deterministic_probability_partition(
                    seed=seed,
                    probabilities=(0.25, 0.25, 0.25),
                )
                for seed in range(20)
            ],
        )
        self.assertTrue(all(choice in {None, 0, 1, 2} for choice in choices))
        with self.assertRaisesRegex(ValueError, "sum to at most 1"):
            deterministic_probability_partition(
                seed=1,
                probabilities=(0.5, 0.5, 0.1),
            )

    def test_post_cfg_bound_limits_final_guided_delta(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
        x = _FakeSparse(torch.zeros((2, 1)), coords)
        flow = PostCFGSupportSLATRolloutFlow(
            _FakeGuidedModel(),
            "positive",
            "negative",
            (torch.tensor(0), torch.tensor(0), torch.tensor(0)),
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
            support_scale=1.0,
            slat_delta_scale=1.0,
            slat_delta_rms_ratio_cap=0.1,
        )
        output = flow(x, torch.tensor([750.0]), "positive")
        # stock-guided = 5*2 - 4*1 = 6; raw-guided = 5*4 - 4*1 = 16.
        # The final guided residual is capped at 0.1 * RMS(stock-guided) = 0.6.
        self.assertTrue(torch.allclose(output.feats, torch.full((2, 1), 6.6)))
        summary = flow.stats_summary()
        self.assertEqual(summary["policy_version"], "post_cfg_v2")
        self.assertEqual(summary["delta_clip_activated_calls"], 1)
        self.assertEqual(summary["negative_calls"], 1)
        self.assertAlmostEqual(
            summary["by_timestep"][0]["raw_guided_delta_rms"], 10.0
        )

    def test_post_cfg_wrapper_respects_guidance_interval(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        x = _FakeSparse(torch.zeros((1, 1)), coords)
        flow = PostCFGSupportSLATRolloutFlow(
            _FakeGuidedModel(),
            "positive",
            "negative",
            (torch.tensor(0), torch.tensor(0), torch.tensor(0)),
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
            support_scale=1.0,
            slat_delta_rms_ratio_cap=0.1,
        )
        output = flow(x, torch.tensor([250.0]), "positive")
        self.assertTrue(torch.allclose(output.feats, torch.tensor([[2.2]])))
        self.assertEqual(flow.negative_calls, 0)

    def test_post_cfg_component_identity_is_forwarded_and_audited(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        x = _FakeSparse(torch.zeros((1, 1)), coords)
        for component in (
            SLAT_ROLLOUT_COMPONENT_LORA_ONLY,
            SLAT_ROLLOUT_COMPONENT_ADAPTER_ONLY,
        ):
            model = _FakeGuidedModel()
            flow = PostCFGSupportSLATRolloutFlow(
                model,
                "positive",
                "negative",
                (torch.tensor(0), torch.tensor(0), torch.tensor(0)),
                cfg_strength=5.0,
                cfg_interval=(0.5, 1.0),
                support_scale=1.0,
                slat_delta_rms_ratio_cap=0.1,
                rollout_component=component,
            )
            flow(x, torch.tensor([750.0]), "positive")
            self.assertEqual(model.last_rollout_component, component)
            self.assertEqual(flow.stats_summary()["rollout_component"], component)

    def test_v4_support_gate_is_exact_stock_below_cfg_interval(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        x = _FakeSparse(torch.zeros((1, 1)), coords)
        flow = PostCFGSupportSLATRolloutFlow(
            _FakeGuidedModel(),
            "positive",
            "negative",
            (torch.tensor(0), torch.tensor(0), torch.tensor(0)),
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
            support_scale=1.0,
            slat_delta_rms_ratio_cap=0.1,
            slat_delta_bound_mode=SLAT_DELTA_BOUND_SMOOTH,
            support_interval_policy=SLAT_SUPPORT_INTERVAL_CFG_ACTIVE,
        )
        output = flow(x, torch.tensor([250.0]), "positive")
        self.assertTrue(torch.equal(output.feats, torch.tensor([[2.0]])))
        self.assertEqual(flow.stats_summary()["by_timestep"][0]["support_active"], 0.0)

    def test_raw_excess_and_support_rank_losses(self) -> None:
        excess_loss, ratio, excess = stock_relative_residual_excess_loss(
            stock_rms_per_batch=torch.tensor([2.0, 4.0]),
            raw_delta_rms_per_batch=torch.tensor([0.2, 2.0]),
            ratio_cap=0.25,
        )
        self.assertTrue(torch.allclose(ratio, torch.tensor([0.1, 0.5])))
        self.assertTrue(torch.allclose(excess, torch.tensor([0.0, 0.25])))
        self.assertAlmostEqual(float(excess_loss.item()), 0.03125)

        rank_loss, advantage = correct_over_wrong_support_rank_loss(
            correct_loss=torch.tensor(0.8),
            wrong_loss=torch.tensor(1.0),
            margin=0.1,
        )
        self.assertAlmostEqual(float(advantage.item()), 0.2, places=6)
        self.assertEqual(float(rank_loss.item()), 0.0)

    def test_wrong_support_selection_is_deterministic_and_object_disjoint(self) -> None:
        rows = [
            {"object_uid": "a", "support_seed": 42},
            {"object_uid": "b", "support_seed": 42},
            {"object_uid": "c", "support_seed": 43},
        ]
        first = deterministic_wrong_support_index(
            rows,
            correct_object_uid="a",
            support_seed=42,
            selection_seed=123,
        )
        second = deterministic_wrong_support_index(
            rows,
            correct_object_uid="a",
            support_seed=42,
            selection_seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(rows[first]["object_uid"], "b")
        self.assertTrue(deterministic_probability_event(seed=1, probability=1.0))
        self.assertFalse(deterministic_probability_event(seed=1, probability=0.0))


class DirectSLatEvaluationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"uid": "a0", "object_uid": "a"},
            {"uid": "b0", "object_uid": "b"},
            {"uid": "a1", "object_uid": "a"},
            {"uid": "c0", "object_uid": "c"},
            {"uid": "b1", "object_uid": "b"},
        ]

    def test_object_selection_is_deterministic_and_keeps_complete_objects(self) -> None:
        first = select_evaluation_indices(self.rows, max_objects=2)
        second = select_evaluation_indices(self.rows, max_objects=2)
        self.assertEqual(first, [0, 1, 2, 4])
        self.assertEqual(second, first)
        self.assertEqual(
            {self.rows[index]["object_uid"] for index in first},
            {"a", "b"},
        )

    def test_legacy_sample_selection_still_limits_rows(self) -> None:
        self.assertEqual(
            select_evaluation_indices(self.rows, max_samples=2),
            [0, 1],
        )

    def test_object_selection_rejects_ambiguous_or_undersized_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            select_evaluation_indices(self.rows, max_samples=1, max_objects=1)
        with self.assertRaisesRegex(ValueError, "only 3 unique objects"):
            select_evaluation_indices(self.rows, max_objects=4)


if __name__ == "__main__":
    unittest.main()
