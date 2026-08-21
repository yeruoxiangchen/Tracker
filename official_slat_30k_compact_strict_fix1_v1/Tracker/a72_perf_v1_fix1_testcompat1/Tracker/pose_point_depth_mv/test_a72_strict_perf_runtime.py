#!/usr/bin/env python3
"""Equivalence tests for the isolated A72 strict performance runtime."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from pose_point_depth_mv.summarize_a72_slat_perf import summarize
from pose_point_depth_mv import train_native_slat_genrecon as training
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import finite_tree


class _TinyTrainable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition = nn.Linear(3, 2, bias=True)
        self.lora_adapter = nn.Linear(2, 2, bias=False)
        self.view_fusion = nn.Linear(2, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        condition = self.condition(value)
        adapted = self.lora_adapter(condition)
        return self.view_fusion(condition + adapted)


def _legacy_gradient_norms(model: nn.Module) -> dict[str, float]:
    values = {"lora": 0.0, "condition": 0.0, "view_fusion": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = (
            "lora"
            if "lora_" in name
            else "view_fusion"
            if name.startswith("view_fusion.")
            else "condition"
        )
        values[key] += float(
            parameter.grad.detach().float().square().sum().item()
        )
    return {key: value**0.5 for key, value in values.items()}


@torch.no_grad()
def _legacy_ema_update(
    state: dict[str, torch.Tensor], model: nn.Module, decay: float
) -> None:
    named = training.trainable_named_parameters(model)
    for name, parameter in named.items():
        state[name].mul_(decay).add_(
            parameter.detach().float(), alpha=1.0 - decay
        )


class A72StrictPerformanceRuntimeTest(unittest.TestCase):
    def test_finite_tree_flag_matches_legacy_nested_semantics(self) -> None:
        cases = (
            {"a": torch.tensor([1.0, -2.0]), "b": [3.0, (4, None)]},
            {"a": torch.tensor([float("nan")])},
            {"a": [torch.tensor([float("inf")])]},
            {"a": torch.tensor([1]), "python": float("-inf")},
            {"empty": [], "ignored": "value"},
        )
        for value in cases:
            with self.subTest(value=value):
                actual = bool(
                    training.finite_tree_flag(
                        value, device=torch.device("cpu")
                    ).item()
                )
                self.assertEqual(actual, finite_tree(value))

    def test_gradient_and_parameter_finite_flags(self) -> None:
        model = _TinyTrainable()
        parameters = list(model.parameters())
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, float(index + 1))
        self.assertTrue(
            bool(
                training.gradients_finite_flag(
                    parameters, device=torch.device("cpu")
                ).item()
            )
        )
        self.assertTrue(
            bool(
                training.parameters_finite_flag(
                    parameters, device=torch.device("cpu")
                ).item()
            )
        )
        parameters[0].grad.reshape(-1)[0] = float("nan")
        self.assertFalse(
            bool(
                training.gradients_finite_flag(
                    parameters, device=torch.device("cpu")
                ).item()
            )
        )
        parameters[0].grad.reshape(-1)[0] = 0.0
        with torch.no_grad():
            parameters[1].reshape(-1)[0] = float("inf")
        self.assertFalse(
            bool(
                training.parameters_finite_flag(
                    parameters, device=torch.device("cpu")
                ).item()
            )
        )

    def test_optimizer_state_finite_flag_detects_tensor_and_python_nan(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertTrue(
            bool(
                training.optimizer_state_finite_flag(
                    optimizer, device=torch.device("cpu")
                ).item()
            )
        )
        state = next(iter(optimizer.state.values()))
        state["exp_avg"].reshape(-1)[0] = float("nan")
        self.assertFalse(
            bool(
                training.optimizer_state_finite_flag(
                    optimizer, device=torch.device("cpu")
                ).item()
            )
        )
        state["exp_avg"].reshape(-1)[0] = 0.0
        state["diagnostic"] = float("nan")
        self.assertFalse(
            bool(
                training.optimizer_state_finite_flag(
                    optimizer, device=torch.device("cpu")
                ).item()
            )
        )

    def test_gradient_norms_match_legacy(self) -> None:
        torch.manual_seed(7)
        model = _TinyTrainable()
        model(torch.randn(4, 3)).square().mean().backward()
        expected = _legacy_gradient_norms(model)
        actual = training.gradient_norms(model)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertTrue(
                math.isclose(actual[key], expected[key], rel_tol=1.0e-6, abs_tol=1.0e-8),
                (key, expected[key], actual[key]),
            )

    def test_foreach_ema_matches_legacy_update(self) -> None:
        torch.manual_seed(11)
        model = _TinyTrainable()
        legacy = training.initialize_ema_state(model)
        optimized = {name: value.clone() for name, value in legacy.items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.25)
        decay = 0.987
        _legacy_ema_update(legacy, model, decay)
        training.update_ema_state(optimized, model, decay=decay)
        self.assertEqual(set(legacy), set(optimized))
        for name in legacy:
            self.assertTrue(
                torch.equal(legacy[name], optimized[name]),
                f"EMA differs for {name}",
            )

    def test_view_sampling_preserves_rng_draws_and_order(self) -> None:
        sample = {
            "condition": {"cond": list(range(8))},
            "lifting_sample": {
                "visual_patch_features": torch.empty(8, 4, 3),
            },
        }
        args = training.make_parser().parse_args(
            [
                "--cache_manifest", "cache.json",
                "--lifting_cache_manifest", "lifting.json",
                "--target_decoder_audit", "decoder.json",
                "--native_ss_report", "ss.json",
                "--stock_slat_freeze", "freeze.json",
                "--output_dir", "output",
                "--min_condition_views", "1",
                "--max_condition_views", "8",
            ]
        )

        def legacy() -> torch.Tensor:
            count_all = int(
                sample["lifting_sample"]["visual_patch_features"].shape[0]
            )
            upper = min(count_all, int(args.max_condition_views))
            lower = min(upper, int(args.min_condition_views))
            count = int(torch.randint(lower, upper + 1, (), device="cpu").item())
            return torch.randperm(count_all, device="cpu")[:count]

        torch.manual_seed(12345)
        expected = legacy()
        expected_tail = torch.rand(4)
        torch.manual_seed(12345)
        actual = training.sample_view_indices(sample, args, torch.device("cpu"))
        actual_tail = torch.rand(4)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_tail, expected_tail))

    def test_performance_flags_are_operational_not_checkpoint_identity(self) -> None:
        args = training.make_parser().parse_args(
            [
                "--cache_manifest", "cache.json",
                "--lifting_cache_manifest", "lifting.json",
                "--target_decoder_audit", "decoder.json",
                "--native_ss_report", "ss.json",
                "--stock_slat_freeze", "freeze.json",
                "--output_dir", "output",
                "--num_workers", "3",
                "--prefetch_factor", "4",
                "--no-persistent_workers",
                "--no-pin_memory",
            ]
        )
        training.validate_args(args)
        payload = training.checkpoint_args(args)
        for name in (
            "prefetch_factor",
            "persistent_workers",
            "pin_memory",
            "torch_num_threads",
            "torch_num_interop_threads",
            "skip_redundant_cache_finite_checks",
        ):
            self.assertNotIn(name, payload)

    def test_performance_argument_validation(self) -> None:
        base = [
            "--cache_manifest", "cache.json",
            "--lifting_cache_manifest", "lifting.json",
            "--target_decoder_audit", "decoder.json",
            "--native_ss_report", "ss.json",
            "--stock_slat_freeze", "freeze.json",
            "--output_dir", "output",
        ]
        for extra in (
            ("--num_workers", "-1"),
            ("--prefetch_factor", "0"),
            ("--torch_num_threads", "-1"),
            ("--torch_num_interop_threads", "-1"),
        ):
            with self.subTest(extra=extra):
                args = training.make_parser().parse_args([*base, *extra])
                with self.assertRaises(ValueError):
                    training.validate_args(args)

    def test_cache_finite_skip_is_resume_only(self) -> None:
        base = [
            "--cache_manifest", "cache.json",
            "--lifting_cache_manifest", "lifting.json",
            "--target_decoder_audit", "decoder.json",
            "--native_ss_report", "ss.json",
            "--stock_slat_freeze", "freeze.json",
            "--output_dir", "output",
            "--skip_redundant_cache_finite_checks",
        ]
        with self.assertRaises(ValueError):
            training.validate_args(training.make_parser().parse_args(base))
        resumed = training.make_parser().parse_args(
            [*base, "--resume", "step.pt"]
        )
        training.validate_args(resumed)
        self.assertNotIn(
            "skip_redundant_cache_finite_checks",
            training.checkpoint_args(resumed),
        )

    def test_performance_summary_uses_requested_stable_window(self) -> None:
        report = {
            "model_summary": {
                "optimization": {"global_effective_batch": 8},
                "runtime_performance": {"profile": "a72_strict_perf_v1"},
            },
            "history": [
                {"step": 10000},
                {"step": 10001, "optimizer_step_wall_seconds": 20.0},
                {"step": 10002, "optimizer_step_wall_seconds": 10.0},
                {"step": 10003, "optimizer_step_wall_seconds": 4.0},
                {"step": 10004, "optimizer_step_wall_seconds": 6.0},
            ],
        }
        result = summarize(
            report,
            start_step=10000,
            end_step=10004,
            discard_first=2,
        )
        self.assertEqual(result["timed_optimizer_steps"], 2)
        self.assertEqual(result["first_timed_step"], 10003)
        self.assertEqual(result["last_timed_step"], 10004)
        self.assertEqual(result["seconds_per_optimizer_step"]["mean"], 5.0)
        self.assertEqual(result["global_samples_per_second"], 1.6)


if __name__ == "__main__":
    unittest.main()
