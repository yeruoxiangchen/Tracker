#!/usr/bin/env python3
"""Regression tests for the strict A72 DDP/CPU-lifting input contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch import nn
from torch.distributed.utils import _to_kwargs
from torch.nn.parallel import DistributedDataParallel

from pose_point_depth_mv.native_slat_genrecon import (
    select_sparse_frustum_inputs_cpu,
    validate_strict_cpu_lifting_sample,
)
from pose_point_depth_mv.native_ss_genrecon import select_dino_features
from pose_point_depth_mv import native_slat_genrecon as slat_runtime
from pose_point_depth_mv import train_native_slat_genrecon as training


def _lifting_sample(views: int = 8) -> dict[str, Any]:
    patches = 4
    channels = 1026
    visual = torch.arange(
        views * patches * channels, dtype=torch.float32
    ).reshape(views, patches, channels)
    return {
        "visual_patch_features": visual,
        "intrinsics": torch.arange(views * 9, dtype=torch.float32).reshape(
            views, 3, 3
        ),
        "extrinsics": torch.arange(views * 16, dtype=torch.float32).reshape(
            views, 4, 4
        ),
        "predicted_depth": torch.zeros(views, 6, 7),
        "depth_confidence": torch.ones(views, 6, 7),
        "masks": torch.ones(views, 6, 7, dtype=torch.bool),
        "prior_coords": torch.zeros(5, 3, dtype=torch.int64),
        "prior_confidence": torch.ones(5),
        "stock_condition": torch.zeros(1, 4, 8),
        "target": torch.zeros(8, 2, 2, 2),
        "grid_transform": "identity",
        "extrinsics_type": "world_to_camera",
        "camera_forward_sign": 1.0,
    }


class _MixedInputProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.observed_lifting_devices: list[str] = []

    def forward(
        self, value: torch.Tensor, lifting_sample: dict[str, Any] | None
    ) -> torch.Tensor:
        if lifting_sample is None:
            offset = value.new_zeros(())
        else:
            validate_strict_cpu_lifting_sample(lifting_sample)
            self.observed_lifting_devices.append(
                lifting_sample["visual_patch_features"].device.type
            )
            offset = lifting_sample["visual_patch_features"][0, 0, 0].to(
                device=value.device, dtype=value.dtype
            )
        return value * self.scale + offset


class _ReducerProbe:
    @staticmethod
    def _rebuild_buckets() -> bool:
        return False


class _DDPPreForwardProbe(DistributedDataParallel):
    """Exercise PyTorch's real DDP._pre_forward without a process group."""

    def __init__(self, device_ids: list[int] | None) -> None:
        nn.Module.__init__(self)
        self._accum_grad_hooks = []
        self._lazy_init_ran = True
        self._delay_all_reduce_all_params = False
        # PyTorch 2.10's DDP._should_disable_cpp_reducer() reads this private
        # field.  The probe intentionally bypasses DDP.__init__, so initialize
        # the non-Python-reducer branch explicitly for cross-version testing.
        self._use_python_reducer = False
        self.require_backward_grad_sync = False
        self.reducer = _ReducerProbe()
        self._join_config = SimpleNamespace(
            is_first_joinable=False,
            enable=False,
            throw_on_early_termination=False,
        )
        self.device_ids = device_ids
        self.device_type = "meta"
        self.use_side_stream_for_tensor_copies = False
        self.mixed_precision = None

    def _should_disable_cpp_reducer(self) -> bool:
        return False

    def _check_sync_bufs_pre_fwd(self) -> bool:
        return False


class A72DDPCPULiftingContractTest(unittest.TestCase):
    def test_real_torch_to_kwargs_recursively_moves_nested_tensor_leaves(self) -> None:
        payload = {"sample": {"nested": {"value": torch.arange(3)}}}
        _, moved_kwargs = _to_kwargs(
            (), payload, torch.device("meta"), False
        )
        moved = moved_kwargs[0]["sample"]["nested"]["value"]
        self.assertEqual(moved.device.type, "meta")
        self.assertEqual(moved.shape, (3,))

    def test_real_ddp_pre_forward_gate_preserves_payload_when_ids_none(self) -> None:
        payload = {"sample": {"nested": torch.arange(3)}}
        _, kept_kwargs = _DDPPreForwardProbe(None)._pre_forward(**payload)
        _, moved_kwargs = _DDPPreForwardProbe([0])._pre_forward(**payload)
        self.assertEqual(
            kept_kwargs["sample"]["nested"].device.type, "cpu"
        )
        self.assertEqual(
            moved_kwargs["sample"]["nested"].device.type, "meta"
        )

    def test_ddp_policy_has_no_automatic_input_device(self) -> None:
        kwargs = training.strict_perf_ddp_kwargs()
        self.assertIn("device_ids", kwargs)
        self.assertIsNone(kwargs["device_ids"])
        self.assertNotIn("output_device", kwargs)
        self.assertTrue(kwargs["gradient_as_bucket_view"])

    def test_strict_contract_checks_every_tensor_leaf(self) -> None:
        sample = _lifting_sample()
        inventory = validate_strict_cpu_lifting_sample(sample)
        self.assertGreaterEqual(inventory["tensor_count"], 10)
        self.assertGreater(inventory["tensor_bytes"], 0)
        sample["unused_nested"] = {"unexpected_gpu_leaf": torch.empty(1, device="meta")}
        with self.assertRaisesRegex(RuntimeError, "unused_nested.unexpected_gpu_leaf"):
            validate_strict_cpu_lifting_sample(sample)

    def test_cpu_view_selection_matches_reference_for_2_4_8_views(self) -> None:
        sample = _lifting_sample()
        order = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], dtype=torch.long)
        for count in (2, 4, 8):
            with self.subTest(count=count):
                indices = order[:count]
                visual, intrinsics, extrinsics, image_shape, inventory = (
                    select_sparse_frustum_inputs_cpu(sample, indices)
                )
                self.assertTrue(
                    torch.equal(
                        visual,
                        select_dino_features(sample["visual_patch_features"]).index_select(
                            0, indices
                        ),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        intrinsics, sample["intrinsics"].index_select(0, indices)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        extrinsics, sample["extrinsics"].index_select(0, indices)
                    )
                )
                self.assertEqual(image_shape, (6, 7))
                self.assertGreater(inventory["tensor_count"], 0)

    def test_selected_view_projection_matches_legacy_order_for_2_4_8_views(
        self,
    ) -> None:
        sample = _lifting_sample()
        order = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], dtype=torch.long)
        coords = torch.tensor(
            [[0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]], dtype=torch.int32
        )

        def fixed_geometry(**kwargs: Any) -> dict[str, torch.Tensor]:
            views = int(kwargs["intrinsics"].shape[0])
            points = int(kwargs["coords"].shape[0])
            grid = torch.zeros(
                views, points, 2, device=kwargs["intrinsics"].device
            )
            valid = torch.ones(
                views,
                points,
                dtype=torch.bool,
                device=kwargs["intrinsics"].device,
            )
            return {"patch_grid": grid, "valid": valid}

        for count in (2, 4, 8):
            with self.subTest(count=count), patch.object(
                slat_runtime, "sparse_projection_geometry", fixed_geometry
            ):
                indices = order[:count]
                actual, valid, _ = slat_runtime.project_sparse_frustum_dino(
                    sample,
                    coords,
                    device=torch.device("cpu"),
                    view_indices=indices,
                )
                # Legacy order: transfer the complete tensors first (a CPU no-op
                # in this source-side test), then select views on the target.
                visual = select_dino_features(sample["visual_patch_features"]).to(
                    dtype=torch.float32
                ).index_select(0, indices)
                maps = visual.permute(0, 2, 1).reshape(count, 1024, 2, 2)
                geometry = fixed_geometry(intrinsics=sample["intrinsics"][:count], coords=coords)
                expected = slat_runtime._sample_patch_maps(
                    maps, geometry["patch_grid"]
                ).permute(0, 2, 1).contiguous()
                expected = expected * geometry["valid"][..., None]
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(valid, geometry["valid"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_select_then_h2d_matches_legacy_h2d_then_select(self) -> None:
        device = torch.device("cuda:0")
        sample = _lifting_sample()
        order = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], device=device)
        for count in (2, 4, 8):
            with self.subTest(count=count):
                indices_cuda = order[:count]
                visual_cpu, intrinsics_cpu, extrinsics_cpu, _, _ = (
                    select_sparse_frustum_inputs_cpu(sample, indices_cuda)
                )
                legacy_visual = select_dino_features(
                    sample["visual_patch_features"]
                ).to(device).index_select(0, indices_cuda)
                legacy_intrinsics = sample["intrinsics"].to(device).index_select(
                    0, indices_cuda
                )
                legacy_extrinsics = sample["extrinsics"].to(device).index_select(
                    0, indices_cuda
                )
                self.assertTrue(torch.equal(visual_cpu.to(device), legacy_visual))
                self.assertTrue(
                    torch.equal(intrinsics_cpu.to(device), legacy_intrinsics)
                )
                self.assertTrue(
                    torch.equal(extrinsics_cpu.to(device), legacy_extrinsics)
                )

    def test_conditional_and_unconditional_forward_routing(self) -> None:
        model = _MixedInputProbe()
        sample = _lifting_sample()
        conditional = model(torch.ones(2), sample)
        unconditional = model(torch.ones(2), None)
        (conditional.square().mean() + unconditional.square().mean()).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(model.observed_lifting_devices, ["cpu"])
        self.assertEqual(tuple(conditional.shape), (2,))
        self.assertEqual(tuple(unconditional.shape), (2,))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        )
        self.assertTrue(any(bool(torch.count_nonzero(gradient)) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
