#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from reconvggt_ar_adapter_a.experiment_gates import strict_decision_exit_code
from reconvggt_ar_adapter_a.pointpose_patch_features import (
    PROJECTED_PATCH_EVIDENCE_COUNT,
    PROJECTED_PATCH_FEATURE_NAMES,
    build_projected_patch_features,
    make_null_projected_patch_features,
)
from reconvggt_ar_adapter_a.stock_preserving_pointpose_bridge import (
    ContentBasedPhysicalVisualBridge,
    ContentPhysicalVisualFusionStage,
    LocalPhysicalFusionStage,
    MultiStageStockPreservingPhysicalBridge,
    PoseGuidedProjectedPatchBridge,
    ProjectedPatchVisualFusionStage,
    StockPreservingPhysicalBridge,
    ZeroCenteredPhysicalEncoder,
    ZeroCenteredPhysicalGridEncoder16,
    ZeroCenteredPhysicalTokenEncoder8,
    ZeroCenteredProjectedPatchEncoder,
    make_null_physical_grid,
)


class FakeBlock(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.context = nn.Linear(3072, cond_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return hidden + self.context(context.mean(dim=1, keepdim=True))


class FakeBridge(nn.Module):
    def __init__(self, cond_dim: int = 32, token_count: int = 9) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.intermediate_layer_idx = [4, 11, 17, 23]
        self.multiview_cond_tokens = nn.Parameter(torch.randn(1, token_count, cond_dim))
        self.cond_blocks = nn.ModuleList(FakeBlock(cond_dim) for _ in range(4))

    def forward(self, aggregated_tokens_list, image_cond):
        batch = aggregated_tokens_list[0].shape[0]
        hidden = self.multiview_cond_tokens.repeat(batch, 1, 1)
        for index, layer_index in enumerate(self.intermediate_layer_idx):
            token = aggregated_tokens_list[layer_index][:, :, 5:]
            context = torch.cat(
                (token.reshape(batch, -1, 2048), image_cond.reshape(batch, -1, 1024)),
                dim=-1,
            )
            hidden = self.cond_blocks[index](hidden, context)
        return hidden


def fake_inputs(batch: int = 2):
    aggregated = [torch.randn(batch, 1, 7, 2048) for _ in range(24)]
    image = torch.randn(batch, 1, 2, 1024)
    physical = torch.randn(batch, 14, 16, 16, 16)
    return aggregated, image, physical


class StockPreservingBridgeTest(unittest.TestCase):
    def test_projected_patch_center_sentinel_and_flatten_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(mask_path)
            features, report = build_projected_patch_features(
                prior_coords=np.asarray([[32, 32, 63]], dtype=np.int32),
                prior_conf=np.asarray([0.75], dtype=np.float32),
                intrinsics=np.asarray(
                    [[[100.0, 0.0, 49.5], [0.0, 100.0, 49.5], [0.0, 0.0, 1.0]]],
                    dtype=np.float32,
                ),
                extrinsics=np.eye(4, dtype=np.float32)[None],
                mask_paths=[mask_path],
                grid_transform="identity",
                extrinsics_type="c2w",
                camera_forward_sign=1.0,
                patch_grid_side=10,
            )
        self.assertEqual(tuple(features.shape), (1, 100, len(PROJECTED_PATCH_FEATURE_NAMES)))
        occupied = torch.nonzero(features[0, :, 0] > 0.5).flatten().tolist()
        self.assertEqual(occupied, [55])
        self.assertEqual(report["occupied_patch_count_per_view"], [1])

    def test_projected_patch_null_preserves_pose_geometry(self) -> None:
        torch.manual_seed(41)
        features = torch.randn(2, 17, len(PROJECTED_PATCH_FEATURE_NAMES))
        null = make_null_projected_patch_features(features)
        self.assertTrue(
            torch.equal(
                null[..., :PROJECTED_PATCH_EVIDENCE_COUNT],
                torch.zeros_like(null[..., :PROJECTED_PATCH_EVIDENCE_COUNT]),
            )
        )
        self.assertTrue(
            torch.equal(
                null[..., PROJECTED_PATCH_EVIDENCE_COUNT:],
                features[..., PROJECTED_PATCH_EVIDENCE_COUNT:],
            )
        )
        encoder = ZeroCenteredProjectedPatchEncoder(
            feature_dim=len(PROJECTED_PATCH_FEATURE_NAMES),
            hidden_dim=16,
            token_dim=8,
        ).eval()
        self.assertTrue(torch.equal(encoder(null), torch.zeros(2, 17, 8)))

    def test_projected_patch_stage_is_local_and_zero_init(self) -> None:
        torch.manual_seed(43)
        stage = ProjectedPatchVisualFusionStage(visual_dim=32, fusion_dim=8).eval()
        visual = torch.randn(1, 11, 32)
        physical = torch.randn(1, 11, 8)
        delta, logit, _, tensors = stage(visual, physical)
        null_delta, null_logit, _, null_tensors = stage(
            visual, torch.zeros_like(physical)
        )
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(null_delta, torch.zeros_like(null_delta)))
        self.assertTrue(torch.equal(null_logit, torch.zeros_like(null_logit)))
        self.assertGreater(float(tensors["attended_centered"].abs().max()), 0.0)
        self.assertTrue(
            torch.equal(
                null_tensors["attended_centered"],
                torch.zeros_like(null_tensors["attended_centered"]),
            )
        )
        with self.assertRaises(ValueError):
            stage(visual, physical[:, :-1])

    def test_zero_centered_encoder_maps_zero_to_zero(self) -> None:
        model = ZeroCenteredPhysicalEncoder(feature_dim=14, hidden_dim=16, cond_dim=32).eval()
        tokens = model(torch.zeros(2, 14, 16, 16, 16))
        self.assertTrue(torch.equal(tokens, torch.zeros_like(tokens)))

    def test_zero_centered_grid16_encoder_maps_zero_to_zero(self) -> None:
        model = ZeroCenteredPhysicalGridEncoder16(
            feature_dim=14, hidden_dim=8, cond_dim=32
        ).eval()
        tokens = model(torch.zeros(1, 14, 16, 16, 16))
        self.assertEqual(tuple(tokens.shape), (1, 4096, 32))
        self.assertTrue(torch.equal(tokens, torch.zeros_like(tokens)))

    def test_grid16_null_evidence_preserves_xyz_and_maps_to_zero(self) -> None:
        torch.manual_seed(4)
        model = ZeroCenteredPhysicalGridEncoder16(
            feature_dim=14, hidden_dim=8, cond_dim=32
        ).eval()
        physical = torch.randn(1, 14, 16, 16, 16)
        null_physical = make_null_physical_grid(physical)
        self.assertTrue(torch.equal(null_physical[:, :11], torch.zeros_like(physical[:, :11])))
        self.assertTrue(torch.equal(null_physical[:, 11:], physical[:, 11:]))
        tokens = model(null_physical)
        self.assertTrue(torch.equal(tokens, torch.zeros_like(tokens)))

    def test_content_encoder8_null_evidence_maps_to_zero(self) -> None:
        torch.manual_seed(23)
        model = ZeroCenteredPhysicalTokenEncoder8(
            feature_dim=14,
            hidden_dim=8,
            token_dim=16,
        ).eval()
        physical = torch.randn(1, 14, 16, 16, 16)
        tokens = model(physical)
        null_tokens = model(make_null_physical_grid(physical))
        self.assertEqual(tuple(tokens.shape), (1, 512, 16))
        self.assertTrue(torch.equal(null_tokens, torch.zeros_like(null_tokens)))

    def test_content_stage_is_internally_sensitive_but_zero_init_external(self) -> None:
        torch.manual_seed(29)
        stage = ContentPhysicalVisualFusionStage(
            visual_dim=32,
            fusion_dim=8,
            num_heads=2,
        ).eval()
        visual = torch.randn(1, 13, 32)
        physical = torch.randn(1, 17, 8)
        shuffled = torch.flip(physical, dims=(1,))
        delta, logit, _, tensors = stage(visual, physical)
        shuffled_delta, shuffled_logit, _, shuffled_tensors = stage(visual, shuffled)
        null_delta, null_logit, _, null_tensors = stage(
            visual,
            torch.zeros_like(physical),
        )
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(shuffled_delta, torch.zeros_like(shuffled_delta)))
        self.assertTrue(torch.equal(null_delta, torch.zeros_like(null_delta)))
        self.assertTrue(torch.equal(null_logit, torch.zeros_like(null_logit)))
        self.assertTrue(
            torch.equal(
                null_tensors["attended_centered"],
                torch.zeros_like(null_tensors["attended_centered"]),
            )
        )
        self.assertGreater(
            float(
                (
                    tensors["attended_centered"]
                    - shuffled_tensors["attended_centered"]
                )
                .abs()
                .max()
            ),
            0.0,
        )
        self.assertGreater(float((logit - shuffled_logit).abs().max()), 0.0)

    def test_local_fusion_residual_is_exactly_zero_for_zero_physical(self) -> None:
        torch.manual_seed(5)
        stage = LocalPhysicalFusionStage(cond_dim=32, hidden_dim=8).eval()
        nn.init.normal_(stage.output_proj.weight, std=0.1)
        nn.init.normal_(stage.output_proj.bias, std=0.1)
        nn.init.normal_(stage.alignment_proj.weight, std=0.1)
        nn.init.normal_(stage.alignment_proj.bias, std=0.1)
        bridge = torch.randn(1, 17, 32)
        physical = torch.zeros_like(bridge)
        delta, alignment, _ = stage(bridge, physical)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(alignment, torch.zeros_like(alignment)))

    def test_zero_init_fused_matches_stock_and_off_route_is_exact(self) -> None:
        torch.manual_seed(7)
        model = StockPreservingPhysicalBridge(
            FakeBridge(), bridge_last_blocks=1, cond_dim=32, hidden_dim=16, num_heads=4
        ).eval()
        aggregated, image, physical = fake_inputs()
        paths = model.condition_paths(aggregated, image, physical)
        self.assertTrue(torch.equal(paths.cond_fused, paths.cond_stock))
        stock = model.condition(
            aggregated, image, None, physical_present=False, physical_scale=1.0
        )
        direct = model.bridge(aggregated, image)
        self.assertTrue(torch.equal(stock, direct))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.bridge.parameters()))

    def test_adapter_learns_while_stock_bridge_remains_frozen(self) -> None:
        torch.manual_seed(9)
        model = StockPreservingPhysicalBridge(
            FakeBridge(), bridge_last_blocks=1, cond_dim=32, hidden_dim=16, num_heads=4
        ).train()
        aggregated, image, physical = fake_inputs(batch=1)
        optimizer = torch.optim.SGD(model.adapter.parameters(), lr=0.1)
        paths = model.condition_paths(aggregated, image, physical, alignment_gate_override=1.0)
        paths.cond_fused.square().mean().backward()
        self.assertGreater(float(model.adapter.output_proj.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.bridge.parameters()))
        optimizer.step()
        changed = model.condition_paths(
            aggregated, image, physical, alignment_gate_override=1.0
        )
        self.assertFalse(torch.equal(changed.cond_fused, changed.cond_stock))
        stock = model.condition(aggregated, image, None, physical_present=False)
        self.assertTrue(torch.equal(stock, model.bridge(aggregated, image)))

    def test_alignment_head_receives_positive_and_negative_gradients(self) -> None:
        torch.manual_seed(13)
        model = StockPreservingPhysicalBridge(
            FakeBridge(), bridge_last_blocks=1, cond_dim=32, hidden_dim=16, num_heads=4
        ).train()
        aggregated, image, physical = fake_inputs(batch=1)
        hidden, _ = model._prefix(aggregated, image)
        positive, _ = model.adapter.alignment_logits(hidden, physical)
        negative, _ = model.adapter.alignment_logits(hidden, torch.flip(physical, dims=(2,)))
        loss = nn.functional.binary_cross_entropy_with_logits(
            torch.cat((positive, negative)), torch.tensor([1.0, 0.0])
        )
        loss.backward()
        self.assertGreater(float(model.adapter.alignment_head[-1].weight.grad.abs().sum()), 0.0)

    def test_multistage_zero_init_and_paired_gradients(self) -> None:
        torch.manual_seed(17)
        model = MultiStageStockPreservingPhysicalBridge(
            FakeBridge(token_count=4096),
            fusion_stages=(0, 1, 2),
            cond_dim=32,
            physical_hidden_dim=8,
            local_hidden_dim=8,
        ).train()
        aggregated, image, physical = fake_inputs(batch=1)
        paths = model.condition_paths(aggregated, image, physical)
        self.assertTrue(torch.equal(paths.cond_fused, paths.cond_stock))
        null_paths = model.condition_paths(
            aggregated,
            image,
            make_null_physical_grid(physical),
            cond_stock=paths.cond_stock,
        )
        self.assertTrue(torch.equal(null_paths.cond_fused, paths.cond_stock))

        shuffled = model.condition_paths(
            aggregated,
            image,
            torch.flip(physical, dims=(2,)),
            cond_stock=paths.cond_stock,
        )
        alignment_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.cat((paths.alignment_logit, shuffled.alignment_logit)),
            torch.tensor([1.0, 0.0]),
        )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        (paths.cond_fused.square().mean() + alignment_loss).backward()
        output_grad = sum(
            float(stage.output_proj.weight.grad.abs().sum())
            for stage in model.stage_adapters.values()
        )
        alignment_grad = sum(
            float(stage.alignment_proj.weight.grad.abs().sum())
            for stage in model.stage_adapters.values()
        )
        self.assertGreater(output_grad, 0.0)
        self.assertGreater(alignment_grad, 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.bridge.parameters()))
        optimizer.step()

        changed = model.condition_paths(
            aggregated,
            image,
            physical,
            alignment_gate_override=1.0,
            cond_stock=paths.cond_stock,
        )
        self.assertFalse(torch.equal(changed.cond_fused, paths.cond_stock))
        null_after_update = model.condition_paths(
            aggregated,
            image,
            make_null_physical_grid(physical),
            alignment_gate_override=1.0,
            cond_stock=paths.cond_stock,
        )
        self.assertTrue(torch.equal(null_after_update.cond_fused, paths.cond_stock))

    def test_multistage_injection_order_is_after_selected_blocks(self) -> None:
        torch.manual_seed(19)
        model = MultiStageStockPreservingPhysicalBridge(
            FakeBridge(token_count=4096),
            fusion_stages=(0, 1, 2),
            cond_dim=32,
            physical_hidden_dim=8,
            local_hidden_dim=8,
        ).eval()
        aggregated, image, physical = fake_inputs(batch=1)
        cond_stock = model.stock_condition(aggregated, image)
        calls: list[str] = []
        handles = []
        for index, block in enumerate(model.bridge.cond_blocks):
            handles.append(
                block.register_forward_hook(
                    lambda _module, _inputs, _output, index=index: calls.append(
                        f"block{index}"
                    )
                )
            )
        for index in model.fusion_stages:
            handles.append(
                model.stage_adapters[str(index)].register_forward_hook(
                    lambda _module, _inputs, _output, index=index: calls.append(
                        f"adapter{index}"
                    )
                )
            )
        try:
            model.condition_paths(
                aggregated,
                image,
                physical,
                cond_stock=cond_stock,
            )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(
            calls,
            ["block0", "adapter0", "block1", "adapter1", "block2", "adapter2", "block3"],
        )

    def test_content_bridge_preblock_order_and_zero_init_sensitivity(self) -> None:
        torch.manual_seed(31)
        model = ContentBasedPhysicalVisualBridge(
            FakeBridge(token_count=9),
            fusion_stages=(0, 1),
            physical_hidden_dim=8,
            fusion_dim=8,
            num_heads=2,
        ).eval()
        aggregated, image, physical = fake_inputs(batch=1)
        shuffled_physical = torch.flip(physical, dims=(2,))
        paths = model.condition_paths(aggregated, image, physical)
        shuffled = model.condition_paths(
            aggregated,
            image,
            shuffled_physical,
            cond_stock=paths.cond_stock,
        )
        self.assertTrue(torch.equal(paths.cond_fused, paths.cond_stock))
        self.assertTrue(torch.equal(shuffled.cond_fused, paths.cond_stock))
        self.assertGreater(
            float(
                (
                    paths.stage_tensors["stage_0"]["attended_centered"]
                    - shuffled.stage_tensors["stage_0"]["attended_centered"]
                )
                .abs()
                .max()
            ),
            0.0,
        )

        calls: list[str] = []
        handles = []
        for index, block in enumerate(model.bridge.cond_blocks):
            handles.append(
                block.register_forward_hook(
                    lambda _module, _inputs, _output, index=index: calls.append(
                        f"block{index}"
                    )
                )
            )
        for index in model.fusion_stages:
            handles.append(
                model.stage_adapters[str(index)].register_forward_hook(
                    lambda _module, _inputs, _output, index=index: calls.append(
                        f"adapter{index}"
                    )
                )
            )
        try:
            model.condition_paths(
                aggregated,
                image,
                physical,
                cond_stock=paths.cond_stock,
            )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(
            calls,
            ["adapter0", "block0", "adapter1", "block1", "block2", "block3"],
        )

    def test_content_bridge_null_is_exact_after_parameter_update(self) -> None:
        torch.manual_seed(37)
        model = ContentBasedPhysicalVisualBridge(
            FakeBridge(token_count=9),
            fusion_stages=(0, 1),
            physical_hidden_dim=8,
            fusion_dim=8,
            num_heads=2,
        ).train()
        aggregated, image, physical = fake_inputs(batch=1)
        shuffled_physical = torch.flip(physical, dims=(2,))
        paths = model.condition_paths(aggregated, image, physical)
        shuffled = model.condition_paths(
            aggregated,
            image,
            shuffled_physical,
            cond_stock=paths.cond_stock,
        )
        alignment_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.cat((paths.alignment_logit, shuffled.alignment_logit)),
            torch.tensor([1.0, 0.0]),
        )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        (paths.cond_fused.square().mean() + alignment_loss).backward()
        self.assertGreater(
            sum(
                float(stage.output_proj.weight.grad.abs().sum())
                for stage in model.stage_adapters.values()
            ),
            0.0,
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.bridge.parameters()))
        optimizer.step()

        changed = model.condition_paths(
            aggregated,
            image,
            physical,
            alignment_gate_override=1.0,
            cond_stock=paths.cond_stock,
        )
        self.assertFalse(torch.equal(changed.cond_fused, paths.cond_stock))
        null_after_update = model.condition_paths(
            aggregated,
            image,
            make_null_physical_grid(physical),
            alignment_gate_override=1.0,
            cond_stock=paths.cond_stock,
        )
        self.assertTrue(torch.equal(null_after_update.cond_fused, paths.cond_stock))
        off = model.condition(
            aggregated,
            image,
            None,
            physical_present=False,
        )
        self.assertTrue(torch.equal(off, model.bridge(aggregated, image)))

    def test_pose_guided_patch_bridge_preblock_and_null_exact(self) -> None:
        torch.manual_seed(47)
        model = PoseGuidedProjectedPatchBridge(
            FakeBridge(token_count=9),
            fusion_stages=(0, 1),
            physical_hidden_dim=8,
            fusion_dim=8,
        ).train()
        aggregated, image, _ = fake_inputs(batch=1)
        features = torch.randn(1, 2, len(PROJECTED_PATCH_FEATURE_NAMES))
        shuffled = features.clone()
        shuffled[..., :PROJECTED_PATCH_EVIDENCE_COUNT] *= -1.0
        paths = model.condition_paths(aggregated, image, features)
        negative = model.condition_paths(
            aggregated, image, shuffled, cond_stock=paths.cond_stock
        )
        self.assertTrue(torch.equal(paths.cond_fused, paths.cond_stock))
        self.assertTrue(torch.equal(negative.cond_fused, paths.cond_stock))
        alignment_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.cat((paths.alignment_logit, negative.alignment_logit)),
            torch.tensor([1.0, 0.0]),
        )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        (paths.cond_fused.square().mean() + alignment_loss).backward()
        optimizer.step()
        changed = model.condition_paths(
            aggregated,
            image,
            features,
            alignment_gate_override=1.0,
            cond_stock=paths.cond_stock,
        )
        self.assertFalse(torch.equal(changed.cond_fused, paths.cond_stock))
        null_after = model.condition_paths(
            aggregated,
            image,
            make_null_projected_patch_features(features),
            alignment_gate_override=1.0,
            cond_stock=paths.cond_stock,
        )
        self.assertTrue(torch.equal(null_after.cond_fused, paths.cond_stock))
        off = model.condition(aggregated, image, None, physical_present=False)
        self.assertTrue(torch.equal(off, model.bridge(aggregated, image)))

    def test_strict_decision_exit_code(self) -> None:
        self.assertEqual(strict_decision_exit_code(True), 0)
        self.assertEqual(strict_decision_exit_code(False), 2)


if __name__ == "__main__":
    unittest.main()
