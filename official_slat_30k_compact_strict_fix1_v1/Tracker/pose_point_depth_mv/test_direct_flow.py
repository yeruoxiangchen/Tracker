from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
for path in (RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pose_point_depth_mv.direct_flow import (
    DIRECT_METADATA_NAMES,
    DirectPhysicalFlowModel,
    DirectViewTokenEncoder,
    flow_tokens_to_volume_xyz,
    lifting_cache_identity,
    null_evidence_like,
    volume_xyz_to_flow_tokens,
)
from trellis.modules.spatial import patchify, unpatchify


class FakeTimeEmbedder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.projection(t.float().reshape(-1, 1))


class FakeBlock(nn.Module):
    def __init__(self, channels: int, core: "FakeCore") -> None:
        super().__init__()
        self.core_reference = [core]
        self.base = nn.Linear(channels, channels, bias=False)
        self.lora_weight = nn.Parameter(torch.zeros(channels))

    def forward(
        self, h: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        del condition
        output = h + 0.05 * self.base(h) + 0.01 * t[:, None]
        if self.core_reference[0].adapter_enabled:
            output = output + self.lora_weight[None, None]
        return output


class FakeCore(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.resolution = 16
        self.in_channels = 8
        self.out_channels = 8
        self.patch_size = 1
        self.model_channels = channels
        self.share_mod = False
        self.dtype = torch.float32
        self.adapter_enabled = True
        self.input_layer = nn.Linear(8, channels)
        self.register_buffer("pos_emb", torch.zeros(16**3, channels))
        self.t_embedder = FakeTimeEmbedder(channels)
        self.blocks = nn.ModuleList([FakeBlock(channels, self)])
        self.out_layer = nn.Linear(channels, 8)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h) + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, t_emb, condition)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(1, 8, 16, 16, 16)
        return unpatchify(h, self.patch_size).contiguous()


class FakeBase(nn.Module):
    def __init__(self, model: FakeCore) -> None:
        super().__init__()
        self.model = model


class FakePeftFlow(nn.Module):
    def __init__(self, core: FakeCore) -> None:
        super().__init__()
        self.base_model = FakeBase(core)

    @contextmanager
    def disable_adapter(self):
        previous = self.base_model.model.adapter_enabled
        self.base_model.model.adapter_enabled = False
        try:
            yield
        finally:
            self.base_model.model.adapter_enabled = previous

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.base_model.model(x, t, condition)


def fake_evidence(
    *, views: int = 4, visual_channels: int = 8, active: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    visual = torch.randn(1, views, 16**3, visual_channels)
    geometry = torch.randn(1, views, 16**3, 18)
    weight = torch.ones(1, views, 16**3) if active else torch.zeros(1, views, 16**3)
    metadata = torch.zeros(1, len(DIRECT_METADATA_NAMES), 16, 16, 16)
    if active:
        metadata[:, DIRECT_METADATA_NAMES.index("active_support")] = 1.0
        metadata[:, DIRECT_METADATA_NAMES.index("reliability")] = 0.75
    return visual, geometry, weight, metadata, {"views": views}


class DirectFlowTests(unittest.TestCase):
    def test_cache_schema_hash_is_split_independent(self) -> None:
        base = {
            "format": "cache.v1",
            "stock_condition_source": "native_unmodified_reconviagen_vggt",
            "lifting_feature_source": "separate_vggt_depth_pipeline",
            "visual_feature_dim": 12,
            "feature_metadata": {"patch_count": 4},
            "metadata_names": ["support"],
            "metadata_schema_hash": "schema",
            "config": {"resolution": 16},
            "config_hash": "config",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, uid in (("train", "A0"), ("val", "B0")):
                path = root / f"{name}.json"
                payload = dict(base)
                payload["samples"] = [{"uid": uid, "object_uid": uid[0]}]
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            train = lifting_cache_identity(paths[0])
            val = lifting_cache_identity(paths[1])
            self.assertEqual(train["cache_schema_hash"], val["cache_schema_hash"])
            self.assertNotEqual(train["uid_hash"], val["uid_hash"])

    def test_distinct_axis_xyz_token_mapping_and_roundtrip(self) -> None:
        volume = torch.zeros(1, 1, 16, 16, 16)
        sentinels = {
            (2, 5, 11): 7.0,
            (1, 0, 0): 11.0,
            (0, 2, 0): 13.0,
            (0, 0, 3): 17.0,
        }
        for (x, y, z), value in sentinels.items():
            volume[0, 0, x, y, z] = value
        tokens = volume_xyz_to_flow_tokens(volume)
        for (x, y, z), value in sentinels.items():
            index = x * 16 * 16 + y * 16 + z
            self.assertEqual(float(tokens[0, index, 0]), value)
        self.assertTrue(
            torch.equal(tokens, volume_xyz_to_flow_tokens(patchify(volume, 1)))
        )
        self.assertTrue(torch.equal(flow_tokens_to_volume_xyz(tokens), volume))

    def test_encoder_maps_16_grid_to_flow_tokens(self) -> None:
        encoder = DirectViewTokenEncoder(
            visual_channels=8,
            flow_channels=16,
            hidden_dim=8,
        )
        tokens, stats = encoder(*fake_evidence()[:4])
        self.assertEqual(tuple(tokens.shape), (1, 16**3, 16))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(float(tokens.abs().max()), 0.0)
        self.assertGreater(float(stats["pair_valid_ratio"]), 0.0)

    def test_null_evidence_is_exact_zero(self) -> None:
        encoder = DirectViewTokenEncoder(
            visual_channels=8,
            flow_channels=16,
            hidden_dim=8,
        )
        null = fake_evidence(active=False)
        tokens, _ = encoder(*null[:4])
        self.assertEqual(float(tokens.abs().max()), 0.0)
        self.assertFalse(encoder.evidence_present(null[3], null[2]))
        generated_null = null_evidence_like(fake_evidence())
        self.assertTrue(all(float(value.abs().max()) == 0.0 for value in generated_null))

    def test_stock_bypass_and_enabled_gradients(self) -> None:
        core = FakeCore()
        encoder = DirectViewTokenEncoder(
            visual_channels=8,
            flow_channels=16,
            hidden_dim=8,
        )
        model = DirectPhysicalFlowModel(FakePeftFlow(core), encoder)
        x_t = torch.randn(1, 8, 16, 16, 16)
        t = torch.tensor([500.0])
        condition = torch.randn(1, 3, 4)
        evidence = fake_evidence()
        stock = model.stock_prediction(x_t, t, condition)
        native_disabled, _ = model.conditioned_prediction(
            x_t,
            t,
            condition,
            *evidence[:4],
            stock_velocity=stock,
            physical_present=False,
        )
        null, _ = model.conditioned_prediction(
            x_t,
            t,
            condition,
            *null_evidence_like(evidence),
            stock_velocity=stock,
        )
        enabled_zero, _ = model.conditioned_prediction(
            x_t, t, condition, *evidence[:4], stock_velocity=stock
        )
        self.assertTrue(torch.equal(native_disabled, stock))
        self.assertTrue(torch.equal(null, stock))
        self.assertTrue(torch.equal(enabled_zero, stock))

        nn.init.normal_(encoder.output.weight, std=0.02)
        core.blocks[0].lora_weight.data.fill_(0.01)
        scale_zero, _ = model.conditioned_prediction(
            x_t,
            t,
            condition,
            *evidence[:4],
            stock_velocity=stock,
            physical_scale=0.0,
        )
        self.assertTrue(torch.equal(scale_zero, stock))
        enabled, _ = model.conditioned_prediction(
            x_t, t, condition, *evidence[:4], stock_velocity=stock
        )
        self.assertGreater(float((enabled - stock).abs().max()), 0.0)
        enabled.square().mean().backward()
        self.assertIsNotNone(encoder.output.weight.grad)
        self.assertIsNotNone(core.blocks[0].lora_weight.grad)
        self.assertGreater(float(encoder.output.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.blocks[0].lora_weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
