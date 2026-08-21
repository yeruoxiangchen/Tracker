from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock
import unittest

from PIL import Image
import torch

from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (
    _build_dino_pipeline,
    encode_object,
)


class PrepareOmniRealDinoOnlyModelInputsTest(unittest.TestCase):
    def test_pipeline_builder_initializes_only_dino(self) -> None:
        class Encoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        class Pipeline:
            initialized_model = None

            def __init__(self) -> None:
                self.models = {"should_be_replaced": object()}

            def _init_image_cond_model(self, name: str) -> None:
                Pipeline.initialized_model = name
                self.models["image_cond_model"] = Encoder()

        with mock.patch(
            "trellis.pipelines.TrellisImageTo3DPipeline", Pipeline
        ):
            pipeline = _build_dino_pipeline("dino-test", torch.device("cpu"))
        self.assertEqual(Pipeline.initialized_model, "dino-test")
        self.assertEqual(set(pipeline.models), {"image_cond_model"})
        self.assertFalse(hasattr(pipeline, "VGGT_model"))
        self.assertFalse(next(pipeline.models["image_cond_model"].parameters()).requires_grad)

    def test_encoder_never_requires_vggt_outputs(self) -> None:
        class DinoPipeline:
            def __init__(self, encoded: torch.Tensor) -> None:
                self.encoded = encoded
                self.calls = 0

            def encode_image(self, images):
                self.calls += 1
                return self.encoded

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(2):
                path = root / f"{index}.png"
                Image.new("RGB", (518, 518), color=(index, 0, 0)).save(path)
                paths.append(str(path))
            encoded = torch.randn(2, 1374, 1024)
            pipeline = DinoPipeline(encoded)
            payload, stats = encode_object(
                pipeline,
                {
                    "category": "category",
                    "object_id": "object",
                    "condition_sha256": "condition",
                    "prepared_rgb_paths": paths,
                },
                ss_context_tokens=64,
            )
        self.assertEqual(pipeline.calls, 1)
        self.assertFalse(hasattr(pipeline, "VGGT_model"))
        torch.testing.assert_close(
            payload["visual_patch_features"].float(), encoded[:, 5:].half().float()
        )
        self.assertEqual(payload["stock_condition"].shape, (1, 64, 1024))
        self.assertEqual(stats["slat_positive_view_count"], 2)
        self.assertFalse(payload["vggt_model_executed"])


if __name__ == "__main__":
    unittest.main()
