"""Real Qwen3-VL processor and tiny random model, without pretrained GPU loading.

Run in chartvlm. These verify the actual HF multimodal interface, NOT evidence
selection quality, 8B GPU capacity, or full-model accuracy.
"""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import transformers

from mpp_video_qa.evidence_search import DEFAULT_CONFIG, SYSTEM_PROMPT
from mpp_video_qa.evidence_hf import HFSearchBackend
from scripts.generate_vlm_teacher import build_processor

PROCESSOR_PATH = Path(os.environ.get("EVIDENCE_TEST_PROCESSOR",
    str(Path(__file__).resolve().parents[2] / "chart_verification_vlm/checkpoints/Qwen3-VL-4B-Instruct")))


@unittest.skipUnless(hasattr(transformers, "Qwen3VLForConditionalGeneration") and PROCESSOR_PATH.is_dir(),
                     "requires the modern environment and a local Qwen3-VL processor")
class RealHFSearchTests(unittest.TestCase):
    def setUp(self):
        self.images = [Image.new("RGB", (320, 200) if i < 5 else
                                ((40, 300), (200, 80), (100, 100))[i % 3],
                                 (i * 10, 90, 120)) for i in range(15)]
        self.backend = HFSearchBackend.__new__(HFSearchBackend)
        self.backend.config = dict(DEFAULT_CONFIG)
        self.backend.processor = build_processor(PROCESSOR_PATH, DEFAULT_CONFIG["min_pixels"], DEFAULT_CONFIG["max_pixels"])

    def tearDown(self):
        for picture in self.images:
            picture.close()

    def messages(self, count):
        originals = [{"type": "image", "image": im} for im in self.images[:5]]
        originals.append({"type": "text", "text": "Question: Why should the ego vehicle stop?"})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": originals}]
        if count > 5:
            messages.extend([{"role": "assistant", "content": '{"action":"inspect","requests":[]}'},
                             {"role": "user", "content": [{"type": "text", "text": "Actual tool crop results:"}] +
                              [{"type": "image", "image": im} for im in self.images[5:count]]}])
        return messages

    def test_fifteen_actual_images_obey_whole_history_budget(self):
        for count in (5, 10, 15):
            inputs, metadata = self.backend.prepare(self.messages(count))
            self.assertEqual(inputs.image_grid_thw.shape[0], count)
            self.assertEqual(metadata["images"], count)
            self.assertLessEqual(metadata["actual_spatial_pixels"], DEFAULT_CONFIG["total_pixels"])
            self.assertLessEqual(metadata["input_tokens"], DEFAULT_CONFIG["max_input_tokens"])
            self.assertEqual(metadata["visual_tokens"], int(inputs.image_grid_thw.prod(-1).sum()) // 4)

    def test_input_is_not_silently_truncated(self):
        self.backend.config["max_input_tokens"] = 5
        with self.assertRaisesRegex(ValueError, "not truncated"):
            self.backend.prepare(self.messages(5))

    def test_real_multimodal_generate_with_tiny_random_qwen3_vl(self):
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        config = Qwen3VLConfig(
            text_config={"vocab_size": 151936, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                         "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
                         "max_position_embeddings": 8192},
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                           "out_hidden_size": 32, "patch_size": 16, "spatial_merge_size": 2,
                           "temporal_patch_size": 2, "num_position_embeddings": 64,
                           "deepstack_visual_indexes": []},
            image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
            vision_end_token_id=151653, eos_token_id=151645, pad_token_id=151643)
        self.backend.model = Qwen3VLForConditionalGeneration(config).eval()
        self.backend.generation_config = copy.deepcopy(self.backend.model.generation_config)
        self.backend.generation_config.do_sample = False
        self.backend.config.update(min_pixels=1024, max_pixels=4096, total_pixels=61440, max_new_tokens=2)
        with patch("torch.cuda.synchronize"), patch("torch.cuda.max_memory_allocated", return_value=0):
            result = self.backend.complete(self.messages(6))
        self.assertEqual(result["observation"]["images"], 6)
        self.assertLessEqual(result["observation"]["completion_tokens"], 2)
        self.assertIsInstance(result["text"], str)


if __name__ == "__main__":
    unittest.main()
