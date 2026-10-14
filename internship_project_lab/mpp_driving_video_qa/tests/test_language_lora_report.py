"""CPU-only checks for condensed PEFT selectors and actual saved A/B coverage."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from mpp_video_qa.lora_expansion import DECODER_LINEAR_TARGETS
from scripts.report_language_lora_sft import validate_saved_language_adapter


class SavedLanguageAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="language-adapter-report-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.config = {"r": 16, "lora_alpha": 32,
                       "target_modules": ["w1", "w2", "c_proj", "c_attn"]}
        self.tensors = {}
        for layer in range(32):
            for tail in DECODER_LINEAR_TARGETS:
                for matrix, shape in (("A", (16, 4)), ("B", (4, 16))):
                    key = "base_model.model.transformer.h.%d.%s.lora_%s.weight" % (layer, tail, matrix)
                    self.tensors[key] = np.zeros(shape, dtype=np.float32)

    def write(self):
        (self.folder / "adapter_config.json").write_text(json.dumps(self.config))
        save_file(self.tensors, str(self.folder / "adapter_model.safetensors"))

    def validate(self, branch="decoder"):
        self.write()
        return validate_saved_language_adapter(self.folder, branch)

    def test_short_full_and_module_type_selectors_cover_all_32_blocks(self):
        full = ["transformer.h.%d.%s" % (layer, tail)
                for layer in range(32) for tail in DECODER_LINEAR_TARGETS]
        for targets in (self.config["target_modules"], full, list(DECODER_LINEAR_TARGETS)):
            with self.subTest(targets_count=len(targets)):
                self.config["target_modules"] = targets
                result = self.validate()
                self.assertEqual(result["module_count"], 160)
                self.assertEqual(result["tensor_count"], 320)

    def test_qkv_control_remains_accepted(self):
        self.config["target_modules"] = ["c_attn"]
        self.tensors = {k: v for k, v in self.tensors.items() if ".attn.c_attn." in k}
        self.assertEqual(self.validate("qkv")["tensor_count"], 64)

    def test_qkv_weights_cannot_masquerade_as_decoder_adapter(self):
        self.tensors = {k: v for k, v in self.tensors.items() if ".attn.c_attn." in k}
        with self.assertRaisesRegex(ValueError, "missing="):
            self.validate()

    def test_missing_matrix_and_missing_whole_layer_are_rejected(self):
        original = dict(self.tensors)
        del self.tensors["base_model.model.transformer.h.31.mlp.c_proj.lora_B.weight"]
        with self.assertRaisesRegex(ValueError, "missing="):
            self.validate()
        self.tensors = {k: v for k, v in original.items() if ".h.31." not in k}
        with self.assertRaisesRegex(ValueError, "missing="):
            self.validate()

    def test_unexpected_weights_are_rejected(self):
        self.tensors["base_model.model.lm_head.lora_A.weight"] = np.zeros((16, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unexpected=.*lm_head"):
            self.validate()

    def test_unmatched_and_incomplete_selectors_are_rejected(self):
        for targets in (["w1", "w2", "c_proj", "c_attn", "lm_head"], ["c_attn"]):
            with self.subTest(targets=targets):
                self.config["target_modules"] = targets
                with self.assertRaisesRegex(ValueError, "selectors do not cover"):
                    self.validate()

    def test_bad_rank_shape_and_alpha_are_rejected(self):
        self.tensors["base_model.model.transformer.h.0.attn.c_attn.lora_A.weight"] = np.zeros((8, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "bad_shapes="):
            self.validate()
        self.config["lora_alpha"] = 16
        with self.assertRaisesRegex(ValueError, "rank 16 / alpha 32"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
