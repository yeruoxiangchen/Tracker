"""CPU Grounding-DINO architecture/API check in the modern detector environment.

Random tiny local weights test only the executable interface, not detection quality.
The Python-3.8 MPP environment skips this test without installing a new Transformers.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl
from mpp_video_qa.region_views import proposal_file


class GroundedDetectorTests(unittest.TestCase):
    def test_real_grounding_dino_forward_postprocess_and_resume(self):
        try:
            from transformers import (BertConfig, BertTokenizer, GroundingDinoConfig,
                GroundingDinoForObjectDetection, GroundingDinoImageProcessor, GroundingDinoProcessor, SwinConfig)
        except ImportError:
            self.skipTest("run in the independent modern detector environment")
        import torch
        from PIL import Image
        torch.set_num_threads(1)
        bert = ROOT.parent / "mpp_llava_video_workspace/checkpoints/ckpt/bert-base-uncased"
        if not (bert / "vocab.txt").is_file():
            self.skipTest("local BERT vocabulary required")
        tokenizer = BertTokenizer.from_pretrained(str(bert), local_files_only=True)
        with tempfile.TemporaryDirectory(prefix="region-detector-api-") as directory:
            root = Path(directory)
            backbone = SwinConfig(image_size=64, embed_dim=32, depths=[1, 1], num_heads=[2, 4],
                                  window_size=4, out_features=["stage2"], out_indices=[2])
            text = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                              num_attention_heads=4, intermediate_size=64)
            config = GroundingDinoConfig(backbone_config=backbone, text_config=text, d_model=32,
                        encoder_layers=1, decoder_layers=1, encoder_attention_heads=4, decoder_attention_heads=4,
                        encoder_ffn_dim=64, decoder_ffn_dim=64, num_queries=4, num_feature_levels=1,
                        disable_custom_kernels=True)
            model = GroundingDinoForObjectDetection(config)
            model.save_pretrained(root / "model")
            image_processor = GroundingDinoImageProcessor(size={"shortest_edge": 64, "longest_edge": 64})
            processor = GroundingDinoProcessor(image_processor, tokenizer)
            processor.save_pretrained(root / "model")
            paths = []
            for index in range(2):
                path = root / ("frame-%d.png" % index)
                Image.new("RGB", (100, 60), (50 + index, 90, 130)).save(path)
                paths.append(str(path))
            dump_jsonl(root / "data.jsonl", [{"segment_id": "sample", "frames": paths,
                       "question_id": "q", "question": "ignored by detector", "answers": ["also ignored"]}])
            command = [sys.executable, str(ROOT / "scripts/propose_regions.py"),
                       "--data", str(root / "data.jsonl"), "--output-dir", str(root / "proposals"),
                       "--model", str(root / "model"), "--device", "cpu", "--box-threshold", "0.0"]
            env = dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            value = json.loads(proposal_file(root / "proposals", "sample").read_text())
            self.assertEqual([f["path"] for f in value["frames"]], paths)
            self.assertTrue(all(f["image_sha256"] for f in value["frames"]))
            self.assertEqual(len(value["frames"]), 2)
            for frame in value["frames"]:
                for candidate in frame["candidates"]:
                    self.assertTrue(all(0 <= x <= 1 for x in candidate["box"]))
            # Complete cached proposals should not load the model on a rerun.
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertNotIn('"status": "loading_detector"', result.stdout)


if __name__ == "__main__":
    unittest.main()
