import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"))
from mpp_video_qa.visual_bridge import VisualBridge, VisualLoRALinear
from mpp_video_qa.model import build_text_model, save_projector
from scripts.prepare_full_qformer_sft import coverage_subset


class FullQFormerTests(unittest.TestCase):
    def test_dev_report_selects_checkpoint_and_rejects_precision_mismatch(self):
        from mpp_video_qa.core import dump_jsonl, file_sha256
        from scripts.generate_answers import directory_fingerprint
        from scripts.report_full_qformer import main, JUDGE_PROTOCOL
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, evaluation, training, joint = [root / n for n in ("data", "eval", "train", "joint")]
            data.mkdir()
            for model, location in (("joint", joint), ("epoch-01", training / "epoch-01")):
                location.mkdir(parents=True)
                adapter = location / ("final_adapter" if model == "joint" else "adapter")
                adapter.mkdir()
                (adapter / "weights.bin").write_bytes(b"test adapter")
                projection = location / ("final_projector.pt" if model == "joint" else "projector.pt")
                projection.write_bytes(b"test projector")
                if model != "joint":
                    (location / "qformer_adapter.pt").write_bytes(b"test bridge")
                for split in ("action", "scenery"):
                    row = {"question_id": split, "segment_id": split, "question": "q", "answers": ["yes"]}
                    data_path = data / ("dev_%s.jsonl" % split)
                    dump_jsonl(data_path, [row])
                    folder = evaluation / (model + "_" + split)
                    (folder / "scored").mkdir(parents=True)
                    prediction = dict(question_id=split, segment_id=split, question="q", references=["yes"],
                                      lingo_judge_correct=model != "joint", token_f1=1., normalized_exact=1.)
                    dump_jsonl(folder / "scored/scored_predictions.jsonl", [prediction])
                    (folder / "scored/summary.json").write_text(json.dumps({"judge_protocol": JUDGE_PROTOCOL}))
                    run = {"data_sha256": file_sha256(data_path), "adapter_identity": directory_fingerprint(adapter),
                           "projector_identity": file_sha256(projection), "feature_stage": "eva",
                           "precision_protocol": "fp32_trainables_v1",
                           "qformer_adapter_sha256": None if model == "joint" else file_sha256(location / "qformer_adapter.pt")}
                    (folder / "run_manifest.json").write_text(json.dumps(run))
            argv = ["report", "--data-dir", str(data), "--eval-dir", str(evaluation),
                    "--train-dir", str(training), "--joint-dir", str(joint), "--models", "joint", "epoch-01"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((evaluation / "report.json").read_text())
            self.assertEqual(report["selected_model"], "epoch-01")
            self.assertEqual(report["paired"]["epoch-01"]["lingo_judge_correct"]["fixed"], 2)
            path = evaluation / "epoch-01_action/run_manifest.json"
            run = json.loads(path.read_text())
            run["precision_protocol"] = "legacy"
            path.write_text(json.dumps(run))
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                main()

    def test_full_qformer_gradient_checkpoint_and_complete_reload(self):
        from transformers import BertConfig
        from lavis.models.minigpt4qwen_models.Qformer import BertLMHeadModel
        config = BertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        config.encoder_width = 16
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 32
        original = BertLMHeadModel(config)
        original.bert.embeddings.word_embeddings = None
        original.bert.embeddings.position_embeddings = None
        original.cls = None
        for layer in original.bert.encoder.layer:
            layer.output = layer.intermediate = None
        state = {"Qformer." + k: v for k, v in original.state_dict().items()}
        state.update(query_tokens=torch.randn(1, 32, 32))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.save_pretrained(root / "bert")
            torch.save({"format": "mpp_visual_bridge_init_v1", "vision_width": 16,
                        "state_dict": state}, root / "init.pt")
            bridge = VisualBridge(root / "init.pt", root / "bert", mode="full", trainable=True,
                                  gradient_checkpointing=True)
            self.assertTrue(all(p.requires_grad for p in bridge.parameters()))
            self.assertFalse(any(isinstance(m, VisualLoRALinear) for m in bridge.modules()))
            inputs = torch.randn(1, 5, 9, 16)
            target = torch.randn(1, 5, 32, 32)
            opt = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
            before = copy.deepcopy(bridge.state_dict())
            (bridge(inputs) * target).mean().backward()
            self.assertTrue(all(p.grad is not None for p in bridge.parameters()))
            self.assertGreater(bridge.Qformer.bert.encoder.layer[0].crossattention.self.key.weight.grad.abs().sum(), 0)
            self.assertGreater(bridge.Qformer.bert.encoder.layer[0].intermediate_query.dense.weight.grad.abs().sum(), 0)
            opt.step()
            self.assertFalse(torch.equal(before["Qformer.bert.encoder.layer.0.intermediate_query.dense.weight"],
                                         bridge.state_dict()["Qformer.bert.encoder.layer.0.intermediate_query.dense.weight"]))
            bridge.save(root / "full.pt")
            saved = torch.load(root / "full.pt")
            self.assertEqual(set(saved["state_dict"]), set(bridge.state_dict()))
            restored = VisualBridge(root / "init.pt", root / "bert", weights=root / "full.pt")
            self.assertEqual(restored.mode, "full")
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            bridge.eval()
            torch.testing.assert_close(bridge(inputs), restored(inputs), atol=0, rtol=0)
            resumed = VisualBridge(root / "init.pt", root / "bert", mode="full", trainable=True,
                                   weights=root / "full.pt")
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(value, bridge.state_dict()[name], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                VisualBridge(root / "init.pt", root / "bert", mode="lora", weights=root / "full.pt")

    def test_coverage_sampling_adapts_cap_and_preserves_questions(self):
        rows = [{"question_id": "%s-%d" % (source, i), "source_subset": source,
                 "segment_id": "%s-%d" % (source, i // per_video), "question": "original",
                 "answers": [str(i)]}
                for source, total, per_video in (("action", 80, 4), ("scenery", 40, 20))
                for i in range(total)]
        selected, cap = coverage_subset(rows, 90)
        self.assertGreater(cap, 4)
        self.assertEqual(len(selected), 90)
        self.assertEqual(sum(r["source_subset"] == "action" for r in selected), 60)
        self.assertEqual(len({r["segment_id"] for r in selected}), 22)
        self.assertEqual(selected, coverage_subset(rows, 90)[0])
        self.assertTrue(all(r in rows for r in selected))

    def test_projector_precision_same_in_train_and_eval(self):
        from lavis.models.base_model import BaseModel
        class FakeBackbone(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.llm_model = nn.Linear(2, 2)
                self.llm_proj = nn.Linear(768, 2)
                self._last_checkpoint_report = {"matched_key_count": 261}
                for p in self.parameters():
                    p.requires_grad = False
        args = types.SimpleNamespace(upstream_root=str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"),
                                     checkpoint_root="/unused", max_txt_len=1024,
                                     feature_stage="qformer", train_projector=True, train_qformer=False,
                                     precision_protocol="fp32_trainables_v1", dtype="bf16",
                                     gradient_checkpointing=False)
        with tempfile.TemporaryDirectory() as directory:
            original = FakeBackbone(args)
            with torch.no_grad():
                original.llm_proj.weight.fill_(0.100001)
            path = Path(directory) / "projector.pt"
            save_projector(original, path)
            with patch("mpp_video_qa.model.TextBackbone", FakeBackbone), \
                 patch.object(BaseModel, "load_checkpoint"), \
                 patch("peft.get_peft_model", side_effect=lambda model, config: model):
                args.lora_target_modules, args.lora_r, args.lora_alpha, args.lora_dropout = ["c_attn"], 2, 4, 0
                train = build_text_model(args, train=True, projector=path)
                evaluation = build_text_model(args, train=False, projector=path)
            self.assertEqual(train.llm_proj.weight.dtype, torch.float32)
            self.assertEqual(evaluation.llm_proj.weight.dtype, torch.float32)
            torch.testing.assert_close(train.llm_proj.weight, original.llm_proj.weight, atol=0, rtol=0)
            torch.testing.assert_close(evaluation.llm_proj.weight, train.llm_proj.weight, atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
