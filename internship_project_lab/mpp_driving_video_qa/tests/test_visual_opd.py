import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.opd import (distillation_loss, completion_logits, SharedFrozenTeacher)
from mpp_video_qa.visual_bridge import VisualBridge, VisualLoRALinear
from mpp_video_qa.model import CachedVideoQA, save_projector
from mpp_video_qa.observations import view_prompt
from scripts.prepare_visual_adaptation import stratified_subset
from scripts.prepare_opd_data import select_rows


class VisualOPDTests(unittest.TestCase):
    def test_zero_lora_matches_and_updates(self):
        torch.manual_seed(7)
        base = nn.Linear(12, 8)
        layer = VisualLoRALinear(copy.deepcopy(base), 4, 8)
        x = torch.randn(3, 12)
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
        layer(x).square().mean().backward()
        self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
        self.assertIsNone(layer.base.weight.grad)

    def test_jsd_chunked_value_gradient_and_frozen_teacher(self):
        torch.manual_seed(9)
        student = torch.randn(11, 19, requires_grad=True)
        teacher = torch.randn(11, 19, requires_grad=True)
        loss = distillation_loss(student, teacher, chunk_tokens=3)
        loss.backward()
        actual_gradient = student.grad.clone()
        reference = student.detach().clone().requires_grad_(True)
        ps, pt = F.softmax(reference, -1), F.softmax(teacher.detach(), -1)
        mixture = (ps + pt) / 2
        expected = 0.5 * ((ps * (ps.log() - mixture.log())).sum(-1) +
                          (pt * (pt.log() - mixture.log())).sum(-1)).mean()
        expected.backward()
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_gradient, reference.grad, atol=1e-7, rtol=1e-5)
        self.assertIsNone(teacher.grad)
        self.assertLess(abs(float(distillation_loss(student.detach(), student.detach()))), 1e-6)

    def test_answer_alignment_with_different_prompt_lengths(self):
        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(23, 6)
                self.transformer = types.SimpleNamespace(get_input_embeddings=lambda: self.embedding)

            def forward(self, inputs_embeds, **kwargs):
                n = inputs_embeds.shape[1]
                return types.SimpleNamespace(logits=torch.arange(n).view(1, n, 1).expand(1, n, 23).float())

        class Model:
            def __init__(self, length):
                self.core = Core()
                self.length = length
                self.backbone = types.SimpleNamespace(max_txt_len=100)

            def _embed(self, *args, **kwargs):
                return self.core, torch.zeros(1, self.length, 6), torch.ones(1, self.length), None, None

        ids = torch.tensor([3, 7, 4])
        a = completion_logits(Model(5), torch.zeros(1, 5, 32, 6), "q", ids)
        b = completion_logits(Model(9), torch.zeros(1, 10, 32, 6), "q", ids)
        self.assertEqual(a[:, 0].tolist(), [4, 5, 6])
        self.assertEqual(b[:, 0].tolist(), [8, 9, 10])
        self.assertEqual(a.shape, b.shape)

    def test_subset_preserves_source_mix_and_segment_cap(self):
        rows = [{"question_id": "%s-%d" % (source, i), "source_subset": source,
                 "segment_id": "%s-%d" % (source, i // 5)}
                for source, count in (("action", 80), ("scenery", 40)) for i in range(count)]
        result = stratified_subset(rows, 36, per_segment=2)
        self.assertEqual(sum(r["source_subset"] == "action" for r in result), 24)
        from collections import Counter
        self.assertLessEqual(max(Counter(r["segment_id"] for r in result).values()), 2)
        self.assertEqual(result, stratified_subset(rows, 36, per_segment=2))

    def test_opd_selection_rejects_held_out_and_requires_teacher_wins(self):
        row = {"question_id": "train-1", "segment_id": "video", "question": "q",
               "answers": ["answer"], "source_subset": "action"}
        pred = {"question_id": "train-1", "segment_id": "video", "question": "q",
                "references": ["answer"], "lingo_judge_correct": True}
        result = select_rows([row], [dict(pred, lingo_judge_correct=False)], [pred], 10)
        self.assertEqual(result[0]["answers"], row["answers"])
        self.assertEqual(result[0]["opd_selection"], "teacher_fixed")
        with self.assertRaises(ValueError):
            select_rows([], [pred], [pred], 10)
        with self.assertRaises(ValueError):
            select_rows([row], [pred], [pred], 10)

    def test_crop_prompt_preserves_time(self):
        prompt = view_prompt("Why stop?", 5, "full_crop")
        self.assertEqual(prompt.count("<ImageHere>"), 10)
        self.assertIn("Same time 4", prompt)
        self.assertIn("not extra time steps", prompt)

    def test_shared_teacher_restores_trainable_policy(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from peft import LoraConfig, get_peft_model
        base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=32))
        llm = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"],
                                            task_type="CAUSAL_LM"))
        backbone = nn.Module()
        backbone.llm_model = llm
        backbone.llm_proj = nn.Linear(768, 16)
        backbone.llm_tokenizer = object()
        backbone.replace_image_string = "<image>"
        backbone.replace_image_token_id = 0
        backbone.max_txt_len = 100
        student = CachedVideoQA(backbone, feature_stage="qformer")
        before = {n for n, p in llm.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llm.save_pretrained(root / "adapter")
            save_projector(backbone, root / "projector.pt")
            args = types.SimpleNamespace(teacher_adapter=str(root / "adapter"),
                                         teacher_projector=str(root / "projector.pt"))
            teacher = SharedFrozenTeacher(student, args)
            frozen_projection = teacher.model.backbone.llm_proj.weight.clone()
            with teacher.active():
                self.assertFalse(any(p.requires_grad for p in llm.parameters()))
            self.assertEqual(before, {n for n, p in llm.named_parameters() if p.requires_grad})
            self.assertEqual(llm.active_adapter, "default")
            opt = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=0.1)
            sum(p.square().sum() for p in student.parameters() if p.requires_grad).backward()
            self.assertTrue(all(p.grad is None for n, p in llm.named_parameters() if "opd_teacher" in n))
            opt.step()
            torch.testing.assert_close(teacher.model.backbone.llm_proj.weight, frozen_projection)

    def test_real_qformer_gradient_and_roundtrip(self):
        sys.path.insert(0, str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"))
        from transformers import BertConfig
        from lavis.models.minigpt4qwen_models.Qformer import BertLMHeadModel
        config = BertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        config.encoder_width = 16
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 32
        original = BertLMHeadModel(config).eval()
        original.bert.embeddings.word_embeddings = None
        original.bert.embeddings.position_embeddings = None
        for layer in original.bert.encoder.layer:
            layer.output = layer.intermediate = None
        original.cls = None
        queries = torch.randn(1, 32, 32)
        state = {"Qformer." + k: v for k, v in original.state_dict().items()}
        state.update(query_tokens=queries, **{"ln_vision.weight": torch.ones(16)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.save_pretrained(root / "bert")
            torch.save({"format": "mpp_visual_bridge_init_v1", "vision_width": 16,
                        "state_dict": state}, root / "init.pt")
            bridge = VisualBridge(root / "init.pt", root / "bert", rank=2, alpha=4, trainable=True)
            features = torch.randn(1, 5, 9, 16)
            bridge.eval()
            with torch.no_grad():
                expected = original.bert(query_embeds=queries.expand(5, -1, -1),
                                         encoder_hidden_states=features.reshape(5, 9, 16),
                                         encoder_attention_mask=torch.ones(5, 9),
                                         return_dict=True).last_hidden_state.reshape(1, 5, 32, 32)
            torch.testing.assert_close(bridge(features), expected)
            bridge.train()
            opt = torch.optim.AdamW([p for p in bridge.parameters() if p.requires_grad], lr=1e-3)
            (bridge(features) * torch.randn_like(expected)).mean().backward()
            self.assertGreater(bridge.query_tokens.grad.abs().sum().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in bridge.named_parameters() if "lora_B" in n))
            opt.step()
            bridge.save(root / "adapter.pt")
            restored = VisualBridge(root / "init.pt", root / "bert", weights=root / "adapter.pt")
            bridge.eval()
            torch.testing.assert_close(bridge(features), restored(features))
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

