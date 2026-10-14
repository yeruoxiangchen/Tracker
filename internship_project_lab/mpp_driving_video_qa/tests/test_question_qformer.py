import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"))
from mpp_video_qa.visual_bridge import VisualBridge
from mpp_video_qa.question_qformer import (QUESTION_FORMAT, is_text_parameter,
                                          question_identity_from_args, question_sidecar)
from scripts.report_question_qformer_sft import LABELS, select_checkpoints


def fixture(root):
    from transformers import BertConfig
    from lavis.models.minigpt4qwen_models.Qformer import BertLMHeadModel
    config = BertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=64,
                        hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    config.encoder_width, config.add_cross_attention = 16, True
    config.cross_attention_freq, config.query_length = 2, 32
    bert = root / "bert"
    config.save_pretrained(bert)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "which", "direction", "left", "right",
             "red", "green", "vehicle", "light", "is", "the", "turn", "now", "?", "a", "b"]
    vocab += ["unused%d" % i for i in range(32 - len(vocab))]
    (bert / "vocab.txt").write_text("\n".join(vocab) + "\n")
    (bert / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True}))
    original = BertLMHeadModel(config)
    text = {}
    for key, value in original.state_dict().items():
        if is_text_parameter("Qformer." + key):
            normalized = key
            if ".LayerNorm.weight" in key:
                normalized = key.replace(".LayerNorm.weight", ".LayerNorm.gamma")
            elif ".LayerNorm.bias" in key:
                normalized = key.replace(".LayerNorm.bias", ".LayerNorm.beta")
            text[normalized] = value.clone()
    save_file(text, str(bert / "model.safetensors"))
    original.bert.embeddings.word_embeddings = original.bert.embeddings.position_embeddings = None
    original.cls = None
    for layer in original.bert.encoder.layer:
        layer.output = layer.intermediate = None
    state = {"Qformer." + k: v for k, v in original.state_dict().items()}
    state["query_tokens"] = torch.randn(1, 32, 32)
    init = root / "init.pt"
    torch.save({"format": "mpp_visual_bridge_init_v1", "vision_width": 16, "state_dict": state}, init)
    parent = VisualBridge(init, bert, mode="full", trainable=True)
    with torch.no_grad():
        for p in parent.parameters():
            p.add_(.005)
    parent.save(root / "parent.pt")
    return init, bert, parent


class QuestionBridgeTests(unittest.TestCase):
    def test_visual_weights_inherited_and_only_text_modules_from_bert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, parent = fixture(root)
            bridge = VisualBridge(init, bert, weights=root / "parent.pt", mode="full", trainable=True,
                                  text_input="question")
            self.assertTrue(bridge.initialization_report["inherited_visual_weights_exact"])
            self.assertGreater(bridge.initialization_report["text_keys_initialized"], 0)
            for key, tensor in parent.state_dict().items():
                torch.testing.assert_close(bridge.state_dict()[key], tensor, rtol=0, atol=0)
            from safetensors import safe_open
            with safe_open(str(bert / "model.safetensors"), framework="pt") as handle:
                expected = handle.get_tensor("bert.encoder.layer.0.output.LayerNorm.gamma")
            torch.testing.assert_close(bridge.Qformer.bert.encoder.layer[0].output.LayerNorm.weight, expected, atol=0, rtol=0)

    def test_questions_change_features_and_text_branch_receives_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, _ = fixture(root)
            bridge = VisualBridge(init, bert, weights=root / "parent.pt", mode="full", trainable=True,
                                  text_input="question", gradient_checkpointing=True)
            images = torch.randn(1, 5, 9, 16).repeat(2, 1, 1, 1)
            result = bridge(images, ["which direction left ?", "which direction right ?"])
            self.assertEqual(result.shape, (2, 5, 32, 32))
            self.assertFalse(torch.equal(result[0], result[1]))
            (result * torch.randn_like(result)).mean().backward()
            for p in (bridge.Qformer.bert.embeddings.word_embeddings.weight,
                      bridge.Qformer.bert.encoder.layer[0].intermediate.dense.weight,
                      bridge.Qformer.bert.encoder.layer[0].crossattention.self.key.weight):
                self.assertIsNotNone(p.grad)
                self.assertGreater(float(p.grad.abs().sum()), 0)
            # Final text FFN cannot affect retained queries until a subsequent layer.
            last_grad = bridge.Qformer.bert.encoder.layer[-1].intermediate.dense.weight.grad
            self.assertTrue(last_grad is None or last_grad.eq(0).all())

    def test_batch_frame_order_padding_mask_and_original_question_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, _ = fixture(root)
            bridge = VisualBridge(init, bert, weights=root / "parent.pt", mode="full", text_input="question")
            with patch.object(bridge.Qformer.bert, "forward", wraps=bridge.Qformer.bert.forward) as call:
                bridge(torch.randn(2, 3, 7, 16), ["left", "turn right now ?"])
            kwargs = call.call_args.kwargs
            ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
            self.assertEqual(ids.shape[0], 6)
            self.assertTrue(torch.equal(ids[0], ids[1]) and torch.equal(ids[1], ids[2]))
            self.assertTrue(torch.equal(ids[3], ids[4]) and torch.equal(ids[4], ids[5]))
            self.assertFalse(torch.equal(ids[0], ids[3]))
            self.assertTrue(mask[:, :32].eq(1).all())
            self.assertTrue(mask[0, 35:].eq(0).all())
            self.assertFalse(kwargs["is_decoder"])
            with self.assertRaises(ValueError):
                bridge(torch.randn(1, 5, 7, 16), [])

    def test_saved_full_bridge_auto_restores_and_resume_next_update_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, _ = fixture(root)
            bridge = VisualBridge(init, bert, weights=root / "parent.pt", mode="full", trainable=True,
                                  text_input="question")
            images, questions = torch.randn(1, 5, 7, 16), ["which direction ?"]
            target = torch.randn(1, 5, 32, 32)
            optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
            def update(model, opt):
                opt.zero_grad(set_to_none=True)
                (model(images, questions) * target).mean().backward()
                opt.step()
            update(bridge, optimizer)
            bridge.save(root / "question.pt")
            state = copy.deepcopy(optimizer.state_dict())
            payload = torch.load(root / "question.pt")
            self.assertEqual(payload["format"], QUESTION_FORMAT)
            self.assertEqual(set(payload["state_dict"]), set(bridge.state_dict()))
            self.assertTrue(question_sidecar(root / "question.pt").is_file())
            # A trained question checkpoint does not need BERT initializer tensors again.
            (bert / "model.safetensors").unlink()
            restored = VisualBridge(init, bert, weights=root / "question.pt", trainable=True)
            torch.testing.assert_close(bridge(images, questions), restored(images, questions), atol=0, rtol=0)
            resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            resumed_optimizer.load_state_dict(state)
            update(bridge, optimizer)
            update(restored, resumed_optimizer)
            for key, value in bridge.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                VisualBridge(init, bert, weights=root / "question.pt", text_input="none")

    def test_missing_text_weights_and_silent_truncation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, _ = fixture(root)
            bridge = VisualBridge(init, bert, weights=root / "parent.pt", mode="full", text_input="question",
                                  question_max_len=4)
            with self.assertRaisesRegex(ValueError, "silent truncation"):
                bridge(torch.randn(1, 5, 7, 16), ["which direction is the vehicle ?"])
            save_file({"bert.embeddings.word_embeddings.weight": torch.zeros(32, 32)}, str(bert / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                VisualBridge(init, bert, weights=root / "parent.pt", mode="full", text_input="question")

    def test_old_visual_checkpoint_stays_on_old_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, parent = fixture(root)
            restored = VisualBridge(init, bert, weights=root / "parent.pt")
            self.assertFalse(restored.question_input)
            images = torch.randn(1, 5, 7, 16)
            torch.testing.assert_close(parent(images), restored(images), atol=0, rtol=0)
            self.assertIsNone(restored.Qformer.bert.embeddings.word_embeddings)


class QuestionWorkflowTests(unittest.TestCase):
    def test_development_selection_uses_equal_checkpoint_opportunities(self):
        metrics = {branch + "_" + label: {"lingo_judge_accuracy": .5}
                   for branch in ("visual", "question") for label in LABELS}
        metrics["visual_final"]["lingo_judge_accuracy"] = .6
        metrics["question_checkpoint-002500"]["lingo_judge_accuracy"] = .61
        self.assertEqual(select_checkpoints(metrics), {"visual": "visual_final", "question": "question_checkpoint-002500"})
        metrics["question_final"]["lingo_judge_accuracy"] = .61
        self.assertEqual(select_checkpoints(metrics)["question"], "question_checkpoint-002500")

    def test_pure_ce_cli_and_question_contract_identity(self):
        from types import SimpleNamespace
        from scripts.train_feedback_opd import parse_args, training_config
        argv = ["train", "--data", "x", "--output-dir", "y", "--training-mode", "sft_control",
                "--qformer-text-input", "question", "--epochs", "2", "--checkpoint-steps", "1000", "2500"]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        identity = {"question_qformer": {"protocol": "mpp_question_qformer_v1"}}
        config = training_config(args, identity, 40000, 4, 5000, None, [1000, 2500])
        self.assertEqual(config["objective"], "assistant_only_ce")
        self.assertEqual(config["training_mode"], "sft_control")
        self.assertEqual(args.language_lora_scope, "qkv")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init, bert, _ = fixture(root)
            bridge = VisualBridge(init, bert, weights=root / "parent.pt", mode="full", text_input="question")
            bridge.save(root / "question.pt")
            cli = SimpleNamespace(qformer_text_input="auto", qformer_adapter=str(root / "question.pt"))
            self.assertEqual(question_identity_from_args(cli), bridge.question_identity)

    def test_complete_shell_train_command_keeps_formal_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "python-stub"
            output = root / "argv.jsonl"
            stub.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
                            "with Path(%r).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n" % str(output))
            stub.chmod(0o755)
            import os
            env = dict(os.environ, PYTHON_BIN=str(stub), CUDA_VISIBLE_DEVICES="4,5,6,7")
            subprocess.run(["bash", str(ROOT / "scripts/run_question_qformer_sft.sh"), "train"], env=env, check=True)
            calls = [json.loads(line) for line in output.read_text().splitlines()]
            command = calls[-1]
            self.assertIn("--nproc_per_node=4", command)
            for flag, value in (("--training-mode", "sft_control"), ("--epochs", "2"),
                                ("--grad-accumulation", "4"), ("--max-steps", "0"),
                                ("--language-lora-scope", "qkv"), ("--qformer-text-input", "question")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command[command.index("--checkpoint-steps") + 1:command.index("--checkpoint-steps") + 3], ["1000", "2500"])


if __name__ == "__main__":
    unittest.main()
