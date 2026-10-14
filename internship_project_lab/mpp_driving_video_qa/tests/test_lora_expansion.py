"""Actual local Qwen/PEFT migration, new-module training and stateful resume."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM

from mpp_video_qa.lora_expansion import expand_qkv_adapter, scope_identity
from scripts.feedback_opd_common import configure_model
from scripts.report_language_lora_sft import (LABELS, select_checkpoints, build_report,
                                             evaluation_folder, validate_saved_language_adapter)
from scripts.train_feedback_opd import parse_args, training_config, training_probe_hash


class LanguageLoraTests(unittest.TestCase):
    def test_local_qwen_preserves_fp32_then_updates_saves_and_resumes(self):
        source = Path(os.environ.get("QWEN_TEST_MODEL", "/tmp/zjr_mpp_fast/checkpoints/ckpt/Qwen7B-chat"))
        if not (source / "config.json").is_file():
            self.skipTest("set QWEN_TEST_MODEL to the installed local Qwen model directory")
        config = AutoConfig.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
        # Four blocks produce 20 targets and trigger PEFT's suffix compression;
        # the old two-block fixture never exercised the production failure.
        config.update(dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=2, kv_channels=16, vocab_size=128,
                           bf16=False, fp16=False, fp32=True, use_flash_attn=False))
        torch.manual_seed(71)
        base = AutoModelForCausalLM.from_config(config, trust_remote_code=True).bfloat16()
        base_state = {name: value.clone() for name, value in base.state_dict().items()}
        def fresh_base():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).bfloat16()
            model.load_state_dict(base_state)
            return model
        def adapter_state(model):
            return get_peft_model_state_dict(model, save_embedding_layers=False)
        old = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                             target_modules=["c_attn"], task_type="CAUSAL_LM"))
        for name, parameter in old.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
                with torch.no_grad():
                    parameter.uniform_(-.07, .07)
        old.eval()
        inputs = torch.tensor([[1, 12, 27, 5, 8, 3, 19, 24]])
        labels = inputs.clone()
        labels[:, :4] = -100
        with tempfile.TemporaryDirectory(prefix="language-lora-test-") as temporary:
            root = Path(temporary)
            old.save_pretrained(root / "old", save_embedding_layers=False)
            before = old(input_ids=inputs, use_cache=False).logits.detach()
            original = {name: value.clone() for name, value in adapter_state(old).items()}
            expanded = expand_qkv_adapter(fresh_base(), root / "old")
            expanded.eval()
            torch.testing.assert_close(expanded(input_ids=inputs, use_cache=False).logits, before, rtol=0, atol=0)
            report = expanded._lora_expansion_report
            self.assertEqual(report["target_module_count"], 20)
            self.assertTrue(report["inherited_weights_exact"] and report["new_B_zero"])
            actual = adapter_state(expanded)
            for name, value in original.items():
                self.assertTrue(torch.equal(actual[name], value))
            new_B = [name for name in actual if name not in original and ".lora_B." in name]
            self.assertEqual(len(new_B), 16)
            self.assertTrue(all(not p.requires_grad for name, p in expanded.named_parameters() if "lora_" not in name))
            expanded.train()
            optimizer = torch.optim.AdamW([p for p in expanded.parameters() if p.requires_grad], lr=1e-3)
            def update(model, optim):
                optim.zero_grad(set_to_none=True)
                loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optim.step()
            for _ in range(2):
                update(expanded, optimizer)
            trained = adapter_state(expanded)
            self.assertTrue(all(torch.count_nonzero(trained[name]).item() > 0 for name in new_B))
            expanded.save_pretrained(root / "expanded", save_embedding_layers=False)
            torch.save(optimizer.state_dict(), root / "optimizer.pt")
            saved = json.loads((root / "expanded/adapter_config.json").read_text())
            self.assertEqual(set(saved["target_modules"]), {"c_attn", "c_proj", "w1", "w2"})
            self.assertFalse(any("lm_head" in name for name in saved["target_modules"]))
            coverage = validate_saved_language_adapter(root / "expanded", "decoder", num_layers=4)
            self.assertEqual(coverage["module_count"], 20)
            self.assertEqual(coverage["tensor_count"], 40)
            # Normal PEFT loading restores all five module types without expansion.
            resumed = PeftModel.from_pretrained(fresh_base(), root / "expanded", is_trainable=True)
            for name, parameter in resumed.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.float()
            set_peft_model_state_dict(resumed, load_peft_weights(root / "expanded", device="cpu"))
            for name, value in trained.items():
                self.assertTrue(torch.equal(adapter_state(resumed)[name], value))
            resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-3)
            resumed_optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cpu"))
            resumed.train()
            update(expanded, optimizer)
            update(resumed, resumed_optimizer)
            uninterrupted = adapter_state(expanded)
            restarted = adapter_state(resumed)
            for name in uninterrupted:
                torch.testing.assert_close(uninterrupted[name], restarted[name], rtol=0, atol=0)

    def test_ce_does_not_require_teacher_probe_and_resume_does_not_expand_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(initial_dir=temporary, language_lora_scope="decoder_linear",
                                         resume_from="", training_mode="sft_control", output_dir=temporary)
            configure_model(args, True)
            self.assertTrue(args.expand_language_lora)
            with patch("scripts.train_feedback_opd.validate_probe", side_effect=AssertionError("teacher probe called")):
                self.assertIsNone(training_probe_hash(args, {}))
            (Path(temporary) / "run_config.json").write_text(json.dumps({"probe_report_sha256": "legacy-probe"}))
            args.resume_from = str(Path(temporary) / "checkpoint-001000")
            configure_model(args, True)
            self.assertFalse(args.expand_language_lora)
            self.assertEqual(training_probe_hash(args, {}), "legacy-probe")

    def test_expanded_cli_records_ce_and_equal_dev_selection_opportunities(self):
        argv = ["train", "--data", "train.jsonl", "--output-dir", "outputs/new", "--epochs", "2",
                "--training-mode", "sft_control", "--language-lora-scope", "decoder_linear",
                "--teacher-mode", "reference_only", "--checkpoint-steps", "1000", "2500"]
        with patch("sys.argv", argv):
            args = parse_args()
        config = training_config(args, {"language_lora": scope_identity("decoder_linear")},
                                 40000, 4, 5000, None, [1000, 2500])
        self.assertEqual(config["objective"], "assistant_only_ce")
        self.assertIsNone(config["probe_report_sha256"])
        metrics = {branch + "_" + label: {"lingo_judge_accuracy": .55}
                   for branch in ("qkv", "decoder") for label in LABELS}
        metrics["decoder_checkpoint-002500"]["lingo_judge_accuracy"] = .58
        self.assertEqual(select_checkpoints(metrics), {"qkv": "qkv_checkpoint-001000",
                                                       "decoder": "decoder_checkpoint-002500"})

    def test_reports_reuse_historical_results_and_official_cannot_reselect(self):
        from mpp_video_qa.core import dump_jsonl
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = types.SimpleNamespace(data_dir=root / "data", control_eval_dir=root / "history",
                                         eval_dir=root / "new")
            args.eval_dir.mkdir()
            hashes = {key: {"adapter_identity": key} for key in
                      ["parent"] + [branch + "_" + label for branch in ("qkv", "decoder") for label in LABELS]}
            specs = {key: {"branch": key.split("_", 1)[0]} for key in hashes if key != "parent"}
            table = {}
            for split in ("action", "scenery", "eval"):
                rows = [{"question_id": split + str(i), "segment_id": split + str(i),
                         "question": "Which direction?", "answers": ["right"]} for i in range(2)]
                dump_jsonl(args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split), rows)
                for key in hashes:
                    if key == "parent" or key.startswith("qkv_"):
                        prefix = "baseline" if key == "parent" else "sft_control_" + key.split("_", 1)[1]
                        historical = args.control_eval_dir / (prefix + "_" + split) / "scored/summary.json"
                        historical.parent.mkdir(parents=True, exist_ok=True)
                        historical.write_text("{}")
                    folder = evaluation_folder(args, key, split)
                    correct = [True, False]
                    if key in ("qkv_final", "decoder_checkpoint-002500"):
                        correct = [True, True]
                    if key == "decoder_final" or (split == "eval" and key == "decoder_checkpoint-002500"):
                        correct = [False, False]
                    table[folder] = [dict(row, references=row["answers"], normalized_exact=0., token_f1=.4,
                                          lingo_judge_correct=correct[i]) for i, row in enumerate(rows)]
            def loaded(folder, data, identity):
                return table[folder], ("same-generation-protocol",)
            with patch("scripts.report_language_lora_sft.model_specs", return_value=(hashes, specs)), \
                    patch("scripts.report_full_qformer_scale.load_result", side_effect=loaded):
                dev = build_report(args)
                self.assertEqual(dev["selected_checkpoints"], {"qkv": "qkv_final", "decoder": "decoder_checkpoint-002500"})
                self.assertEqual(evaluation_folder(args, "qkv_final", "action"),
                                 args.control_eval_dir / "sft_control_final_action")
                (args.eval_dir / "report.json").write_text(json.dumps(dev))
                official = build_report(args, official=True)
                self.assertEqual(official["selected_checkpoints"], dev["selected_checkpoints"])
                self.assertEqual(official["selected_model"], dev["selected_model"])
                self.assertEqual(official["selected_comparison"]["models"]["decoder_checkpoint-002500"]["lingo_judge_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
