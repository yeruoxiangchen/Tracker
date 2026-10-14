"""Automatic observations: QA isolation, native crop identity, and real Qwen IO."""
import copy
import json
import os
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from mpp_video_qa.core import PROTOCOL, dump_jsonl, file_sha256, stable_key
from mpp_video_qa.data import CachedQADataset, cache_file, qa_collate
from mpp_video_qa.model import CachedVideoQA, assistant_only_preprocess
from mpp_video_qa.region_views import (PROPOSAL_PROTOCOL, REGION_FEATURE_PROTOCOL, compose_features,
    full_replay, json_atomic, plan_views, proposal_file, question_groups, region_file, validate_plan, view_prompt)
from scripts.prepare_region_views import prepare
from scripts.report_auto_regions import JUDGE_PROTOCOL, fingerprint, make_report, paired, phase_status, weight_paths


def example(segment="s1", qid="q1", question="What colour is the car ahead?"):
    paths = ["/frames/%s/%d.jpg" % (segment, i) for i in range(5)]
    row = {"protocol": PROTOCOL, "question_id": qid, "segment_id": segment,
           "question": question, "answers": ["The car is red."], "frames": paths, "source_subset": "action"}
    proposal = {"protocol": PROPOSAL_PROTOCOL, "segment_id": segment, "detector_identity": {"test": True},
                "frames": []}
    for index, path in enumerate(paths):
        proposal["frames"].append({"path": path, "width": 1000, "height": 500,
            "image_sha256": stable_key(segment, index), "candidates": [
                {"label": "car", "score": .9, "box": [.30, .3, .42, .6], "candidate_id": "car"},
                {"label": "traffic light", "score": .8, "box": [.7, .1, .75, .2], "candidate_id": "light"},
                {"label": "car", "score": .1, "box": [.5, .3, .6, .6], "candidate_id": "weak"}]})
    return row, proposal


def make_plan(row, proposal, **kwargs):
    return plan_views(row["question_id"], row["segment_id"], row["question"], row["frames"], proposal, **kwargs)


class RegionViewTests(unittest.TestCase):
    def test_question_routing_without_answer_labels(self):
        row, proposal = example()
        vehicle = make_plan(row, proposal)
        light_row = dict(row, question_id="light", question="What colour is the traffic signal?")
        signal = make_plan(light_row, proposal)
        self.assertEqual(question_groups(light_row["question"]), ["signal"])
        self.assertEqual({v["candidate_id"] for v in vehicle["views"] if v["kind"] == "region"}, {"car"})
        self.assertEqual({v["candidate_id"] for v in signal["views"] if v["kind"] == "region"}, {"light"})
        row["answers"] = ["Changed answer, unknown green truck on the right"]
        self.assertEqual(vehicle, make_plan(row, proposal))

    def test_full_fallback_and_replay_keep_chronology(self):
        row, proposal = example(question="Is the ego vehicle accelerating?")
        no_target = make_plan(row, proposal)
        self.assertEqual(len(no_target["views"]), 5)
        self.assertEqual([v["frame_index"] for v in no_target["views"]], list(range(5)))
        targeted = dict(row, question="Why are you slowing down?")
        plan = make_plan(targeted, proposal, replay=True)
        self.assertEqual(len(plan["views"]), 5)
        self.assertTrue(plan["full_replay"])
        self.assertEqual(full_replay("qid"), full_replay("qid"))
        self.assertTrue(full_replay("qid", 1))
        self.assertFalse(full_replay("qid", 0))

    def test_native_pixel_cache_dedup_and_plan_validation(self):
        row, proposal = example()
        plan = make_plan(row, proposal)
        validate_plan(plan, row)
        other = dict(row, question_id="q2", question="What is the car ahead doing?")
        second = make_plan(other, proposal)
        self.assertNotEqual(plan["plan_sha256"], second["plan_sha256"])
        self.assertEqual(plan["views"][1]["region_key"], second["views"][1]["region_key"])
        pixels = plan["views"][1]["pixel_box"]
        self.assertLessEqual(pixels[0], 300)
        self.assertGreaterEqual(pixels[2], 420)
        corrupt = copy.deepcopy(plan)
        corrupt["views"][1]["frame_index"] = 4
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_plan(corrupt, row)
        with self.assertRaises(ValueError):
            validate_plan(plan, dict(row, question="Different question"))

    def test_masked_candidate_dup_and_roi_have_equal_slots_and_prompt(self):
        row, proposal = example()
        plan = make_plan(row, proposal)
        full = torch.arange(5.).reshape(5, 1, 1).expand(5, 2, 3)
        duplicate = compose_features(full, plan, "duplicate", lambda view: None)
        roi = compose_features(full, plan, "roi", lambda view: torch.full((2, 3), 9.))
        self.assertEqual(tuple(duplicate.shape), (10, 2, 3))
        self.assertEqual(tuple(duplicate.shape), tuple(roi.shape))
        torch.testing.assert_close(duplicate[::2], full)
        torch.testing.assert_close(duplicate[1::2], full)
        torch.testing.assert_close(roi[::2], full)
        self.assertTrue(torch.all(roi[1::2] == 9))
        prompt = view_prompt(row["question"], plan, "roi")
        self.assertEqual(prompt, view_prompt(row["question"], plan, "duplicate"))
        self.assertEqual(prompt.count("<ImageHere>"), 10)
        self.assertEqual(prompt.count("T0"), 2)
        self.assertNotIn('candidate_id', prompt)
        self.assertNotIn('traffic light', prompt)

    def test_dataset_loads_frozen_eva_crops_and_preserves_original_qa(self):
        row, proposal = example()
        plan = make_plan(row, proposal)
        encoder = {"test": "original_encoder"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump_jsonl(root / "data.jsonl", [row])
            dump_jsonl(root / "views.jsonl", [plan])
            full_path = cache_file(root / "full", row["segment_id"])
            full_path.parent.mkdir(parents=True)
            full = torch.zeros(5, 257, 1408, dtype=torch.bfloat16)
            torch.save({"protocol": PROTOCOL, "segment_id": row["segment_id"], "features": full,
                        "feature_stage": "eva", "view": "full", "frame_paths": row["frames"]}, full_path)
            json_atomic(root / "roi/cache_manifest.json", {"encoder_identity": encoder})
            for view in plan["views"]:
                if view["kind"] == "region":
                    path = region_file(root / "roi", view["region_key"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": REGION_FEATURE_PROTOCOL, "region_key": view["region_key"],
                                "pixel_box": view["pixel_box"], "image_sha256": view["image_sha256"],
                                "encoder_identity": encoder,
                                "features": torch.ones(257, 1408, dtype=torch.bfloat16)}, path)
            for condition, expected in (("full", 5), ("duplicate", 10), ("roi", 10)):
                dataset = CachedQADataset(root / "data.jsonl", root / "full", "eva",
                                          region_views=root / "views.jsonl", region_cache_dir=root / "roi",
                                          region_condition=condition)
                item = dataset[0]
                self.assertEqual(item["row"], row)
                self.assertEqual(item["answer"], row["answers"][0])
                self.assertEqual(len(item["features"]), expected)
                self.assertEqual(len(qa_collate([item])["prompts"]), 1)
            view = plan["views"][1]
            path = region_file(root / "roi", view["region_key"])
            payload = torch.load(path)
            payload["pixel_box"][0] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "stale or mismatched"):
                dataset[0]

    def test_prepare_disjoint_splits_and_original_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = {}
            identity = {"test": True}
            for split, segment in (("train", "train_s"), ("dev", "dev_s"), ("eval", "eval_s")):
                row, proposal = example(segment, split)
                originals[split] = [row]
                dump_jsonl(root / "source" / (split + ".jsonl"), [row])
                json_atomic(proposal_file(root / "proposals", segment), proposal)
            dump_jsonl(root / "source/dev_action.jsonl", originals["dev"])
            dump_jsonl(root / "source/dev_scenery.jsonl", [])
            json_atomic(root / "proposals/manifest.json", {"detector_identity": identity})
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(root / "source", root / "proposals", root / "plans", replay_fraction=0)
            self.assertTrue(manifest["qa_values_unchanged"])
            self.assertEqual(manifest["unique_regions"], 15)
            self.assertEqual(manifest["splits"]["train"]["records_with_roi"], 1)
            dump_jsonl(root / "source/eval.jsonl", originals["train"])
            with self.assertRaisesRegex(ValueError, "share segment"):
                prepare(root / "source", root / "proposals", root / "plans_other")

    def test_paired_and_failure_without_progress(self):
        base = [{"question_id": str(i), "segment_id": "s%d" % i, "lingo_judge_correct": x}
                for i, x in enumerate((False, True, False))]
        candidate = [dict(r, lingo_judge_correct=True) for r in base]
        result = paired(base, candidate)
        self.assertEqual(result["fixed"], 2)
        self.assertEqual(result["regressed"], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_atomic(root / "failure_rank_0.json", {"status": "failed", "updated_unix_seconds": 9})
            self.assertEqual(phase_status(root)["rank_progress"][0]["status"], "failed")

    def test_report_requires_complete_dev_and_official_never_selects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(data_dir=root / "data", train_root=root / "train", init_dir=root / "init",
                                   eval_dir=root / "eval", region_cache_dir=root / "crops", branches=["roi"], epochs=2)
            data = {}
            plans = {}
            for split in ("train", "dev", "eval"):
                row, proposal = example(split, split)
                data[split], plans[split] = [row], [make_plan(row, proposal)]
                dump_jsonl(args.data_dir / (split + ".jsonl"), data[split])
                dump_jsonl(args.data_dir / (split + ".views.jsonl"), plans[split])
            encoder = {"test": "encoder"}
            json_atomic(args.region_cache_dir / "cache_manifest.json", {"encoder_identity": encoder})
            for branch, checkpoint in (("frozen", "init"), ("roi", "epoch-01"), ("roi", "epoch-02")):
                adapter, projector, bridge = weight_paths(args, branch, checkpoint)
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text(json.dumps({"epoch": checkpoint}))
                projector.write_text("projector " + checkpoint)
                bridge.write_text("bridge " + checkpoint)
            a, p, q = weight_paths(args, "frozen", "init")
            json_atomic(args.train_root / "roi/run_config.json", {
                "data_sha256": file_sha256(args.data_dir / "train.jsonl"), "epochs": 2,
                "initial_adapter_sha256": fingerprint(a), "initial_projector_sha256": file_sha256(p),
                "initial_qformer_sha256": file_sha256(q), "feature_stage": "eva", "target_mode": "reference",
                "region_identity": {"region_condition": "roi",
                    "region_views_sha256": file_sha256(args.data_dir / "train.views.jsonl"),
                    "dev_region_views_sha256": file_sha256(args.data_dir / "dev.views.jsonl")}})
            def write_scores(name, branch, condition, checkpoint, split, correct):
                a, p, q = weight_paths(args, branch, checkpoint)
                folder = args.eval_dir / (name + "_" + split)
                run = {"data_sha256": file_sha256(args.data_dir / (split + ".jsonl")),
                       "region_views_sha256": file_sha256(args.data_dir / (split + ".views.jsonl")),
                       "region_condition": condition, "region_view_protocol": plans[split][0]["protocol"],
                       "feature_stage": "eva", "task_instruction_version": "legacy_v1", "max_new_tokens": 96,
                       "batch_size": 1, "precision_protocol": "fp32_trainables_v1",
                       "adapter_identity": fingerprint(a), "projector_identity": file_sha256(p),
                       "qformer_adapter_sha256": file_sha256(q), "region_encoder_identity": encoder}
                json_atomic(folder / "run_manifest.json", run)
                json_atomic(folder / "scored/summary.json", {"judge_protocol": JUDGE_PROTOCOL})
                row, plan = data[split][0], plans[split][0]
                result = dict(row, references=row["answers"], prediction="answer", token_f1=.5, normalized_exact=0.,
                              lingo_judge_correct=correct, region_observation={"plan_sha256": plan["plan_sha256"],
                              "condition": condition, "views": 5 if condition == "full" else len(plan["views"])})
                dump_jsonl(folder / "scored/scored_predictions.jsonl", [result])
            for condition in ("full", "duplicate", "roi"):
                write_scores("frozen_" + condition, "frozen", condition, "init", "dev", False)
            write_scores("roi_epoch-01", "roi", "roi", "epoch-01", "dev", True)
            self.assertEqual(make_report(args)["selected"], {})
            write_scores("roi_epoch-02", "roi", "roi", "epoch-02", "dev", True)
            # The unselected epoch's perfect official score must be ignored.
            write_scores("roi_epoch-01", "roi", "roi", "epoch-01", "eval", False)
            write_scores("roi_epoch-02", "roi", "roi", "epoch-02", "eval", True)
            report = make_report(args)
            self.assertEqual(report["selected"]["roi"]["checkpoint"], "epoch-01")
            self.assertNotIn("roi_epoch-02", report["official"])
            self.assertTrue(report["development_complete"])
            adapter = weight_paths(args, "roi", "epoch-01")[0]
            (adapter / "adapter_config.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "different input/weights"):
                make_report(args)


class RealQwenRegionTest(unittest.TestCase):
    def test_actual_qwen_tokenization_ce_gradients_and_generation(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        source = Path(os.environ.get("QWEN_TEST_MODEL", "/tmp/zjr_mpp_fast/checkpoints/ckpt/Qwen7B-chat"))
        if not (source / "config.json").is_file():
            self.skipTest("local Qwen config/tokenizer required")
        torch.set_num_threads(1)
        torch.manual_seed(44)
        tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=False, trust_remote_code=True, local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        config = AutoConfig.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
        config.update(dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                           kv_channels=8, vocab_size=len(tokenizer), bf16=False, fp16=False, fp32=True,
                           use_flash_attn=False, attn_dropout_prob=0., emb_dropout_prob=0.))
        core = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        core.gradient_checkpointing_enable()
        core.config.use_cache = False
        wrapped = get_peft_model(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                              target_modules=["c_attn"], task_type="CAUSAL_LM"))
        backbone = nn.Module()
        backbone.llm_model = wrapped
        backbone.llm_proj = nn.Linear(8, 16)
        # Small trainable stand-in verifies autograd across the EVA boundary;
        # it is not a claim that the full pretrained MPP Q-Former was loaded.
        class Bridge(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(1408, 8)
            def forward(self, features):
                return self.projection(features.mean(dim=2, keepdim=True)).expand(-1, -1, 32, -1)
        backbone.visual_bridge = Bridge()
        backbone.llm_tokenizer = tokenizer
        backbone.max_txt_len = 1536
        backbone.replace_image_string = "<|extra_0|>"
        backbone.replace_image_token_id = tokenizer("<|extra_0|>").input_ids[0]
        import types
        backbone.preprocess = types.MethodType(assistant_only_preprocess, backbone)
        model = CachedVideoQA(backbone, feature_stage="eva").train()
        row, proposal = example(question="Why are you slowing down?")
        plan = make_plan(row, proposal)
        self.assertEqual(len(plan["views"]), 15)
        prompt = view_prompt(row["question"], plan)
        features = torch.randn(1, 15, 257, 1408)
        losses, gradients = [], []
        for fast in (False, True):
            model.zero_grad(set_to_none=True)
            loss = model(features, [row["question"]], [row["answers"][0]], prompts=[prompt], answer_only_logits=fast).loss
            loss.backward()
            losses.append(loss.detach())
            gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=1e-5)
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for name, gradient in gradients[0].items():
            torch.testing.assert_close(gradient, gradients[1][name], atol=1e-5, rtol=1e-4)
        for name in ("backbone.llm_proj.weight", "backbone.visual_bridge.projection.weight"):
            self.assertGreater(float(gradients[1][name].abs().sum()), 0)
        self.assertTrue(any("lora_B" in n and float(g.abs().sum()) > 0 for n, g in gradients[1].items()))
        model.eval()
        output = model.generate(features, [row["question"]], max_new_tokens=2, prompts=[prompt])
        self.assertEqual(len(output), 1)
        self.assertNotIn("Question:", output[0])

    def test_two_rank_variable_views_and_accumulation(self):
        with tempfile.TemporaryDirectory(prefix="region-ddp-") as directory:
            torch.multiprocessing.spawn(region_ddp_worker, args=(directory,), nprocs=2, join=True)
            for rank in (0, 1):
                result = json.loads((Path(directory) / ("rank-%d.json" % rank)).read_text())
                self.assertEqual(result["optimizer_steps"], 2)
                self.assertEqual(result["micro_steps"], 8)
                self.assertEqual(result["checksum_spread"], 0.)
                self.assertGreater(result["projector_delta"], 0)


def region_ddp_worker(rank, directory):
    """Different image counts per rank exercise DDP's actual forward path."""
    import contextlib
    import re
    import types
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    torch.manual_seed(44)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
                                         init_method="file://" + directory + "/rendezvous")
    try:
        core = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=1536,
                               vocab_size=128, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
        llm = get_peft_model(core, LoraConfig(r=2, lora_alpha=4, lora_dropout=0., target_modules=["c_attn"],
                                             task_type="CAUSAL_LM"))
        class Tokenizer:
            im_start_id, im_end_id, eod_id, pad_token_id = 1, 2, 0, 0
            def __call__(self, text):
                pieces = re.findall(r"<\|extra_0\|>|\w+|[^\w\s]", text)
                return SimpleNamespace(input_ids=[7 if p == "<|extra_0|>" else
                    8 + int(stable_key(p)[:4], 16) % 120 for p in pieces])
        backbone = nn.Module()
        backbone.llm_model, backbone.llm_proj = llm, nn.Linear(6, 16)
        backbone.llm_tokenizer, backbone.max_txt_len = Tokenizer(), 1536
        backbone.replace_image_string, backbone.replace_image_token_id = "<|extra_0|>", 7
        backbone.preprocess = types.MethodType(assistant_only_preprocess, backbone)
        raw = CachedVideoQA(backbone, feature_stage="qformer")
        # GPT2's static Boolean causal-mask buffers are identical on both
        # ranks, and old CPU Gloo cannot broadcast Boolean buffers. Actual
        # training still uses Qwen/NCCL and retains its existing buffer policy.
        raw._ddp_params_and_buffers_to_ignore = {name for name, value in raw.named_buffers()
                                                 if value.dtype == torch.bool}
        model = DistributedDataParallel(raw, find_unused_parameters=False, gradient_as_bucket_view=True,
                                        broadcast_buffers=False)
        before = backbone.llm_proj.weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        row, proposal = example(question="Why are you slowing down?")
        plan = make_plan(row, proposal)
        for micro in range(8):
            condition = "full" if (rank + micro) % 2 else "roi"
            views = 5 if condition == "full" else 15
            prompt = view_prompt(row["question"], plan, condition)
            with model.no_sync() if micro % 4 != 3 else contextlib.nullcontext():
                result = model(torch.randn(1, views, 32, 6), [row["question"]], [row["answers"][0]],
                               prompts=[prompt], answer_only_logits=True)
                (result.loss / 4).backward()
            if micro % 4 == 3:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        checksum = torch.stack([p.detach().sum(dtype=torch.float64) for p in model.parameters() if p.requires_grad])
        checksums = [torch.empty_like(checksum) for _ in range(2)]
        torch.distributed.all_gather(checksums, checksum)
        spread = float((checksums[0] - checksums[1]).abs().max())
        json_atomic(Path(directory) / ("rank-%d.json" % rank), {"optimizer_steps": 2, "micro_steps": 8,
            "checksum_spread": spread, "projector_delta": float((backbone.llm_proj.weight - before).abs().max())})
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
