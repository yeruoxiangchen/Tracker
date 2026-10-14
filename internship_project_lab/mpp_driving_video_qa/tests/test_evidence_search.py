"""Native-pixel observe loop, label isolation, frozen MPP wiring and reporting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

from mpp_video_qa.core import PROTOCOL, dump_jsonl, file_sha256, load_jsonl
from mpp_video_qa.data import CachedQADataset, cache_file
from mpp_video_qa.evidence_search import (DEFAULT_CONFIG, INPUT_PROTOCOL, SEARCH_PROTOCOL, choose_observed,
    content_hash, crop_request, evidence_plan, parse_action, result_file, search_one, validate_config, validate_input,
    validate_result)
from mpp_video_qa.region_views import (EVIDENCE_VIEW_PROTOCOL, REGION_FEATURE_PROTOCOL, json_atomic,
    region_file, region_runtime_identity, validate_plan, view_prompt)
from scripts.prepare_evidence_search import assemble_views, export_inputs
from scripts.evidence_workflow import JUDGE_PROTOCOL, check_official_mpp_identity, evaluate, freeze, make_report
from scripts.report_auto_regions import fingerprint, summarize
from scripts.search_video_evidence import initialize, worker


def inspect(frame=0, box=None, purpose="Check target relative to ego lane"):
    return {"action": "inspect", "requests": [{"frame_index": frame,
             "box": box or [250, 250, 750, 750], "purpose": purpose}]}


def finish(ids=None):
    return {"action": "finish", "selected_region_ids": ids or [], "reason": "Visible evidence only"}


class FakeBackend:
    def __init__(self, actions):
        self.actions, self.observations = list(actions), []

    def complete(self, messages):
        images = [v["image"] for m in messages if isinstance(m["content"], list)
                  for v in m["content"] if v["type"] == "image"]
        text = "\n".join(m["content"] if isinstance(m["content"], str) else
                         "\n".join(v["text"] for v in m["content"] if v["type"] == "text") for m in messages)
        self.observations.append({"sizes": [im.size for im in images], "text": text,
                                  "pixels": [im.getpixel((0, 0)) for im in images]})
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return {"text": json.dumps(action), "observation": {"finished": True, "images": len(images)}}


class EvidenceSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.data, self.search = self.root / "source", self.root / "data", self.root / "run/search_dev"
        self.source.mkdir()
        self.original = {}
        for split, segment in (("train", "train_seg"), ("dev", "dev_seg"), ("eval", "eval_seg")):
            paths = []
            for index in range(5):
                path = self.root / (segment + "_%d.png" % index)
                picture = Image.new("RGB", (120, 80), (index * 30, 50, 90))
                picture.save(path); picture.close()
                paths.append(str(path))
            row = {"protocol": PROTOCOL, "question_id": split + "_q", "segment_id": segment,
                   "question": "Why should you yield to the vehicle coming from the left?",
                   "frames": paths, "answers": ["ANSWER_SENTINEL not visible to search"], "source_subset": "action"}
            self.original[split] = row
            dump_jsonl(self.source / (split + ".jsonl"), [row])
        export_inputs(self.source, self.data)
        self.row = load_jsonl(self.data / "inputs/dev.jsonl")[0]
        self.policy = {"model": {"path": "fake_backend_not_a_quality_test"}, "config": dict(DEFAULT_CONFIG)}
        self.policy_hash = content_hash(self.policy)
        self.run = {"protocol": SEARCH_PROTOCOL, "split": "dev", "max_records": 0,
                    "input_sha256": file_sha256(self.data / "inputs/dev.jsonl"),
                    "policy_identity": self.policy, "policy_sha256": self.policy_hash}
        json_atomic(self.search / "run_manifest.json", self.run)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, actions):
        backend = FakeBackend(actions)
        result = search_one(self.row, backend, policy_sha256=self.policy_hash)
        validate_result(result, self.row, self.policy_hash, DEFAULT_CONFIG)
        return result, backend

    def store_assemble(self, result):
        json_atomic(result_file(self.search, self.row["question_id"]), result)
        return assemble_views(self.data, self.search, "dev")

    def test_export_excludes_references_and_diagnostic_fields(self):
        self.assertNotIn("ANSWER_SENTINEL", (self.data / "inputs/dev.jsonl").read_text())
        self.assertEqual(load_jsonl(self.data / "original/dev.jsonl"), [self.original["dev"]])
        self.assertNotIn("answers", self.row)
        self.assertEqual(self.row["question"], self.original["dev"]["question"])

    def test_raw_qa_is_rejected_by_search_allowlist(self):
        with self.assertRaisesRegex(ValueError, "allowlist"):
            validate_input(self.original["dev"])
        with self.assertRaises(ValueError):
            validate_input(dict(self.row, teacher_answer="secret"))

    def test_actual_crop_is_returned_before_selection(self):
        result, backend = self.execute([inspect(frame=2), finish(["r001"])])
        self.assertEqual(result["status"], "selected")
        self.assertEqual([len(x["sizes"]) for x in backend.observations], [5, 6])
        self.assertEqual(backend.observations[1]["sizes"][-1], (60, 40))
        self.assertEqual(backend.observations[1]["pixels"][-1], (60, 50, 90))
        self.assertIn("r001 from T2", backend.observations[1]["text"])
        self.assertNotIn("ANSWER_SENTINEL", " ".join(x["text"] for x in backend.observations))
        plan = evidence_plan(self.row, result, self.policy_hash, DEFAULT_CONFIG)
        validate_plan(plan, self.original["dev"])
        self.assertEqual([v["frame_index"] for v in plan["views"]], [0, 1, 2, 2, 3, 4])
        self.assertEqual(len(plan["views"]), 6)
        self.assertNotIn("purpose", json.dumps(plan))
        self.assertNotIn("Visible evidence only", view_prompt(self.row["question"], plan))

    def test_two_rounds_observe_and_final_only_call(self):
        result, backend = self.execute([inspect(0), inspect(4), finish(["r001", "r002"])])
        self.assertEqual(result["status"], "selected")
        self.assertEqual([len(x["sizes"]) for x in backend.observations], [5, 6, 7])
        self.assertTrue(result["trace"][-1]["final_only"])
        self.assertIn("Crop budget is exhausted", backend.observations[-1]["text"])

    def test_valid_no_crop_is_not_execution_failure(self):
        result, backend = self.execute([finish()])
        self.assertEqual(result["status"], "full_only")
        self.assertIsNone(result["error"])
        self.assertEqual(len(backend.observations), 1)

    def test_unobserved_crop_cannot_enter_mpp(self):
        result, _ = self.execute([finish(["r999"])])
        self.assertEqual(result["status"], "failed")
        self.assertIn("unobserved", result["error"]["message"])
        self.assertEqual(len(evidence_plan(self.row, result, self.policy_hash, DEFAULT_CONFIG)["views"]), 5)

    def test_inspect_is_unavailable_on_final_call(self):
        result, _ = self.execute([inspect(0), inspect(1), inspect(2)])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["selected_region_ids"], [])
        self.assertIn("unavailable", result["error"]["message"])

    def test_repeat_crop_does_not_duplicate_view_or_image(self):
        result, backend = self.execute([inspect(0), inspect(0), finish(["r001"])])
        self.assertEqual(len(result["observed_regions"]), 1)
        self.assertEqual([len(x["sizes"]) for x in backend.observations], [5, 6, 6])

    def test_wide_context_crop_is_allowed_not_old_area_heuristic(self):
        result, _ = self.execute([inspect(0, [0, 0, 1000, 900]), finish(["r001"])])
        self.assertEqual(result["status"], "selected")
        self.assertGreater(result["observed_regions"][0]["box"][3], .65)

    def test_coordinate_bounds_and_finite_values(self):
        for box in ([0, 0, 1001, 800], [1, 2, 1, 4], [0, 0, float("nan"), 100], [0, 0, 1, 1]):
            with self.assertRaises(ValueError):
                crop_request(inspect(box=box)["requests"][0], self.row, DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            crop_request(inspect(frame=True)["requests"][0], self.row, DEFAULT_CONFIG)

    def test_selection_total_and_per_frame_budgets(self):
        observed = [{"region_id": "r%d" % i, "frame_index": i // 3} for i in range(6)]
        for ids in (["r0", "r1", "r2"], ["r0"] * 2, ["r%d" % i for i in range(6)]):
            with self.assertRaises(ValueError):
                choose_observed(ids, observed, DEFAULT_CONFIG)

    def test_action_contract_forbids_backend_answer_fields(self):
        with self.assertRaises(ValueError):
            parse_action(json.dumps(dict(finish(), answer="The vehicle is blue")))
        with self.assertRaises(ValueError):
            parse_action("Answer first. " + json.dumps(finish()))
        self.assertEqual(parse_action("```json\n" + json.dumps(finish()) + "\n```"), finish())

    def test_backend_failure_is_recorded_and_counted(self):
        result, _ = self.execute([RuntimeError("CUDA out of memory")])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["trace"]), 1)
        self.assertIn("CUDA", result["error"]["message"])

    def test_truncated_action_is_failure_not_valid_stop(self):
        backend = SimpleNamespace(complete=lambda messages: {"text": json.dumps(finish()),
                                                            "observation": {"finished": False}})
        result = search_one(self.row, backend, policy_sha256=self.policy_hash)
        self.assertEqual(result["status"], "failed")
        self.assertIn("without EOS", result["error"]["message"])

    def test_changed_image_is_not_reused(self):
        Path(self.row["frames"][0]).write_bytes(b"modified")
        result, backend = self.execute([finish()])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(backend.observations), 0)

    def test_result_fingerprint_and_question_identity(self):
        result, _ = self.execute([finish()])
        with self.assertRaises(ValueError):
            validate_result(result, dict(self.row, question="Different"), self.policy_hash, DEFAULT_CONFIG)
        changed = copy.deepcopy(result); changed["reason"] = "modified"
        with self.assertRaises(ValueError):
            validate_result(changed, self.row, self.policy_hash, DEFAULT_CONFIG)

    def test_split_isolation_rejects_overlap(self):
        dump_jsonl(self.source / "eval.jsonl", [dict(self.original["eval"], segment_id="dev_seg")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            export_inputs(self.source, self.root / "other_export")

    def test_assemble_preserves_qa_and_separate_native_catalog(self):
        result, _ = self.execute([inspect(1), finish(["r001"])])
        manifest = self.store_assemble(result)
        self.assertEqual(manifest["unique_regions"], 1)
        self.assertTrue(manifest["qa_values_unchanged"])
        self.assertEqual(load_jsonl(self.data / "dev/data.jsonl"), [self.original["dev"]])
        catalog = load_jsonl(self.data / "dev/regions.jsonl")
        self.assertEqual(catalog[0]["pixel_box"], [30, 20, 90, 60])
        self.assertFalse(manifest["external_text_passed_to_mpp"])

    def test_assemble_does_not_accept_incomplete_search(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            assemble_views(self.data, self.search, "dev")

    def test_new_protocol_wires_into_existing_cached_dataset(self):
        result, _ = self.execute([inspect(2), finish(["r001"])])
        self.store_assemble(result)
        full_cache, crops = self.root / "full_cache", self.root / "crops"
        full_path = cache_file(full_cache, self.row["segment_id"])
        full_path.parent.mkdir(parents=True)
        torch.save({"protocol": PROTOCOL, "segment_id": self.row["segment_id"], "feature_stage": "eva",
                    "frame_paths": self.row["frames"], "view": "full",
                    "features": torch.zeros(5, 257, 1408, dtype=torch.bfloat16)}, full_path)
        identity = {"fixture": "frozen_eva"}
        json_atomic(crops / "cache_manifest.json", {"encoder_identity": identity})
        entry = load_jsonl(self.data / "dev/regions.jsonl")[0]
        region_path = region_file(crops, entry["region_key"]); region_path.parent.mkdir(parents=True)
        torch.save(dict(entry, protocol=REGION_FEATURE_PROTOCOL, encoder_identity=identity,
                        features=torch.ones(257, 1408, dtype=torch.bfloat16)), region_path)
        dataset = CachedQADataset(self.data / "dev/data.jsonl", full_cache, "eva",
                                  region_views=self.data / "dev/views.jsonl", region_cache_dir=crops)
        item = dataset[0]
        self.assertEqual(item["features"].shape, (6, 257, 1408))
        self.assertTrue(torch.all(item["features"][3] == 1))
        self.assertEqual(item["answer"], self.original["dev"]["answers"][0])
        self.assertEqual(item["region_metadata"]["evidence_search"]["status"], "selected")
        self.assertNotIn("ANSWER_SENTINEL", item["prompt"])
        args = SimpleNamespace(region_views=self.data / "dev/views.jsonl", region_cache_dir=crops, region_condition="roi")
        self.assertEqual(region_runtime_identity(args)["region_view_protocol"], EVIDENCE_VIEW_PROTOCOL)
        self.assertEqual(region_runtime_identity(args)["evidence_policy_sha256"], self.policy_hash)

    def test_config_enforces_whole_history_budget(self):
        with self.assertRaises(ValueError):
            validate_config(dict(DEFAULT_CONFIG, total_pixels=10000))
        with self.assertRaises(ValueError):
            validate_config(dict(DEFAULT_CONFIG, max_selected_regions=6))

    def test_resume_completed_question_does_not_load_model(self):
        result, _ = self.execute([finish()]); json_atomic(result_file(self.search, self.row["question_id"]), result)
        args = SimpleNamespace(worker_index=0, workers=1, output_dir=self.search, retry_failed=False, model="unused")
        with patch("mpp_video_qa.evidence_hf.HFSearchBackend", side_effect=AssertionError("must not load")):
            worker(args, [self.row], self.run)
        self.assertEqual(json.loads((self.search / "progress_worker_0.json").read_text())["status"], "completed")

    def test_child_worker_does_not_require_other_shards_to_finish(self):
        from scripts import search_video_evidence
        args = SimpleNamespace(worker_index=1, workers=4)
        with patch.object(search_video_evidence, "parse_args", return_value=args), \
                patch.object(search_video_evidence, "initialize", return_value=([self.row], self.run)), \
                patch.object(search_video_evidence, "worker") as called:
            search_video_evidence.main()
        called.assert_called_once()

    def test_launcher_polls_failed_later_worker_without_waiting_for_first(self):
        from scripts import search_video_evidence
        args = SimpleNamespace(worker_index=-1, workers=2, output_dir=self.root / "launcher")
        first, failed = MagicMock(), MagicMock()
        first.pid, failed.pid = 11, 12
        first.poll.return_value = None
        failed.poll.return_value = 1
        with patch.object(search_video_evidence, "parse_args", return_value=args), \
                patch.object(search_video_evidence, "initialize", return_value=([self.row], self.run)), \
                patch.object(search_video_evidence.subprocess, "Popen", side_effect=[first, failed]), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                patch.object(search_video_evidence.time, "sleep", side_effect=AssertionError("failure must stop immediately")):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                search_video_evidence.main()
        first.terminate.assert_called_once()
        failed.terminate.assert_not_called()

    def test_official_search_requires_explicit_dev_freeze(self):
        # A dummy local weight allows identity construction, without loading a model.
        model = self.root / "model"; model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"fixture identity only")
        args = SimpleNamespace(input=self.data / "inputs/eval.jsonl", model=model, output_dir=self.root / "official",
                               split="eval", workers=1, max_records=0, frozen_config=None,
                               **{k: DEFAULT_CONFIG[k] for k in ("min_pixels", "max_pixels", "total_pixels",
                                                                "max_input_tokens", "max_new_tokens")})
        with self.assertRaisesRegex(ValueError, "configuration frozen"):
            initialize(args)

    def test_paired_report_checks_same_mpp_weights_and_no_external_qa(self):
        result, _ = self.execute([inspect(1), finish(["r001"])])
        manifest = self.store_assemble(result)
        init = self.root / "weights"; (init / "adapter").mkdir(parents=True)
        (init / "adapter/adapter_model.bin").write_bytes(b"lora fixture")
        for filename in ("projector.pt", "qformer_adapter.pt", "bridge.pt"):
            (init / filename).write_bytes(filename.encode())
        a = SimpleNamespace(data_dir=self.data, source_dir=self.source, run_dir=self.search.parent,
                            split="dev", mpp_init=init, bridge=init / "bridge.pt", cache_root=self.root / "crops",
                            include_duplicate=False)
        plan = load_jsonl(self.data / "dev/views.jsonl")[0]
        identities = {"adapter_identity": fingerprint(init / "adapter"), "projector_identity": file_sha256(init / "projector.pt"),
                      "qformer_adapter_sha256": file_sha256(init / "qformer_adapter.pt"),
                      "bridge_init_sha256": file_sha256(a.bridge), "dtype": "bf16", "qformer_mode": "auto"}
        for condition in ("full", "evidence"):
            folder = a.run_dir / ("mpp_%s_dev" % condition)
            kind = "roi" if condition == "evidence" else "full"
            row = {"question_id": self.row["question_id"], "segment_id": self.row["segment_id"],
                   "question": self.row["question"], "references": self.original["dev"]["answers"],
                   "source_subset": "action", "prediction": "answer", "latency_ms": 123.,
                   "region_observation": {"plan_sha256": plan["plan_sha256"], "views": 6 if kind == "roi" else 5,
                                          "condition": kind, "evidence_search": plan["evidence_search"]}}
            dump_jsonl(folder / "predictions.jsonl", [row])
            scored = dict(row, normalized_exact=0., token_f1=.2, lingo_judge_correct=condition == "evidence")
            dump_jsonl(folder / "scored/scored_predictions.jsonl", [scored])
            json_atomic(folder / "scored/summary.json", dict(summarize([scored]), judge_protocol=JUDGE_PROTOCOL))
            json_atomic(folder / "generation_summary.json", {"completed": 1, "total": 1})
            json_atomic(folder / "run_manifest.json", {"data_sha256": manifest["qa_sha256"],
                "region_views_sha256": manifest["views_sha256"], "region_view_protocol": EVIDENCE_VIEW_PROTOCOL,
                "evidence_policy_sha256": self.policy_hash, "region_condition": kind, "feature_stage": "eva",
                "task_instruction_version": "legacy_v1", "max_records": 0, "max_new_tokens": 96, "batch_size": 1,
                "precision_protocol": "fp32_trainables_v1", **identities})
        report = make_report(a)
        self.assertTrue(report["completed"])
        self.assertEqual(report["paired"]["full_vs_evidence"]["fixed"], 1)
        self.assertFalse(report["external_answers_evaluated"])
        self.assertEqual(set(report["models"]), {"full", "evidence"})
        json_atomic(a.cache_root / "dev/cache_summary.json", {"completed": True,
                                                            "catalog_sha256": manifest["catalog_sha256"]})
        for condition in ("full", "evidence"):
            folder = a.run_dir / ("mpp_%s_dev" % condition)
            json_atomic(folder / "scored/input_identity.json", {"judge_protocol": JUDGE_PROTOCOL,
                        "predictions_sha256": file_sha256(folder / "predictions.jsonl")})
        with patch("scripts.evidence_workflow.command", side_effect=AssertionError("complete runs must be reused")):
            self.assertTrue(evaluate(a)["completed"])
        frozen = freeze(a)
        self.assertEqual(frozen["selection_split"], "dev")
        self.assertEqual(frozen["mpp_weight_identity"], identities)
        a.split = "eval"
        with self.assertRaisesRegex(ValueError, "MPP weights"):
            check_official_mpp_identity(a, dict(identities, dtype="fp32"))
        a.split = "dev"
        bad = json.loads((a.run_dir / "mpp_evidence_dev/run_manifest.json").read_text())
        bad["max_new_tokens"] = 95
        json_atomic(a.run_dir / "mpp_evidence_dev/run_manifest.json", bad)
        with self.assertRaisesRegex(ValueError, "changed QA"):
            make_report(a)


if __name__ == "__main__":
    unittest.main()
