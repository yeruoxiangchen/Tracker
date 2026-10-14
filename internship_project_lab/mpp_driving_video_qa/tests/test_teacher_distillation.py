import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl, PROTOCOL
from mpp_video_qa.distillation import assert_disjoint, select_training_rows, validate_predictions
from mpp_video_qa.data import training_answer, CachedQADataset, cache_file
from scripts.generate_vlm_teacher import messages_for, model_identity
from scripts.teacher_distillation import (prepare_probes, ensure_probes, require_teacher,
    validate_matched, train, teacher_report, prepare_training, build_report)
from scripts.report_full_qformer_continue import identity
from scripts.report_preference import summarize
from scripts.report_answer_opsd import JUDGE_PROTOCOL


def examples(prefix="train", size=12):
    return [dict(question_id=prefix + str(i), segment_id=prefix + "-video-" + str(i // 2),
                 frames=["/%s/%d/%d.jpg" % (prefix, i // 2, frame) for frame in range(5)],
                 source_subset="action" if i < size // 2 else "scenery",
                 question="What is happening %d?" % i, answers=["original-%d" % i]) for i in range(size)]


def predictions(rows, correct):
    return [dict(question_id=r["question_id"], segment_id=r["segment_id"], question=r["question"],
                 references=r["answers"], prediction="teacher-%d" % i,
                 lingo_judge_correct=i in correct, finished=True,
                 normalized_exact=float(i in correct), token_f1=float(i in correct)) for i, r in enumerate(rows)]


class DistillationTests(unittest.TestCase):
    def test_filtered_teacher_targets_replay_and_immutable_references(self):
        rows = examples()
        snapshot = copy.deepcopy(rows)
        student = predictions(rows, {2, 3, 4, 5, 6, 7})
        teacher = predictions(rows, {0, 1, 3, 4, 5, 8})
        teacher[8]["finished"] = False
        selected, manifest = select_training_rows(rows, student, teacher, .5)
        self.assertEqual(manifest["teacher_targets"], 2)
        self.assertEqual(manifest["reference_replay"], 2)
        self.assertEqual(rows, snapshot)
        for row in selected:
            self.assertEqual(training_answer(row), row["answers"][0])
            if row["target_source"] == "teacher":
                self.assertNotEqual(training_answer(row, "distillation"), training_answer(row))
            else:
                self.assertEqual(training_answer(row, "distillation"), training_answer(row))
        self.assertNotIn("train8", {r["question_id"] for r in selected})
        self.assertEqual(selected, select_training_rows(rows, student, teacher, .5)[0])

    def test_held_out_and_wrong_prediction_identity_rejected(self):
        train_rows, dev, evaluation = examples(), examples("dev"), examples("eval")
        assert_disjoint(train_rows, dev, evaluation)
        dev[0]["frames"] = train_rows[0]["frames"]
        with self.assertRaisesRegex(ValueError, "frames"):
            assert_disjoint(train_rows, dev, evaluation)
        scored = predictions(train_rows, {1})
        scored[0]["references"] = ["different answer"]
        with self.assertRaises(ValueError):
            validate_predictions(train_rows, scored)
        with self.assertRaises(ValueError):
            validate_predictions(train_rows, scored[1:])
        with self.assertRaises(ValueError):
            training_answer(dict(train_rows[0], target_source="teacher", teacher_answer="bad"), "distillation")

    def test_teacher_prompt_cannot_see_answers(self):
        row = examples()[0]
        row["answers"] = ["SECRET REFERENCE ANSWER"]
        row["teacher_answer"] = "ANOTHER SECRET"
        messages = messages_for(row, ["image"] * 5, "instruction")
        self.assertNotIn("SECRET", json.dumps(messages))
        self.assertEqual(sum(x["type"] == "image" for x in messages[0]["content"]), 5)

    def test_real_cached_dataset_uses_target_mode_only_in_training(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = dict(examples()[0], distillation_source="train", target_source="teacher", teacher_answer="corrected")
            dump_jsonl(root / "data.jsonl", [row])
            path = cache_file(root, row["segment_id"])
            path.parent.mkdir()
            torch.save({"protocol": PROTOCOL, "segment_id": row["segment_id"], "feature_stage": "eva",
                        "features": torch.zeros(5, 2, 1408)}, path)
            original = CachedQADataset(root / "data.jsonl", root, "eva")[0]
            distill = CachedQADataset(root / "data.jsonl", root, "eva", "distillation")[0]
            self.assertEqual(original["answer"], row["answers"][0])
            self.assertEqual(distill["answer"], "corrected")
            self.assertEqual(original["row"], distill["row"])
            self.assertTrue(torch.equal(original["features"], distill["features"]))

    def test_probe_preparation_is_reproducible_and_preserves_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "data"
            rows = examples(size=24)
            dev, evaluation = examples("dev", 12), examples("eval", 12)
            for name, data in (("train", rows), ("dev", dev), ("eval", evaluation),
                               ("dev_action", dev[:6]), ("dev_scenery", dev[6:])):
                dump_jsonl(source / (name + ".jsonl"), data)
            args = types.SimpleNamespace(source_dir=source, data_dir=output, seed=44,
                                         train_probe_size=12, dev_probe_size=8)
            prepare_probes(args)
            first = file_sha256(output / "train_probe.jsonl")
            ensure_probes(args)
            prepare_probes(args)
            self.assertEqual(first, file_sha256(output / "train_probe.jsonl"))
            selected = load_jsonl(output / "train_probe.jsonl")
            self.assertEqual(sum(r["source_subset"] == "action" for r in selected), 6)
            self.assertTrue(all(row in rows for row in selected))
            args.seed += 1
            with self.assertRaises(ValueError):
                ensure_probes(args)

    def test_unqualified_teacher_cannot_start_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "teacher_report.json").write_text(json.dumps({"teacher_qualified": False}))
            with self.assertRaisesRegex(ValueError, "positive paired"):
                require_teacher(types.SimpleNamespace(output_dir=root))

    def test_matched_budget_detects_different_training_or_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            (parent / "final_adapter").mkdir(parents=True)
            (parent / "final_adapter/weights.bin").write_bytes(b"adapter")
            (parent / "final_projector.pt").write_bytes(b"projector")
            (parent / "final_qformer_adapter.pt").write_bytes(b"qformer")
            data = root / "data"
            dump_jsonl(data / "train.jsonl", examples())
            hashes = identity(parent, final=True)
            config = dict(data_sha256=file_sha256(data / "train.jsonl"), epochs=3, total_steps=30,
                          initial_adapter_sha256=hashes["adapter_identity"],
                          initial_projector_sha256=hashes["projector_identity"],
                          initial_qformer_sha256=hashes["qformer_adapter_sha256"])
            for mode, target in (("control", "reference"), ("distill", "distillation")):
                folder = root / ("train_" + mode)
                folder.mkdir()
                (folder / "run_config.json").write_text(json.dumps(dict(config, target_mode=target)))
                (folder / "summary.json").write_text(json.dumps({"completed": True, "optimizer_steps": 30}))
            args = types.SimpleNamespace(output_dir=root, data_dir=data, parent=parent)
            validate_matched(args)
            (root / "train_distill/run_config.json").write_text(json.dumps(dict(config, target_mode="distillation", epochs=4)))
            with self.assertRaisesRegex(ValueError, "not matched"):
                validate_matched(args)

    def test_shell_status_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(["bash", str(ROOT / "scripts/run_teacher_distillation.sh"), "status",
                                     "--output-dir", temporary], cwd=temporary,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertFalse(value["process_liveness_verified"])
            self.assertFalse(value["report_ready"])

    def test_teacher_report_to_data_and_matched_training_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, teacher_model = root / "parent", root / "teacher_model"
            (parent / "final_adapter").mkdir(parents=True)
            (parent / "final_adapter/weights.bin").write_bytes(b"adapter")
            (parent / "final_projector.pt").write_bytes(b"projector")
            (parent / "final_qformer_adapter.pt").write_bytes(b"qformer")
            teacher_model.mkdir()
            (teacher_model / "config.json").write_text("{}")
            (teacher_model / "pytorch_model.bin").write_bytes(b"teacher")
            args = types.SimpleNamespace(output_dir=root, data_dir=root / "data", source_dir=root / "source",
                parent=parent, teacher_model=teacher_model, teacher_max_pixels=1003520,
                teacher_python="teacher-python", python=sys.executable, cache=root / "cache",
                checkpoints=root / "weights", visual_checkpoints=root / "weights", bridge=root / "bridge",
                upstream=root / "upstream", replay_fraction=.3, seed=44, dry_run=False,
                resume_from="", nproc=4, master_port=29689, epochs=3)
            for split in ("dev", "train"):
                originals = examples(split)
                data = args.data_dir / (split + "_probe.jsonl")
                dump_jsonl(data, originals)
                for role in ("student", "teacher"):
                    folder = root / ("probe_" + split) / role
                    scored = predictions(originals, set() if split == "dev" and role == "student" else
                                         ({2, 3, 4, 5, 6, 7} if role == "student" else set(range(12))))
                    dump_jsonl(folder / "predictions.jsonl", scored)
                    dump_jsonl(folder / "scored/scored_predictions.jsonl", scored)
                    (folder / "scored/summary.json").write_text(json.dumps(dict(summarize(scored), judge_protocol=JUDGE_PROTOCOL)))
                    manifest = dict(data_sha256=file_sha256(data), task_instruction_version="legacy_v1")
                    if role == "student":
                        manifest.update(identity(parent, final=True))
                    else:
                        manifest.update(model=model_identity(teacher_model), references_in_prompt=False,
                                        max_pixels=1003520)
                    (folder / "run_manifest.json").write_text(json.dumps(manifest))
            with patch("scripts.teacher_distillation.run") as runner, patch("builtins.print"):
                report = teacher_report(args)
                self.assertTrue(report["teacher_qualified"])
                prepare_training(args)
                manifest = json.loads((args.data_dir / "train_manifest.json").read_text())
                self.assertEqual(manifest["teacher_targets"], 6)
                train(args, "control")
                control = [str(x) for x in runner.call_args.args[1:]]
                train(args, "distill")
                distill = [str(x) for x in runner.call_args.args[1:]]
                for option in ("--epochs", "--seed", "--data", "--lr", "--qformer-lr", "--grad-accumulation",
                               "--init-adapter", "--projector", "--qformer-adapter"):
                    self.assertEqual(control[control.index(option)+1], distill[distill.index(option)+1])
                self.assertEqual(control[control.index("--target-mode")+1], "reference")
                self.assertEqual(distill[distill.index("--target-mode")+1], "distillation")

    def test_official_results_never_reselect_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = types.SimpleNamespace(output_dir=root, source_dir=root)
            def rows(model, split):
                correct = set(range(12)) if (model == "distill" and split != "eval") or (model == "baseline" and split == "eval") else set()
                return predictions(examples(split), correct), ("same-generation-protocol",)
            def candidate(folder, data, hashes):
                model, split = folder.name.split("_")
                return rows(model, split)
            with patch("scripts.teacher_distillation.validate_matched"), \
                 patch("scripts.teacher_distillation.baseline", side_effect=lambda a,s: rows("baseline", s)), \
                 patch("scripts.report_full_qformer_continue.identity", return_value={}), \
                 patch("scripts.report_full_qformer_scale.load_result", side_effect=candidate), patch("builtins.print"):
                dev = build_report(args)
                self.assertEqual(dev["selected_model"], "distill")
                official = build_report(args, official=True)
                self.assertEqual(official["selected_model"], "distill")
                self.assertGreater(official["combined"]["baseline"]["lingo_judge_accuracy"],
                                   official["combined"]["distill"]["lingo_judge_accuracy"])


if __name__ == "__main__":
    unittest.main()
