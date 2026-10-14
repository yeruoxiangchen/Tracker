import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, load_jsonl, file_sha256
from scripts.prepare_full_qformer_sft import main as prepare
from scripts.report_full_qformer_scale import (
    JUDGE_PROTOCOL, build_report, identities, model_specs, select_model, summarize,
)


class FullQFormerScaleTests(unittest.TestCase):
    def test_launcher_uses_full_split_fresh_joint_and_one_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "python-recorder"
            executable.write_text("#!%s\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n" % sys.executable)
            executable.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(executable), SCALE_DATA_DIR=str(root / "all-data"),
                       SCALE_TRAIN_DIR=str(root / "training"), SCALE_EVAL_DIR=str(root / "evaluation"),
                       JOINT_DIR=str(root / "joint"), SFT_EPOCHS="2", QFORMER_MODE="frozen")
            env.pop("SFT_RESUME_FROM", None)
            completed = subprocess.run(["bash", str(ROOT / "scripts/run_full_qformer_scale.sh"), "train"],
                                       env=env, text=True, capture_output=True, check=True)
            calls = [json.loads(line) for line in completed.stdout.splitlines()]
            train = next(c for c in calls if "scripts/train_sft.py" in c)
            for key, expected in (("--data", str(root / "all-data/train.jsonl")),
                                  ("--output-dir", str(root / "training")), ("--epochs", "1"),
                                  ("--qformer-mode", "full"),
                                  ("--init-adapter", str(root / "joint/final_adapter"))):
                self.assertEqual(train[train.index(key) + 1], expected)
            self.assertNotIn("--resume-from", train)
            self.assertNotIn("--qformer-adapter", train)

    def test_all_train_keeps_every_original_and_fixed_holdouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            source.mkdir()
            rows = [{"question_id": str(i), "segment_id": "train-video", "source_subset": "action",
                     "question": "original %d" % i, "answers": [str(i)], "frames": ["frame.jpg"]}
                    for i in range(30)]
            dump_jsonl(source / "train.jsonl", rows)
            for split in ("dev", "dev_action", "dev_scenery", "eval"):
                dump_jsonl(source / (split + ".jsonl"), [dict(rows[0], question_id=split, segment_id=split)])
            argv = ["prepare", "--source-dir", str(source), "--output-dir", str(target), "--all-train"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                prepare()
                prepare()  # Idempotent, including videos with >11 questions.
            self.assertEqual(load_jsonl(target / "train.jsonl"), rows)
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["records"], 30)
            self.assertIsNone(manifest["adaptive_max_qa_per_segment"])
            self.assertEqual(manifest["sampling"], "all_original_training_rows")
            for split in ("dev", "dev_action", "dev_scenery", "eval"):
                self.assertEqual(file_sha256(source / (split + ".jsonl")), file_sha256(target / (split + ".jsonl")))

    def make_comparison(self, root):
        args = types.SimpleNamespace(**{k: root / k for k in
            ("data_dir", "train_dir", "eval_dir", "joint_dir", "reference_train_dir", "reference_eval_dir")})
        args.data_dir.mkdir()
        specs = model_specs(args)
        for model, (evaluation, prefix, weights, original) in specs.items():
            weights.mkdir(parents=True)
            adapter = weights / ("final_adapter" if original else "adapter")
            adapter.mkdir()
            (adapter / "weights.bin").write_bytes(model.encode())
            (weights / ("final_projector.pt" if original else "projector.pt")).write_bytes(b"projection")
            if not original:
                (weights / "qformer_adapter.pt").write_bytes(b"bridge")
            for split in ("action", "scenery"):
                data = args.data_dir / ("dev_%s.jsonl" % split)
                row = {"question_id": split, "segment_id": split, "question": "q", "answers": ["yes"]}
                dump_jsonl(data, [row])
                folder = evaluation / (prefix + "_" + split)
                (folder / "scored").mkdir(parents=True)
                correct = model == "full" or (model == "qformer100k" and split == "action")
                prediction = dict(question_id=split, segment_id=split, question="q", references=["yes"],
                                  prediction="yes" if correct else "no")
                dump_jsonl(folder / "predictions.jsonl", [prediction])
                score = dict(prediction, lingo_judge_correct=correct, token_f1=float(correct), normalized_exact=float(correct))
                dump_jsonl(folder / "scored/scored_predictions.jsonl", [score])
                (folder / "scored/summary.json").write_text(json.dumps(dict(summarize([score]), judge_protocol=JUDGE_PROTOCOL)))
                run = dict(identities(specs[model]), data_sha256=file_sha256(data), protocol="mpp_driving_video_qa_v1",
                           cache_dir="/same/cache", feature_stage="eva", precision_protocol="fp32_trainables_v1",
                           task_instruction_version="legacy_v1", task_instruction="same instruction", dtype="bf16",
                           max_new_tokens=96, max_records=0, batch_size=1, bridge_init_sha256="same")
                (folder / "run_manifest.json").write_text(json.dumps(run))
        return args

    def test_three_model_report_compares_against_current_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_comparison(Path(temporary))
            report = build_report(args, ("joint", "qformer100k", "full"), ("action", "scenery"))
            self.assertEqual(report["selected_model"], "full")
            paired = report["dev_combined"]["paired"]
            self.assertEqual(paired["qformer100k_vs_full"]["lingo_judge_correct"]["fixed"], 1)
            self.assertEqual(paired["joint_vs_full"]["lingo_judge_correct"]["fixed"], 2)
            self.assertEqual(select_model({m: {"lingo_judge_accuracy": .5} for m in ("full", "joint", "qformer100k")}), "qformer100k")

    def test_reuse_rejects_wrong_precision_or_stale_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_comparison(Path(temporary))
            path = args.reference_eval_dir / "epoch-02_action/run_manifest.json"
            run = json.loads(path.read_text())
            run["precision_protocol"] = "legacy"
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "protocol"):
                build_report(args, ("joint", "qformer100k", "full"), ("action", "scenery"))
            run["precision_protocol"] = "fp32_trainables_v1"
            path.write_text(json.dumps(run))
            pred = path.parent / "predictions.jsonl"
            rows = load_jsonl(pred)
            rows[0]["prediction"] = "new unscored prediction"
            dump_jsonl(pred, rows)
            with self.assertRaisesRegex(ValueError, "different generated"):
                build_report(args, ("joint", "qformer100k", "full"), ("action", "scenery"))

    def test_official_results_do_not_participate_in_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_comparison(Path(temporary))
            folder = args.eval_dir / "epoch-01_eval/scored"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text("deliberately invalid official result")
            result = build_report(args, ("joint", "qformer100k", "full"), ("action", "scenery"))
            self.assertEqual(result["selected_model"], "full")
            self.assertNotIn("eval", result["splits"])

    def test_resume_keeps_original_initialization_identity(self):
        from scripts.train_sft import retain_initialization_provenance
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = {"initial_adapter_sha256": "original-language", "initial_projector_sha256": "original-projector",
                      "initial_qformer_sha256": None}
            (root / "run_config.json").write_text(json.dumps(origin))
            config = {"initial_qformer_sha256": "resumed-bridge"}
            retain_initialization_provenance(config, root / "checkpoint-002000")
            for k, v in origin.items():
                self.assertEqual(config[k], v)


if __name__ == "__main__":
    unittest.main()
