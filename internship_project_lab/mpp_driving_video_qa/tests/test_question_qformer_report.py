import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mpp_video_qa.core import dump_jsonl, file_sha256
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_preference import summarize
from scripts.report_question_qformer_sft import LABELS, PROTOCOL, build_report, save_report


def report_fixture(root):
    args = SimpleNamespace(data_dir=root / "data", eval_dir=root / "eval",
                           control_eval_dir=root / "history", train_dir=root / "training")
    args.data_dir.mkdir()
    config = {"question_qformer": {"protocol": "mpp_question_qformer_v1", "mode": "question"}}
    models = {"parent": {"adapter_identity": "parent", "projector_identity": "proj", "qformer_adapter_sha256": "parent"}}
    specs = {}
    for branch in ("visual", "question"):
        for label in LABELS:
            key = branch + "_" + label
            models[key] = {"adapter_identity": key, "projector_identity": "proj", "qformer_adapter_sha256": key}
            specs[key] = {"branch": branch, "label": label}
    for split in ("action", "scenery", "eval"):
        rows = [{"question_id": "%s%d" % (split, i), "segment_id": "%s-video%d" % (split, i),
                 "question": "Original question", "answers": ["Original answer"]} for i in range(2)]
        data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        dump_jsonl(data, rows)
        for key, identity in models.items():
            if key == "parent" or key.startswith("visual_"):
                prefix = "baseline" if key == "parent" else "sft_control_" + key.split("_", 1)[1]
                folder = args.control_eval_dir / (prefix + "_" + split)
            else:
                folder = args.eval_dir / (key + "_" + split)
            (folder / "scored").mkdir(parents=True)
            run = {"protocol": "test-language-v1", "data_sha256": file_sha256(data),
                   "cache_dir": "same-cache", "feature_stage": "eva", "task_instruction_version": "legacy_v1",
                   "task_instruction": "same", "precision_protocol": "fp32_trainables_v1", "dtype": "bf16",
                   "max_new_tokens": 96, "max_records": 0, "batch_size": 1, "bridge_init_sha256": "same"}
            run.update(identity)
            if key.startswith("question_"):
                run["question_qformer"] = config["question_qformer"]
            (folder / "run_manifest.json").write_text(json.dumps(run))
            correct = [True, False]
            if key == "visual_final" or key == "question_checkpoint-002500":
                correct = [True, True]
            elif key == "question_checkpoint-001000":
                correct = [False, False]
            if split == "eval" and key == "question_final":
                correct = [True, True]  # Official best is not the development choice.
            predictions = [{"question_id": r["question_id"], "segment_id": r["segment_id"],
                            "question": r["question"], "references": r["answers"], "prediction": "model answer",
                            "normalized_exact": float(c), "token_f1": float(c), "lingo_judge_correct": c}
                           for r, c in zip(rows, correct)]
            dump_jsonl(folder / "predictions.jsonl", predictions)
            dump_jsonl(folder / "scored/scored_predictions.jsonl", predictions)
            summary = dict(summarize(predictions), judge_protocol=JUDGE_PROTOCOL)
            (folder / "scored/summary.json").write_text(json.dumps(summary))
    return args, (models, specs, config)


class QuestionReportTests(unittest.TestCase):
    def test_complete_report_reuses_history_and_official_follows_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec = report_fixture(Path(directory))
            with patch("scripts.report_question_qformer_sft.model_specs", return_value=spec):
                dev = build_report(args)
                self.assertEqual(dev["selected_checkpoints"]["question"], "question_checkpoint-002500")
                save_report(args, dev)
                official = build_report(args, True)
                self.assertEqual(official["selected_checkpoints"], dev["selected_checkpoints"])
                self.assertNotIn("question_final", official["splits"]["eval"]["models"])
                self.assertEqual(official["protocol"], PROTOCOL)
                self.assertIn("parent_correct", official["by_parent_correctness"])

    def test_missing_question_generation_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec = report_fixture(Path(directory))
            path = args.eval_dir / "question_checkpoint-002500_action/run_manifest.json"
            run = json.loads(path.read_text())
            del run["question_qformer"]
            path.write_text(json.dumps(run))
            with patch("scripts.report_question_qformer_sft.model_specs", return_value=spec):
                with self.assertRaisesRegex(ValueError, "question-input bridge"):
                    build_report(args)

    def test_changed_development_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec = report_fixture(Path(directory))
            with patch("scripts.report_question_qformer_sft.model_specs", return_value=spec):
                dev = build_report(args)
                dev["selected_checkpoints"]["question"] = "question_final"
                save_report(args, dev)
                with self.assertRaisesRegex(ValueError, "selection changed"):
                    build_report(args, True)


if __name__ == "__main__":
    unittest.main()
