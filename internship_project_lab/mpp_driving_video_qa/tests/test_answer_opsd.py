import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
import tempfile
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import build_user_prompt, dump_jsonl, file_sha256
from mpp_video_qa.observations import teacher_prompt, ANSWER_HINT_PROTOCOL
from mpp_video_qa.opd import sample_completions
from scripts.prepare_answer_opsd import prepare_rows
from scripts.prepare_opd_data import select_reliable_rows
from scripts.prepare_opd_data import main as select_main
from scripts.report_answer_opsd import paired_metrics


def fixture():
    rows = [{"question_id": str(i), "segment_id": str(i), "question": "Why slow?",
             "frames": ["frame%d" % j for j in range(5)], "answers": ["A pedestrian is crossing."],
             "source_subset": "action" if i < 4 else "scenery"} for i in range(8)]
    student, teacher = [], []
    for i, row in enumerate(rows):
        for k in range(4):
            pred = {"question_id": row["question_id"], "segment_id": row["segment_id"],
                    "question": row["question"], "references": row["answers"], "view": "full",
                    "sample_id": k, "finished": True, "lingo_judge_correct": True}
            teacher.append(dict(pred, teacher_context="answer_hint"))
            student.append(dict(pred, teacher_context="view", lingo_judge_correct=(i % 4 >= 2)))
    return rows, student, teacher


class AnswerOPSDTests(unittest.TestCase):
    def test_hint_is_teacher_only_and_preserves_five_frames(self):
        refs = ["SECRET_REFERENCE", "UNUSED_REFERENCE"]
        s = teacher_prompt("Question?", 5, refs, context="view")
        t = teacher_prompt("Question?", 5, refs, context="answer_hint")
        self.assertEqual(s, build_user_prompt("Question?", 5))
        self.assertNotIn(refs[0], s)
        self.assertIn(refs[0], t)
        self.assertNotIn(refs[1], t)
        self.assertEqual(t.count("<ImageHere>"), 5)
        with self.assertRaises(ValueError):
            teacher_prompt("q", 5, refs, context="answer_hint", view="full_crop")
        with self.assertRaises(ValueError):
            teacher_prompt("q", 5, [], context="answer_hint")

    def test_prepare_excludes_held_out_preserves_qa(self):
        rows, _, _ = fixture()
        prepared = prepare_rows(rows, [], 4)
        for row in prepared:
            expected = dict(rows[int(row["question_id"])], opd_training_source="train")
            self.assertEqual(row, expected)
        with self.assertRaises(ValueError):
            prepare_rows(rows, [rows[0]], 4)

    def test_reliable_selection_mix_replay_and_original_answers(self):
        rows, s, t = fixture()
        selected, report = select_reliable_rows(rows, s, t, 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(r["source_subset"] == "action" for r in selected), 3)
        self.assertEqual(sum(r["opd_selection"] == "both_correct" for r in selected), 2)
        self.assertEqual(report["eligible_teacher_fixed"], 4)
        self.assertEqual(selected, select_reliable_rows(rows, s, t, 6)[0])
        for r in selected:
            self.assertEqual(r["answers"], rows[int(r["question_id"])]["answers"])
            self.assertEqual(r["teacher_success_rate"], 1)

    def test_reject_partial_duplicate_or_foreign_draws(self):
        rows, s, t = fixture()
        for broken in (t[:-1], t + [t[0]], [dict(t[0], question_id="eval")] + t[1:]):
            with self.assertRaises(ValueError):
                select_reliable_rows(rows, s, broken, 6)
        with self.assertRaises(ValueError):
            select_reliable_rows(rows, s[4:], t[4:], 6)

    def test_truncated_or_inconsistent_teacher_not_selected(self):
        rows, s, t = fixture()
        t[0]["finished"] = False
        t[4]["lingo_judge_correct"] = False
        selected, _ = select_reliable_rows(rows, s, t, 6)
        self.assertNotIn("0", {r["question_id"] for r in selected})
        self.assertNotIn("1", {r["question_id"] for r in selected})
        for pred in s:
            pred["lingo_judge_correct"] = True
        with self.assertRaises(ValueError):
            select_reliable_rows(rows, s, t, 6)

    def test_sampling_raw_token_ids_eos_and_prompt(self):
        class FakeCore:
            def generate(self, **kw):
                self.kw = kw
                return torch.cat([kw["input_ids"], torch.tensor([[7, 9, 11]])], 1)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = FakeCore()
                self.backbone = types.SimpleNamespace(llm_tokenizer=types.SimpleNamespace(im_end_id=9, eod_id=0))

            def _embed(self, *args, **kw):
                self.prompt = kw["prompts"]
                return self.core, torch.zeros(1, 6, 2), torch.ones(1, 6), None, torch.ones(1, 6, dtype=torch.long)

        model = FakeModel().train()
        ids, finished = sample_completions(model, torch.zeros(1, 5, 32, 2), "q", 20, "teacher prefix")
        self.assertEqual(ids.tolist(), [7, 9])
        self.assertTrue(finished)
        self.assertTrue(model.training)
        self.assertEqual(model.prompt, ["teacher prefix"])
        self.assertTrue(model.core.kw["do_sample"])
        self.assertEqual(model.core.kw["temperature"], 1)
        self.assertEqual(model.core.kw["top_p"], 1)
        self.assertEqual(model.core.kw["top_k"], 0)

    def test_selection_cli_manifests_and_multisample_scores(self):
        rows, student, teacher = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "probe.jsonl"
            dump_jsonl(data, rows)
            scored_paths = []
            for role, context, predictions in (("student", "view", student), ("teacher", "answer_hint", teacher)):
                folder = root / role
                scored = folder / "scored/scored_predictions.jsonl"
                dump_jsonl(scored, predictions)
                scored_paths.append(str(scored))
                (scored.parent / "summary.json").write_text(json.dumps({
                    "judge_protocol": "official_lower_strip_maxref_logit_gt0_maxlength128_v1"}))
                (folder / "run_manifest.json").write_text(json.dumps({
                    "num_samples": 4, "sampling": True, "teacher_context": context,
                    "view": "full", "data_sha256": file_sha256(data),
                    "answer_hint_protocol": ANSWER_HINT_PROTOCOL if role == "teacher" else None}))
            destination = root / "train.jsonl"
            argv = ["select", "--train", str(data), "--student-scored", scored_paths[0],
                    "--teacher-scored", scored_paths[1], "--output", str(destination),
                    "--max-records", "6", "--num-samples", "4"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                select_main()
            manifest = json.loads(destination.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["records"], 6)
            self.assertEqual(manifest["selected_sha256"], file_sha256(destination))
            self.assertEqual(manifest["probe_sampling"]["questions"], 8)
            self.assertNotIn("probe_paired", manifest)

    def test_paired_report_requires_same_questions_and_counts(self):
        rows, _, _ = fixture()
        base = [dict(r, references=r["answers"], lingo_judge_correct=False, token_f1=0, normalized_exact=0) for r in rows]
        candidate = copy.deepcopy(base)
        candidate[0]["lingo_judge_correct"] = True
        report = paired_metrics(base, candidate)
        self.assertEqual(report["lingo_judge_correct"]["fixed"], 1)
        with self.assertRaises(ValueError):
            paired_metrics(base, candidate[:-1])
        with self.assertRaises(ValueError):
            paired_metrics(base, candidate + candidate[:1])
        candidate[0]["references"] = ["different"]
        with self.assertRaises(ValueError):
            paired_metrics(base, candidate)


if __name__ == "__main__":
    unittest.main()
