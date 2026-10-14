import json
import math
from pathlib import Path
import tempfile
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mpp_video_qa.core import PROTOCOL, dump_jsonl, file_sha256
from mpp_video_qa.data import cache_file
from mpp_video_qa.preference import (PREFERENCE_PROTOCOL, REFERENCE_PROTOCOL, PreferenceDataset,
                                    completion_logprob, dpo_loss, encode_completion, read_reference_bundle,
                                    row_identity, sum_token_logps)
from scripts.prepare_preference_data import factual_contrast, select_pairs
from scripts.report_preference import cluster_interval


def fixture():
    rows, predictions = [], []
    for i in range(4):
        row = {"protocol": PROTOCOL, "question_id": str(i), "segment_id": str(i),
               "question": "Which way are you turning?", "answers": ["I am turning left."],
               "frames": ["/test/%d/%d.jpg" % (i, j) for j in range(5)],
               "source_subset": "action" if i % 2 == 0 else "scenery"}
        rows.append(row)
        for j in range(4):
            good = j % 2 == 0 or i == 3
            predictions.append({"question_id": str(i), "segment_id": str(i), "question": row["question"],
                                "references": row["answers"], "sample_id": j, "finished": True,
                                "view": "full", "teacher_context": "view",
                                "prediction": "I am turning %s." % ("left" if good else "right"),
                                "lingo_judge_correct": good, "lingo_judge_logit": 3 if good else -3})
    return rows, predictions


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 6)
        self.output = nn.Linear(6, 11)
        self.transformer = types.SimpleNamespace(get_input_embeddings=lambda: self.embedding)

    def forward(self, inputs_embeds, **kwargs):
        return types.SimpleNamespace(logits=self.output(inputs_embeds.cumsum(1)))


class TinyQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = TinyCore()
        self.projection = nn.Linear(2, 6)
        self.backbone = types.SimpleNamespace(max_txt_len=30)

    def _embed(self, features, questions, answers, **kwargs):
        ids = torch.tensor([[1, 2, 3]])
        prefix = self.core.embedding(ids) + self.projection(features.mean((1, 2))).unsqueeze(1)
        return self.core, prefix, torch.ones_like(ids), None, ids


class PreferenceTests(unittest.TestCase):
    def test_selection_uses_facts_and_preserves_original_inputs(self):
        rows, predictions = fixture()
        selected, candidates, stats = select_pairs(rows, predictions, [], replay_fraction=.25)
        self.assertEqual(stats["pairs"], 3)
        self.assertEqual(stats["replay"], 1)
        self.assertEqual(stats["visually_reviewed_pairs"], 0)
        self.assertEqual(len({r["question_id"] for r in selected}), len(selected))
        for row in selected:
            original = rows[int(row["question_id"])]
            for key in ("question", "frames", "answers"):
                self.assertEqual(row[key], original[key])
            self.assertEqual(row["row_sha256"], row_identity(row))
        self.assertTrue(candidates)

    def test_reject_leaks_duplicates_missing_draws_and_hint(self):
        rows, predictions = fixture()
        for bad in (predictions[:-1], predictions + predictions[:1],
                    [dict(predictions[0], teacher_context="answer_hint")] + predictions[1:]):
            with self.assertRaises(ValueError):
                select_pairs(rows, bad, [])
        with self.assertRaises(ValueError):
            select_pairs(rows, predictions, [rows[0]])
        with self.assertRaises(ValueError):
            select_pairs(rows, predictions, [dict(rows[0], segment_id="renamed")])

    def test_ambiguous_and_negated_facts_not_automatic_pairs(self):
        for chosen, rejected, refs in (("I am not turning left.", "I am turning right.", ["I am turning left."]),
                                       ("I am turning left.", "I am turning right.", ["I am turning left.", "I am turning right."]),
                                       ("I am turning left and turning right.", "I am turning right.", ["I am turning left."])):
            self.assertFalse(factual_contrast(chosen, rejected, refs, "Which way?"))
        self.assertFalse(factual_contrast("It is safe to go.", "I may proceed.", ["I can go."], "What now?"))

    def test_explicit_review_can_exclude_pair(self):
        rows, predictions = fixture()
        _, candidates, _ = select_pairs(rows, predictions, [])
        review = {r["example_id"]: {"approved": False} for r in candidates if r["question_id"] == "0"}
        selected, _, _ = select_pairs(rows, predictions, [], review=review)
        self.assertNotIn("0", {r["question_id"] for r in selected})

    def test_dpo_reference_equality_and_gradient_direction(self):
        chosen = torch.tensor(-4., requires_grad=True)
        rejected = torch.tensor(-6., requires_grad=True)
        reference = torch.tensor(-4., requires_grad=True)
        loss, margin = dpo_loss(chosen, rejected, reference, -6.)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        self.assertEqual(float(margin), 0.)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        self.assertIsNone(reference.grad)
        improved, _ = dpo_loss(torch.tensor(-3.), torch.tensor(-7.), -4., -6.)
        self.assertLess(improved, loss)

    def test_chunked_logps_and_gradients_match_direct_sum(self):
        torch.manual_seed(1)
        logits = torch.randn(7, 13, requires_grad=True)
        targets = torch.tensor([1, 3, 5, 7, 9, 11, 0])
        result = sum_token_logps(logits, targets, 2)
        direct = F.log_softmax(logits, -1).gather(1, targets[:, None]).sum()
        torch.testing.assert_close(result, direct)
        a = torch.autograd.grad(result, logits, retain_graph=True)[0]
        b = torch.autograd.grad(direct, logits)[0]
        torch.testing.assert_close(a, b)

    def test_completion_alignment_and_visual_gradient(self):
        torch.manual_seed(2)
        model = TinyQA()
        features = torch.randn(1, 5, 32, 2)
        completion = [4, 5, 10]
        score = completion_logprob(model, features, "question", completion, 2)
        core, prefix, _, _, _ = model._embed(features, [], [])
        all_inputs = torch.cat((prefix, core.embedding(torch.tensor([[4, 5]]))), 1)
        logits = core(inputs_embeds=all_inputs).logits[0, 2:5]
        expected = F.log_softmax(logits, -1)[torch.arange(3), torch.tensor(completion)].sum()
        torch.testing.assert_close(score, expected)
        (-score).backward()
        self.assertGreater(model.projection.weight.grad.abs().sum(), 0)
        self.assertGreater(model.core.output.weight.grad.abs().sum(), 0)

    def test_eos_once_and_plain_text(self):
        class Tokenizer:
            im_end_id = 10
            def __call__(self, text):
                return types.SimpleNamespace(input_ids=[4, 5])
        self.assertEqual(encode_completion(Tokenizer(), "answer"), [4, 5, 10])
        with self.assertRaises(ValueError):
            encode_completion(Tokenizer(), "answer<|im_end|>")

    def test_cache_identity_rejects_modified_reference(self):
        rows, predictions = fixture()
        selected, _, _ = select_pairs(rows, predictions, [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "train.jsonl"
            dump_jsonl(data, selected)
            for row in selected:
                path = cache_file(root / "visual", row["segment_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": PROTOCOL, "segment_id": row["segment_id"], "feature_stage": "qformer",
                            "frame_paths": row["frames"], "features": torch.zeros(5, 32, 768)}, path)
            dataset = PreferenceDataset(data, root / "visual")
            self.assertEqual(dataset[0]["features"].shape, (5, 32, 768))
            refs = [{"example_id": r["example_id"], "row_sha256": r["row_sha256"]} for r in selected]
            dump_jsonl(root / "reference.jsonl", refs)
            (root / "manifest.json").write_text(json.dumps({"protocol": REFERENCE_PROTOCOL, "data_sha256": file_sha256(data)}))
            (root / "summary.json").write_text(json.dumps({"completed": True, "reference_sha256": file_sha256(root / "reference.jsonl")}))
            self.assertEqual(len(read_reference_bundle(root, data)[2]), len(selected))
            refs[0]["row_sha256"] = "modified"
            dump_jsonl(root / "reference.jsonl", refs)
            with self.assertRaises(ValueError):
                read_reference_bundle(root, data)

    def test_cluster_bootstrap_uses_video_unit(self):
        base = [{"question_id": str(i), "segment_id": str(i // 2), "lingo_judge_correct": False} for i in range(4)]
        candidate = [dict(r, lingo_judge_correct=True) for r in base]
        result = cluster_interval(base, candidate, draws=100)
        self.assertEqual(result["segments"], 2)
        self.assertEqual(result["net_change_95pct_interval"], [1., 1.])


if __name__ == "__main__":
    unittest.main()
