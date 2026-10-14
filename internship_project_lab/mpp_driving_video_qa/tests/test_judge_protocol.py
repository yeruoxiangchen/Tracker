"""Regression for official LingoQA preprocessing and multi-reference decisions."""
from types import SimpleNamespace
import unittest

import torch
from scripts.evaluate_answers import judge_score


class Tokens(dict):
    def to(self, device):
        return self


class Tokenizer:
    cls_token = "[CLS]"

    def __call__(self, texts, **kwargs):
        self.texts, self.options = texts, kwargs
        return Tokens(input_ids=torch.zeros((len(texts), 1), dtype=torch.long))


class Model(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.logits = logits

    def forward(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor(self.logits).reshape(-1, 1))


class JudgeProtocolTest(unittest.TestCase):
    def test_official_preprocessing_and_best_reference(self):
        tokenizer = Tokenizer()
        score, correct = judge_score(tokenizer, Model([-2.0, 0.25]),
                                     "What COLOR?", [" Red\n", " GREEN "], " Green \n")
        self.assertEqual(tokenizer.texts, [
            "[CLS]\nQuestion: What COLOR?\nAnswer: red\nStudent: green",
            "[CLS]\nQuestion: What COLOR?\nAnswer: green\nStudent: green"])
        self.assertEqual(tokenizer.options["max_length"], 128)
        self.assertTrue(tokenizer.options["truncation"])
        self.assertEqual(score, 0.25)
        self.assertTrue(correct)

    def test_zero_is_not_correct(self):
        self.assertFalse(judge_score(Tokenizer(), Model([0.0]), "Q", ["A"], "A")[1])


if __name__ == "__main__":
    unittest.main()
