import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_dev_failures import comparison, paired_prompt, transform_features


class DevelopmentObservationTests(unittest.TestCase):
    def test_time_controls_keep_shape_and_only_change_observation(self):
        x = torch.arange(30).reshape(5, 2, 3)
        self.assertTrue(torch.equal(transform_features(x, "full_replay"), x))
        self.assertTrue(torch.equal(transform_features(x, "reverse"), x.flip(0)))
        self.assertTrue(torch.equal(transform_features(x, "last_repeat"), x[-1:].repeat(5, 1, 1)))

    def test_roi_and_duplicate_have_same_interleaved_token_budget(self):
        x = torch.arange(30).reshape(5, 2, 3)
        roi = x + 100
        duplicate = transform_features(x, "paired_duplicate")
        paired = transform_features(x, "paired_roi", roi)
        self.assertEqual(duplicate.shape, paired.shape)
        self.assertEqual(paired.shape[0], 10)
        self.assertTrue(torch.equal(duplicate[0::2], x))
        self.assertTrue(torch.equal(duplicate[1::2], x))
        self.assertTrue(torch.equal(paired[0::2], x))
        self.assertTrue(torch.equal(paired[1::2], roi))
        self.assertEqual(paired_prompt("Which vehicle?").count("<ImageHere>"), 10)
        with self.assertRaises(ValueError):
            transform_features(x, "paired_roi", roi[:4])

    def test_comparison_uses_replay_not_historical_score(self):
        original = {"a": {"prediction": "ensure safe", "lingo_judge_correct": False}}
        replay = {"a": {"prediction": "give way", "lingo_judge_correct": True}}
        intervention = {"a": {"prediction": "ensure safe", "lingo_judge_correct": False}}
        cases = {"a": {"case_number": 19}}
        self.assertEqual(comparison(original, replay, cases)["fixed"], 1)
        result = comparison(replay, intervention, cases)
        self.assertEqual(result["fixed"], 0)
        self.assertEqual(result["regressed_cases"], [19])

    def test_roi_comparison_counts_only_shared_subset(self):
        reference = {"a": {"prediction": "car", "lingo_judge_correct": False},
                     "b": {"prediction": "car", "lingo_judge_correct": True}}
        candidate = {"a": {"prediction": "motorcycle", "lingo_judge_correct": True}}
        cases = {"a": {"case_number": 2}, "b": {"case_number": 88}}
        result = comparison(reference, candidate, cases)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["fixed_cases"], [2])
        self.assertEqual(result["regressed"], 0)


if __name__ == "__main__":
    unittest.main()
