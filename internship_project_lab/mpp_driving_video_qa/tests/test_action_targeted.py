import random
import unittest
from collections import Counter

from scripts.prepare_action_targeted_sft import balanced_take, tags_for, PARAPHRASES
from scripts.diagnose_action_targeted import paired


class TargetedTest(unittest.TestCase):
    def test_sampling_caps_and_determinism(self):
        rows = [dict(question_id=str(i), segment_id=str(i // 2),
                     question="q" + str(i % 3), group=str(i % 2)) for i in range(30)]
        def sample():
            vc, qc = Counter(), Counter()
            selected = balanced_take(rows, 10, random.Random(42), vc, qc,
                                     1, 4, lambda r: r["group"])
            self.assertLessEqual(max(vc.values()), 1)
            self.assertLessEqual(max(qc.values()), 4)
            self.assertEqual(len({r["question_id"] for r in selected}), 10)
            return selected
        self.assertEqual(sample(), sample())

    def test_insufficient_pool_fails(self):
        with self.assertRaises(ValueError):
            balanced_take([], 1, random.Random(42), Counter(), Counter(), 4, 300, str)

    def test_tags_and_polarity(self):
        self.assertIn("speed", tags_for({"question": "Why are you decelerating?"}))
        self.assertIn("unsafe", PARAPHRASES["Why is it unsafe for you to accelerate in this situation?"])

    def test_paired_counts(self):
        a = {"a": dict(lingo_judge_correct=False, prediction="x"),
             "b": dict(lingo_judge_correct=True, prediction="y")}
        b = {"a": dict(lingo_judge_correct=True, prediction="z"),
             "b": dict(lingo_judge_correct=False, prediction="w")}
        self.assertEqual(paired(a, b), dict(records=2, fixed=1, regressed=1, answer_changed=2))


if __name__ == "__main__":
    unittest.main()
