import copy
import unittest

import torch

from mpp_video_qa.opd_diagnostics import (
    add_gradients, assert_split_isolation, gradient_geometry,
    select_effective_batches, select_trajectories, validate_annotations,
)


def original(qid, source="action"):
    return dict(question_id=qid, segment_id=qid, source_subset=source,
                question="What is the car doing?", answers=["Turning right."],
                frames=["%s_%d.jpg" % (qid, i) for i in range(5)],
                feedback_training_source="train")


def draw(row, correct=False):
    return dict(row, student_token_ids=[9, 10], student_answer="Turning left.",
                student_finished=True, student_judge={"correct": correct, "logit": 1. if correct else -1.},
                sample_id=0, modes={"reference_only": dict(teacher_correction="Turning right.",
                    teacher_judge={"correct": True, "logit": 1.}, teacher_finished=True,
                    eligible=True, gate_reason="eligible")})


class OPDDiagnosticTests(unittest.TestCase):
    def test_effective_batch_geometry_handles_conflict_and_missing_group(self):
        lang, proj = "backbone.llm_model.lora", "backbone.llm_proj.weight"
        result = gradient_geometry({lang: torch.tensor([3., 4.]), proj: torch.tensor([2.])},
                                   {lang: torch.tensor([-3., -4.])}, {lang: 2, proj: 1})
        self.assertAlmostEqual(result["language"]["cosine"], -1.)
        self.assertAlmostEqual(result["language"]["combined_l2_before_clip"], 0.)
        self.assertIsNone(result["projector"]["cosine"])
        self.assertAlmostEqual(result["all"]["combined_l2_before_clip"], 2.)
        self.assertAlmostEqual(result["all"]["combined_global_clip_scale"], .5, places=5)

    def test_accumulate_then_measure_differs_from_average_microbatch_cosines(self):
        name = "backbone.visual_bridge.weight"
        ce, jsd = {}, {}
        for a, b in (([1., 0.], [-1., 0.]), ([-1., 2.], [1., 2.])):
            add_gradients(ce, {name: torch.tensor(a) / 2})
            add_gradients(jsd, {name: torch.tensor(b) / 2})
        result = gradient_geometry(ce, jsd, {name: 2})
        self.assertAlmostEqual(result["qformer"]["cosine"], 1.)
        self.assertAlmostEqual(result["qformer"]["jsd_to_ce_norm_ratio"], 1.)

    def test_replay_uses_last_resume_row_and_complete_all_rank_batch(self):
        row = original("q1")
        logs = {r: [dict(question_id="q1", source_subset="action", micro_step=m,
                        student_token_ids=[r + m, 10], teacher_mode="reference_only",
                        eligible=m == 1, gate_reason="eligible" if m == 1 else "student_correct_ce_only")
                    for m in range(2)] for r in range(2)}
        logs[0].append(dict(logs[0][0], student_token_ids=[99, 10]))
        batch, duplicates = select_effective_batches(logs, {"q1": row}, [1], 2, 2)
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(batch), 4)
        self.assertEqual(batch[0]["student_token_ids"], [99, 10])
        self.assertEqual({b["divisor"] for b in batch}, {4})
        logs[1].pop()
        with self.assertRaisesRegex(ValueError, "incomplete effective batch"):
            select_effective_batches(logs, {"q1": row}, [1], 2, 2)

    def test_controls_are_distinct_correct_and_source_proportional(self):
        originals = {str(i): original(str(i), "action" if i < 6 else "scenery") for i in range(10)}
        error = draw(originals["0"])
        probe = [draw(v, correct=True) for v in originals.values()]
        cases = select_trajectories([error], probe + probe, originals, controls=5)
        controls = cases[1:]
        self.assertEqual(sum(r["source_subset"] == "action" for r in controls), 3)
        self.assertEqual(sum(r["source_subset"] == "scenery" for r in controls), 2)
        self.assertEqual(len({r["case_id"] for r in cases}), 6)
        self.assertTrue(all(not r["eligible"] for r in controls))
        changed = copy.deepcopy(error)
        changed["question"] = "Edited question"
        with self.assertRaisesRegex(ValueError, "changed original QA"):
            select_trajectories([changed], probe, originals)

    def test_isolation_detects_shared_frames_despite_new_question_and_segment(self):
        a, b = original("train"), original("dev")
        self.assertEqual(assert_split_isolation([a], {"dev": [b]})["dev"]["frames"], 0)
        b["frames"][0] = a["frames"][0]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            assert_split_isolation([a], {"dev": [b]})

    def test_factual_labels_require_explicit_review_and_nonoverlapping_positions(self):
        cases = [dict(original("q1"), case_id="q1", student_token_ids=[9, 10])]
        annotation = dict(case_id="q1", reviewed=True, reviewer="reviewer", reference_consistent=True,
                          spans=[dict(label="wrong_fact", positions=[0])])
        self.assertEqual(validate_annotations([annotation], cases), [annotation])
        annotation["spans"].append(dict(label="wording", positions=[0]))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_annotations([annotation], cases)


if __name__ == "__main__":
    unittest.main()
