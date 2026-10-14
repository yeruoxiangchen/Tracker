import json

from mpp_video_qa.core import (DRIVING_PERSPECTIVE_TASK_INSTRUCTION,
                               best_reference_scores, build_user_prompt,
                               normalize_answer, token_f1)
from mpp_video_qa.schema import normalize_source_row


def test_prompt_has_one_slot_per_frame():
    prompt = build_user_prompt("What is happening?", 5)
    assert prompt.count("<ImageHere>") == 5
    assert "time interval" not in prompt.lower()


def test_driving_perspective_prompt_clarifies_pronouns_without_changing_question():
    question = "What are you currently doing and why?"
    prompt = build_user_prompt(question, 5, DRIVING_PERSPECTIVE_TASK_INSTRUCTION)
    assert prompt.count("<ImageHere>") == 5
    assert "ego vehicle or its driver" in prompt
    assert "not the assistant" in prompt
    assert prompt.endswith("Question: " + question)


def test_answer_metrics_support_multiple_references():
    exact, f1 = best_reference_scores("The car is slowing down.",
                                      ["A vehicle stopped", "car is slowing down"])
    assert exact == 1.0
    assert f1 == 1.0
    assert normalize_answer("The car!") == "car"
    assert token_f1("red car", "car") > 0


def test_normalize_source_row():
    row = normalize_source_row(
        {"segment_id": "s1", "question_id": "q1", "question": "Where?",
         "answer": "Ahead", "images": ["a.jpg", "b.jpg"]},
        "action", "/dataset")
    assert row["answers"] == ["Ahead"]
    assert row["frames"] == ["/dataset/a.jpg", "/dataset/b.jpg"]
