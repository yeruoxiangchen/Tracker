import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path


PROTOCOL = "mpp_driving_video_qa_v1"
SYSTEM_MESSAGE = "You are a helpful visual assistant."
TASK_INSTRUCTION = (
    "The images are chronological frames from a driving video. "
    "Answer the question using only visible evidence. Give a concise, factual answer."
)
DRIVING_PERSPECTIVE_TASK_INSTRUCTION = (
    "The images are chronological frames from a driving video captured by the ego "
    "vehicle's forward-facing camera. In the question, 'you' and 'your' refer to the "
    "ego vehicle or its driver, not the assistant. Answer the question using only "
    "visible evidence. Give a concise, factual answer."
)
TASK_INSTRUCTIONS = {
    "legacy_v1": TASK_INSTRUCTION,
    "driving_perspective_v1": DRIVING_PERSPECTIVE_TASK_INSTRUCTION,
    "driving_complete_v2": (
        "The visual input shows chronological frames captured by the ego vehicle's "
        "forward-facing camera. 'You' and 'your' refer to the ego vehicle or its driver. "
        "Answer the question directly in concise, complete English sentences. "
        "Address every part requested: when asked for an action and its reason, state "
        "both the action and the visible reason, including the relevant object when visible. "
        "Distinguish what the ego vehicle is currently doing from what it should do next. "
        "Describe a change in motion only when supported by the frame sequence. "
        "For a yes/no question, state what your yes/no refers to; explain why when asked. "
        "Use only visible evidence, and do not invent objects or causes to fill an explanation."
    ),
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def build_user_prompt(question, frame_count, task_instruction=TASK_INSTRUCTION):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    slots = "<Img>" + "<ImageHere>" * frame_count + "</Img>\n"
    return slots + str(task_instruction).strip() + "\nQuestion: " + str(question).strip()


def build_generation_prompt(user_prompt, system_message=SYSTEM_MESSAGE):
    return (
        "<|im_start|>system\n" + system_message + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize_answer(value):
    value = str(value).lower().strip()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction, reference):
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_reference_scores(prediction, references):
    references = references if isinstance(references, list) else [references]
    normalized = normalize_answer(prediction)
    exact = max(float(normalized == normalize_answer(ref)) for ref in references)
    f1 = max(token_f1(prediction, ref) for ref in references)
    return exact, f1
