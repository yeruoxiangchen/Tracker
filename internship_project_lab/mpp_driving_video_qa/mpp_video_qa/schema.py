import json
from pathlib import Path

from .core import PROTOCOL, stable_key


def _listify(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return list(value) if isinstance(value, (list, tuple)) else [value]


def first_present(row, names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_source_row(row, source, frames_root):
    segment = first_present(row, ("segment_id", "scene_id", "video_id", "segment", "scene"))
    question = first_present(row, ("question", "query", "instruction"))
    question_id = first_present(row, ("question_id", "qid", "id"))
    answers = _listify(first_present(row, ("answers", "answer", "output")))
    frames = _listify(first_present(row, ("frames", "images", "image_paths")))
    if segment is None or question is None or not answers or not frames:
        return None
    frames_root = Path(frames_root)
    normalized_frames = []
    for frame in frames:
        if isinstance(frame, dict):
            frame = frame.get("path") or frame.get("filename")
        if frame is None:
            return None
        frame = Path(str(frame))
        normalized_frames.append(str(frame if frame.is_absolute() else frames_root / frame))
    question_id = question_id or stable_key(segment, question)[:20]
    return {
        "protocol": PROTOCOL,
        "question_id": str(question_id),
        "segment_id": str(segment),
        "source_subset": str(source),
        "frames": normalized_frames,
        "question": str(question).strip(),
        "answers": [str(answer).strip() for answer in answers if str(answer).strip()],
    }

