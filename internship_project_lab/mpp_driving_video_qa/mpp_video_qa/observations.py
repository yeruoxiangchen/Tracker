"""Chronological full frames with explicitly paired, fixed-ROI detail views."""
from PIL import Image
from .core import build_user_prompt, TASK_INSTRUCTION

VIEW_PROTOCOL = "chronological_full_plus_center_roi_v1"
# A reproducible, question/answer-independent first teacher view. Not an oracle crop.
DEFAULT_ROI = (0.2, 0.1, 0.8, 0.8)
ANSWER_HINT_PROTOCOL = "full_frames_first_reference_teacher_only_v1"


def teacher_prompt(question, frames, answers, context="view", view="full",
                   instruction=TASK_INSTRUCTION):
    prompt = view_prompt(question, frames, view, instruction)
    if context == "view":
        return prompt
    if context != "answer_hint" or view != "full":
        raise ValueError("answer_hint teachers require full chronological frames")
    if not answers or not str(answers[0]).strip():
        raise ValueError("answer_hint requires a nonempty training reference")
    # The same deterministic reference is used by CachedQADataset for CE replay.
    # Keep it in teacher context, never in the student's generation prefix.
    return (prompt + "\nTraining-only reference answer:\n" + str(answers[0]) +
            "\nAnswer the original question concisely using the video and this reference."
            " Do not mention the reference or these training instructions.")


def open_views(paths, view="full", roi=DEFAULT_ROI):
    images = []
    for path in paths:
        with Image.open(path) as source:
            full = source.convert("RGB")
        images.append(full)
        if view == "full_crop":
            width, height = full.size
            images.append(full.crop((round(roi[0] * width), round(roi[1] * height),
                                     round(roi[2] * width), round(roi[3] * height))))
        elif view != "full":
            raise ValueError("unknown view: %s" % view)
    return images


def view_prompt(question, frames, view="full", instruction=TASK_INSTRUCTION):
    if view == "full":
        return build_user_prompt(question, frames, instruction)
    if view != "full_crop":
        raise ValueError("unknown view")
    slots = []
    for index in range(frames):
        slots.append("Time %d, full frame: <Img><ImageHere></Img>\n"
                     "Same time %d, central detail crop: <Img><ImageHere></Img>" % (index, index))
    return ("\n".join(slots) + "\n" + instruction +
            " Detail crops are additional views of the same moments, not extra time steps."
            " Preserve full-frame context when identifying lanes, signals and ego motion."
            "\nQuestion: " + question)
