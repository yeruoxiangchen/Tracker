"""Answer-free, budgeted visual evidence search, independent of the MPP runtime.

The backend sees the question, original frames, and returned native-pixel crops.
Only selected image regions (not backend prose) enter the downstream MPP model.
This is a video adapter/policy, not a reproduction of ZoomEye or an OPD loss.
"""
import json
import math
import time
from collections import Counter
from pathlib import Path

from .core import file_sha256, stable_key
from .region_views import EVIDENCE_VIEW_PROTOCOL, REGION_FEATURE_PROTOCOL, normalized_box


SEARCH_PROTOCOL = "driving_external_vlm_evidence_search_v1"
INPUT_PROTOCOL = "driving_answer_free_evidence_input_v1"
INPUT_KEYS = {"protocol", "question_id", "segment_id", "question", "frames",
              "frame_metadata", "source_subset"}
DEFAULT_CONFIG = {
    "crop_rounds": 2, "requests_per_round": 5, "max_selected_regions": 5,
    "max_regions_per_frame": 2, "min_crop_side": 8,
    "min_pixels": 50176, "max_pixels": 200704, "total_pixels": 2007040,
    "max_input_tokens": 6144, "max_new_tokens": 512,
}
SYSTEM_PROMPT = """You select visual evidence for a separate driving-video QA model. Do NOT answer the question.
Read the complete question and inspect all five chronological forward-camera frames T0..T4.
'You/your' in the question means the ego vehicle/driver. Select observations needed to distinguish
the question's actual target, its relation to the ego lane/other objects, and any relevant change.
Do not substitute a keyword, the most conspicuous object, or an assumed target for the full question.
You may request contextual regions containing several objects, road layout, signs or signals.
For motion questions compare the relevant frames; a crop from one time is not a new time step.
The originals are always retained. If a crop does not reveal reliable relevant detail, discard it.
If full frames suffice, finish with no selected regions. More crops are not intrinsically better.

Return ONE JSON object, without Markdown, an answer, or additional fields:
{"action":"inspect","requests":[{"frame_index":0,"box":[100,200,500,600],"purpose":"What visible detail to check"}]}
or {"action":"finish","selected_region_ids":["r001"],"reason":"Why these observed views are useful, or why none is reliable"}.
Every box is [left,top,right,bottom] in ORIGINAL full-frame coordinates normalized to 0..1000,
not coordinates within a returned crop. Frame index must be 0..4. Requested crops will be returned
as images labeled with their region IDs, source frame and actual original-frame box.
Finish may select ONLY IDs whose crop images have actually been returned, not imagined regions.
At most five selected crops total and two per source frame. There are two crop-request rounds,
at most five requests each, followed by a final selection-only call. You can finish earlier.
"""


def content_hash(value):
    return stable_key(json.dumps(value, sort_keys=True, ensure_ascii=False))


def result_file(directory, question_id):
    key = stable_key(question_id)
    return Path(directory) / "results" / key[:2] / (key + ".json")


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("unknown/missing evidence-search settings")
    if any(type(value) is not int or value <= 0 for value in config.values()):
        raise ValueError("search budgets must be positive integers")
    # These limits are also part of the fixed policy and downstream view contract.
    if (config["crop_rounds"] != 2 or config["requests_per_round"] > 5 or
            config["max_selected_regions"] > 5 or config["max_regions_per_frame"] > 2):
        raise ValueError("policy supports two rounds, <=5 requests/selected crops, <=2 per frame")
    if config["min_pixels"] > config["max_pixels"]:
        raise ValueError("minimum image budget exceeds maximum")
    image_count = 5 + config["crop_rounds"] * config["requests_per_round"]
    if config["total_pixels"] < image_count * config["min_pixels"]:
        raise ValueError("total pixel budget cannot fit the maximum observation history")


def validate_input(row):
    if set(row) != INPUT_KEYS or row["protocol"] != INPUT_PROTOCOL:
        raise ValueError("search accepts only the exported question/frame allowlist, never source QA")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError("question is empty")
    if len(row["frames"]) != 5 or len(row["frame_metadata"]) != 5:
        raise ValueError("the five original chronological frames are required")
    for index, (path, meta) in enumerate(zip(row["frames"], row["frame_metadata"])):
        if (set(meta) != {"frame_index", "width", "height", "image_sha256"} or
                meta["frame_index"] != index or meta["width"] <= 0 or meta["height"] <= 0 or
                not Path(path).is_absolute()):
            raise ValueError("invalid original-frame provenance")
    return row


def parse_action(text):
    value = str(text).strip()
    # Accept a single fenced JSON object, but not embedded commentary/answers.
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4].strip()
    action = json.loads(value)
    if not isinstance(action, dict):
        raise ValueError("tool action must be a JSON object")
    if action.get("action") == "inspect":
        if set(action) != {"action", "requests"} or not isinstance(action["requests"], list):
            raise ValueError("inspect requires only action and requests")
    elif action.get("action") == "finish":
        if (set(action) != {"action", "selected_region_ids", "reason"} or
                not isinstance(action["selected_region_ids"], list) or
                not isinstance(action["reason"], str)):
            raise ValueError("finish requires observed region IDs and a reason, not an answer")
    else:
        raise ValueError("unknown evidence-search action")
    return action


def crop_request(request, row, config):
    if set(request) != {"frame_index", "box", "purpose"}:
        raise ValueError("crop request must contain frame_index, box and purpose only")
    index = request["frame_index"]
    if type(index) is not int or not 0 <= index < 5:
        raise ValueError("crop frame index must be 0..4")
    box = request["box"]
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000 for x in box) or
            not box[0] < box[2] or not box[1] < box[3]):
        raise ValueError("crop box must be nonempty original coordinates in 0..1000")
    if not isinstance(request["purpose"], str) or not 0 < len(request["purpose"]) <= 400:
        raise ValueError("a short observation purpose is required")
    meta = row["frame_metadata"][index]
    w, h = meta["width"], meta["height"]
    pixels = [math.floor(box[0] * w / 1000), math.floor(box[1] * h / 1000),
              math.ceil(box[2] * w / 1000), math.ceil(box[3] * h / 1000)]
    if min(pixels[2] - pixels[0], pixels[3] - pixels[1]) < config["min_crop_side"]:
        raise ValueError("requested crop is too small in original pixels")
    return {"frame_index": index, "pixel_box": pixels, "box": normalized_box(pixels, w, h),
            "image_sha256": meta["image_sha256"], "frame_path": row["frames"][index],
            "region_key": stable_key(REGION_FEATURE_PROTOCOL, meta["image_sha256"], json.dumps(pixels)),
            "purpose": request["purpose"]}


def choose_observed(ids, observed, config):
    if any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("selected region IDs must be distinct strings")
    by_id = {item["region_id"]: item for item in observed}
    if len(ids) > config["max_selected_regions"] or not set(ids) <= set(by_id):
        raise ValueError("selection exceeds budget or refers to an unobserved crop")
    selected = [by_id[x] for x in ids]
    if any(n > config["max_regions_per_frame"] for n in Counter(x["frame_index"] for x in selected).values()):
        raise ValueError("selection exceeds the per-frame crop budget")
    return selected


def search_one(row, backend, config=None, policy_sha256="", on_event=None):
    """Run the real observe/crop/observe loop; backend.complete returns text+metadata.

    Backend prose stays in the trace. Failed executions explicitly fall back to
    original frames, and are counted separately from a valid no-crop decision.
    Images are PIL objects; each returned crop is made from the original pixels.
    """
    from PIL import Image
    config = dict(DEFAULT_CONFIG if config is None else config)
    validate_config(config)
    validate_input(row)
    started = time.monotonic()
    images, crops, observed, trace, selected = [], [], [], [], []
    status, reason, error = "failed", "", None
    messages = []
    try:
        content = []
        for index, path in enumerate(row["frames"]):
            meta = row["frame_metadata"][index]
            if file_sha256(path) != meta["image_sha256"]:
                raise ValueError("original frame content changed after export")
            with Image.open(path) as source:
                image = source.convert("RGB")
            images.append(image)
            if image.size != (meta["width"], meta["height"]):
                raise ValueError("original frame dimensions changed")
            content.extend([{"type": "text", "text": "T%d, original frame %dx%d, chronological order:" %
                             (index, image.width, image.height)}, {"type": "image", "image": image}])
        content.append({"type": "text", "text": "Run budgets: " + json.dumps(config, sort_keys=True) +
                        "\nOriginal question (do not answer it): " + row["question"]})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        for call in range(config["crop_rounds"] + 1):
            final_only = call == config["crop_rounds"]
            if final_only:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Crop budget is exhausted. Return finish only. Select only useful "
                     "already observed region IDs, or [] if none is reliable. Available IDs: " +
                     json.dumps([x["region_id"] for x in observed])}]})
            if on_event:
                on_event("selecting_final_evidence" if final_only else "observing", call=call + 1,
                         observed_crops=len(observed))
            entry = {"call": call + 1, "final_only": final_only, "response": None,
                     "model_observation": {}, "returned_regions": []}
            trace.append(entry)
            response = backend.complete(messages)
            entry.update(response=response["text"], model_observation=response.get("observation", {}))
            if entry["model_observation"].get("finished") is False:
                raise ValueError("tool JSON generation hit its token budget without EOS")
            action = parse_action(response["text"])
            entry["action"] = action
            if action["action"] == "finish":
                selected = choose_observed(action["selected_region_ids"], observed, config)
                status = "selected" if selected else "full_only"
                reason = action["reason"]
                break
            if final_only:
                raise ValueError("inspect is unavailable after the two crop rounds")
            requests = action["requests"]
            if not 1 <= len(requests) <= config["requests_per_round"]:
                raise ValueError("invalid number of requested crops")
            feedback = [{"type": "text", "text": "Tool result: these are ACTUAL native-pixel crops. "
                         "Boxes refer to their original full frames. Inspect the images before selecting; "
                         "discard wrong-target/irrelevant/uninformative crops."}]
            for request in requests:
                item = crop_request(request, row, config)
                existing = next((r for r in observed if r["region_key"] == item["region_key"] and
                                 r["frame_index"] == item["frame_index"]), None)
                if existing:
                    feedback.append({"type": "text", "text": "Already returned as " + existing["region_id"]})
                    continue
                item["region_id"] = "r%03d" % (len(observed) + 1)
                crop = images[item["frame_index"]].crop(tuple(item["pixel_box"]))
                crops.append(crop)
                observed.append(item)
                entry["returned_regions"].append(item["region_id"])
                feedback.extend([{"type": "text", "text": "%s from T%d, original box %s/1000; crop %dx%d:" %
                                  (item["region_id"], item["frame_index"],
                                   json.dumps([round(x * 1000, 3) for x in item["box"]]), crop.width, crop.height)},
                                 {"type": "image", "image": crop}])
            if not entry["returned_regions"]:
                feedback.append({"type": "text", "text": "No new crop was requested. Inspect the already "
                                 "returned images and finish if further crops would not help."})
            messages.extend([{"role": "assistant", "content": response["text"]},
                             {"role": "user", "content": feedback}])
    except Exception as exc:
        # This is an observable failed tool execution, not a positive no-crop choice.
        selected = []
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        for picture in images + crops:
            picture.close()
    result = {"protocol": SEARCH_PROTOCOL, "question_id": row["question_id"],
              "segment_id": row["segment_id"], "input_sha256": content_hash(row),
              "policy_sha256": policy_sha256, "status": status, "reason": reason, "error": error,
              "observed_regions": observed, "selected_region_ids": [x["region_id"] for x in selected],
              "trace": trace, "elapsed_seconds": time.monotonic() - started}
    result["result_sha256"] = content_hash(result)
    return result


def validate_result(result, row, policy_sha256, config):
    if (result.get("protocol") != SEARCH_PROTOCOL or result.get("question_id") != row["question_id"] or
            result.get("segment_id") != row["segment_id"] or result.get("input_sha256") != content_hash(row) or
            result.get("policy_sha256") != policy_sha256 or
            result.get("result_sha256") != content_hash({k: v for k, v in result.items() if k != "result_sha256"})):
        raise ValueError("search result identity differs from frozen input/policy")
    if result["status"] not in ("selected", "full_only", "failed"):
        raise ValueError("unknown search completion state")
    selected = choose_observed(result["selected_region_ids"], result["observed_regions"], config)
    if bool(selected) != (result["status"] == "selected"):
        raise ValueError("search status does not agree with final evidence")
    if (result["status"] == "failed") != (result["error"] is not None):
        raise ValueError("search failure must be explicit")
    returned = [i for t in result["trace"] for i in t["returned_regions"]]
    observed = result["observed_regions"]
    if (len(result["trace"]) > config["crop_rounds"] + 1 or
            len(observed) > config["crop_rounds"] * config["requests_per_round"] or
            len({x["region_id"] for x in observed}) != len(observed) or
            returned != [x["region_id"] for x in observed]):
        raise ValueError("observation history exceeds budget or lacks returned crop provenance")
    for item in observed:
        # Normalized coordinates are rounded; validate exact pixels instead of recropping.
        index = item["frame_index"]
        if type(index) is not int or not 0 <= index < 5:
            raise ValueError("observed region has invalid original-frame index")
        meta = row["frame_metadata"][index]
        pixels = item["pixel_box"]
        if (len(pixels) != 4 or any(type(x) is not int for x in pixels) or
                not 0 <= pixels[0] < pixels[2] <= meta["width"] or
                not 0 <= pixels[1] < pixels[3] <= meta["height"] or
                item["image_sha256"] != meta["image_sha256"] or item["frame_path"] != row["frames"][index] or
                item["box"] != normalized_box(pixels, meta["width"], meta["height"]) or
                item["region_key"] != stable_key(REGION_FEATURE_PROTOCOL, meta["image_sha256"], json.dumps(pixels))):
            raise ValueError("observed region does not describe the original native crop")
    return selected


def evidence_plan(row, result, policy_sha256, config):
    selected = validate_result(result, row, policy_sha256, config)
    views = []
    for index in range(5):
        views.append({"kind": "full", "frame_index": index})
        for item in selected:
            if item["frame_index"] != index:
                continue
            # No purpose, reasoning, answer or transcript is placed in the MPP input.
            view = {k: item[k] for k in ("frame_index", "pixel_box", "box", "image_sha256", "region_key")}
            view.update(kind="region", candidate_id=item["region_id"])
            views.append(view)
    plan = {"protocol": EVIDENCE_VIEW_PROTOCOL, "question_id": row["question_id"],
            "segment_id": row["segment_id"], "question_sha256": stable_key(row["question"]),
            "frame_paths": row["frames"], "frame_sha256": [x["image_sha256"] for x in row["frame_metadata"]],
            "frame_sizes": [[x["width"], x["height"]] for x in row["frame_metadata"]],
            "focus_groups": [], "full_replay": False, "views": views,
            "candidate_provenance": {"protocol": SEARCH_PROTOCOL, "policy_sha256": policy_sha256},
            "evidence_search": {"status": result["status"], "result_sha256": result["result_sha256"]}}
    plan["plan_sha256"] = stable_key(json.dumps(plan, sort_keys=True))
    return plan
