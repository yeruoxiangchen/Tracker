"""Question-conditioned candidate views; no reference answers enter selection.

Detection scores and geometric preferences propose observations, not correct
answers or ground-truth object identities.  Full frames remain in every plan.
"""
import json
import math
import re
from pathlib import Path

from .core import TASK_INSTRUCTION, build_user_prompt, stable_key


PROPOSAL_PROTOCOL = "driving_grounding_dino_candidates_v1"
REGION_VIEW_PROTOCOL = "driving_question_candidate_views_v1"
EVIDENCE_VIEW_PROTOCOL = "driving_vlm_evidence_views_v1"
REGION_FEATURE_PROTOCOL = "driving_original_crop_eva_v1"
DETECTOR_LABELS = (
    "car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light",
    "traffic sign", "traffic cone", "bollard", "road barrier",
)
CONDITIONS = ("full", "duplicate", "roi")


def json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def proposal_file(directory, segment_id):
    key = stable_key(segment_id)
    return Path(directory) / key[:2] / (key + ".json")


def region_file(directory, key):
    return Path(directory) / key[:2] / (key + ".pt")


def label_group(label):
    value = str(label).lower()
    if "traffic light" in value:
        return "signal"
    if "sign" in value:
        return "sign"
    if any(word in value for word in ("car", "truck", "bus", "motorcycle", "bicycle", "van", "vehicle")):
        return "vehicle"
    if any(word in value for word in ("person", "pedestrian")):
        return "person"
    if any(word in value for word in ("cone", "bollard", "barrier", "obstacle")):
        return "obstacle"
    return "other"


def question_groups(question):
    """A transparent routing vocabulary, not an answer or a learned grounder."""
    q = str(question).lower()
    # The forward camera does not observe the ego car as an external vehicle.
    q = re.sub(r"\b(ego|our|your|host)\s+(vehicle|car)\b", "ego", q)
    groups = set()
    if re.search(r"\b(vehicles?|cars?|trucks?|bus|buses|vans?|motorcycl\w*|bicycl\w*|cyclists?|follow\w*|overtak\w*)\b", q):
        groups.add("vehicle")
    if re.search(r"\b(pedestrian\w*|person|people|someone|walking)\b", q):
        groups.add("person")
    if re.search(r"\b(traffic lights?|signals?|light colour|light color)\b", q):
        groups.add("signal")
    if re.search(r"\b(signs?|signpost\w*|speed limit\w*|road marking\w*)\b", q):
        groups.add("sign")
    if re.search(r"\b(obstacle\w*|bollard\w*|barrier\w*|cone\w*|roadworks?|construction)\b", q):
        groups.add("obstacle")
    if not groups and re.search(r"\b(why|reason|hazard\w*|avoid\w*|yield\w*)\b", q):
        groups.update(("vehicle", "person", "signal", "sign", "obstacle"))
    return sorted(groups)


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-12, aa + bb - intersection)


def normalized_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError("invalid detector box")
    result = [max(0.0, min(1.0, float(value) / scale))
              for value, scale in zip(box, (width, height, width, height))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("empty detector box")
    return [round(x, 6) for x in result]


def expand_box(box, width, height, padding=0.2):
    """Return actual integer source pixels; cache keys describe exactly this crop."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    pixels = [max(0, math.floor((x1 - dx) * width)),
              max(0, math.floor((y1 - dy) * height)),
              min(width, math.ceil((x2 + dx) * width)),
              min(height, math.ceil((y2 + dy) * height))]
    if pixels[2] - pixels[0] < 4 or pixels[3] - pixels[1] < 4:
        return None
    if (pixels[2] - pixels[0]) * (pixels[3] - pixels[1]) > 0.65 * width * height:
        return None
    return pixels


def candidate_support(candidate, adjacent_frames):
    # Weak continuity bonus only. Sparse input frames do not establish identity.
    group = label_group(candidate["label"])
    return sum(any(label_group(other["label"]) == group and
                   box_iou(candidate["box"], other["box"]) >= 0.15
                   for other in frame["candidates"]) for frame in adjacent_frames)


def select_candidates(question, frame, adjacent_frames, groups, max_regions=2, min_score=0.35):
    if max_regions <= 0:
        return []
    q = question.lower()
    left = bool(re.search(r"\bleft\b", q))
    right = bool(re.search(r"\bright\b", q))
    ranked = []
    for candidate in frame["candidates"]:
        if label_group(candidate["label"]) not in groups or candidate["score"] < min_score:
            continue
        x1, y1, x2, y2 = candidate["box"]
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or area > 0.55:
            continue
        score = float(candidate["score"]) + 0.04 * candidate_support(candidate, adjacent_frames)
        center = (x1 + x2) / 2
        if left != right:
            score += 0.12 * ((1.0 - center) if left else center)
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]["candidate_id"]))
    chosen = []
    for _, candidate in ranked:
        # Suppress overlapping vehicle hypotheses even if car/truck labels differ.
        if any(label_group(candidate["label"]) == label_group(other["label"]) and
               box_iou(candidate["box"], other["box"]) > 0.65 for other in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= max_regions:
            break
    return chosen


def plan_views(question_id, segment_id, question, frame_paths, proposal,
               max_regions=2, min_score=0.35, padding=0.2, replay=False):
    if max_regions < 0 or max_regions > 2:
        raise ValueError("max_regions must be 0, 1, or 2")
    if proposal["protocol"] != PROPOSAL_PROTOCOL or proposal["segment_id"] != segment_id:
        raise ValueError("candidate segment/protocol mismatch")
    if [x["path"] for x in proposal["frames"]] != list(frame_paths):
        raise ValueError("candidate frames differ from original chronological frames")
    groups = question_groups(question)
    views = []
    for index, frame in enumerate(proposal["frames"]):
        views.append({"kind": "full", "frame_index": index})
        if replay or not groups or max_regions == 0:
            continue
        adjacent = [proposal["frames"][j] for j in (index - 1, index + 1)
                    if 0 <= j < len(frame_paths)]
        for candidate in select_candidates(question, frame, adjacent, groups, max_regions, min_score):
            pixels = expand_box(candidate["box"], frame["width"], frame["height"], padding)
            if pixels is None:
                continue
            key = stable_key(REGION_FEATURE_PROTOCOL, frame["image_sha256"], json.dumps(pixels))
            views.append({"kind": "region", "frame_index": index,
                          "box": normalized_box(pixels, frame["width"], frame["height"]),
                          "pixel_box": pixels, "image_sha256": frame["image_sha256"],
                          "region_key": key, "candidate_id": candidate["candidate_id"]})
    result = {"protocol": REGION_VIEW_PROTOCOL, "question_id": question_id,
              "segment_id": segment_id, "question_sha256": stable_key(question),
              "frame_paths": list(frame_paths), "frame_sha256": [x["image_sha256"] for x in proposal["frames"]],
              "focus_groups": groups, "full_replay": bool(replay), "views": views,
              "candidate_provenance": proposal["detector_identity"]}
    result["plan_sha256"] = stable_key(json.dumps(result, sort_keys=True))
    return result


def full_replay(question_id, fraction=0.2, seed=44):
    if not 0 <= fraction <= 1:
        raise ValueError("replay fraction must be in [0,1]")
    return int(stable_key("region_replay", seed, question_id)[:16], 16) / float(2 ** 64) < fraction


def validate_plan(plan, row):
    if (plan["protocol"] not in (REGION_VIEW_PROTOCOL, EVIDENCE_VIEW_PROTOCOL) or plan["question_id"] != row["question_id"] or
            plan["segment_id"] != row["segment_id"] or plan["question_sha256"] != stable_key(row["question"]) or
            plan["frame_paths"] != row["frames"]):
        raise ValueError("view plan does not belong to this question and original frames")
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != stable_key(json.dumps(content, sort_keys=True)):
        raise ValueError("view plan fingerprint mismatch")
    previous, full_frames = -1, []
    counts = {}
    for view in plan["views"]:
        frame = view["frame_index"]
        if not 0 <= frame < len(row["frames"]) or frame < previous:
            raise ValueError("view plan changes chronological order")
        if view["kind"] == "full":
            full_frames.append(frame)
        elif view["kind"] == "region":
            if not full_frames or full_frames[-1] != frame:
                raise ValueError("region must follow its original full frame")
            counts[frame] = counts.get(frame, 0) + 1
            if counts[frame] > 2:
                raise ValueError("more than two regions at one time")
        else:
            raise ValueError("unknown view kind")
        previous = frame
    if full_frames != list(range(len(row["frames"]))):
        raise ValueError("view plan must retain every original full frame once")
    if plan["protocol"] == EVIDENCE_VIEW_PROTOCOL:
        if len(row["frames"]) != 5 or sum(counts.values()) > 5 or len(plan["frame_sizes"]) != 5:
            raise ValueError("evidence search must retain five frames and append at most five crops")
        for view in plan["views"]:
            if view["kind"] != "region":
                continue
            frame = view["frame_index"]
            w, h = plan["frame_sizes"][frame]
            pixels = view["pixel_box"]
            if (len(pixels) != 4 or any(type(x) is not int for x in pixels) or
                    not 0 <= pixels[0] < pixels[2] <= w or not 0 <= pixels[1] < pixels[3] <= h or
                    view["image_sha256"] != plan["frame_sha256"][frame] or
                    view["box"] != normalized_box(pixels, w, h) or
                    view["region_key"] != stable_key(REGION_FEATURE_PROTOCOL, view["image_sha256"], json.dumps(pixels))):
                raise ValueError("evidence crop pixel/content provenance is invalid")


def view_prompt(question, plan, condition="roi", instruction=TASK_INSTRUCTION):
    if condition not in CONDITIONS:
        raise ValueError("unknown region condition")
    if condition == "full" or not any(v["kind"] == "region" for v in plan["views"]):
        return build_user_prompt(question, len(plan["frame_paths"]), instruction)
    slots = []
    for view in plan["views"]:
        index = view["frame_index"]
        if view["kind"] == "full":
            label = "T%d full" % index
        else:
            box = ",".join(str(round(x * 1000)) for x in view["box"])
            label = "T%d additional view; candidate box [%s]/1000" % (index, box)
        slots.append(label + ": <Img><ImageHere></Img>")
    # Identical prompt and coordinates for the ROI and duplicate conditions.
    # Additional images can be crops OR full frames; no false claim that a
    # duplicated full image is itself bounded by the candidate coordinates.
    return ("\n".join(slots) + "\n" + instruction +
            " Additional views show the same time, not extra time steps; they may be full frames or crops."
            " Boxes locate candidate regions in the full frame, with coordinates from 0 to 1000."
            " Candidates may be irrelevant: use the full scene to identify the question's target."
            "\nQuestion: " + question)


def compose_features(full_features, plan, condition, load_region):
    import torch
    if condition == "full":
        return full_features
    if condition not in ("duplicate", "roi"):
        raise ValueError("unknown region condition")
    views = []
    for view in plan["views"]:
        feature = (load_region(view) if condition == "roi" and view["kind"] == "region"
                   else full_features[view["frame_index"]])
        if feature.shape != full_features.shape[1:]:
            raise ValueError("region/full EVA feature shapes differ")
        views.append(feature)
    return torch.stack(views)


def add_region_arguments(parser):
    parser.add_argument("--region-views", default="", help="question-specific view JSONL")
    parser.add_argument("--region-cache-dir", default="", help="shared original-crop EVA cache")
    parser.add_argument("--region-condition", choices=CONDITIONS, default="roi")


def region_runtime_identity(args):
    if not getattr(args, "region_views", ""):
        return {}
    from .core import file_sha256, load_jsonl
    plans = load_jsonl(args.region_views)
    protocols = {p["protocol"] for p in plans}
    if len(protocols) != 1 or not protocols <= {REGION_VIEW_PROTOCOL, EVIDENCE_VIEW_PROTOCOL}:
        raise ValueError("view plans must use one recognized observation protocol")
    protocol = next(iter(protocols))
    result = {"region_view_protocol": protocol,
              "region_views_sha256": file_sha256(args.region_views),
              "region_condition": args.region_condition,
              "region_cache_dir": str(Path(args.region_cache_dir).resolve()) if args.region_cache_dir else None}
    if protocol == EVIDENCE_VIEW_PROTOCOL:
        policies = {p["candidate_provenance"]["policy_sha256"] for p in plans}
        if len(policies) != 1:
            raise ValueError("evidence plans mix external search policies")
        result["evidence_policy_sha256"] = next(iter(policies))
    manifest = Path(args.region_cache_dir) / "cache_manifest.json" if args.region_cache_dir else None
    if args.region_condition == "roi":
        if manifest is None or not manifest.is_file():
            raise ValueError("ROI condition requires a completed/initialized region EVA cache")
        result["region_encoder_identity"] = json.loads(manifest.read_text())["encoder_identity"]
    return result
