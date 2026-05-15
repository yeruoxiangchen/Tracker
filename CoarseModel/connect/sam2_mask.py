import os
import json
import subprocess
import sys
import tempfile
from functools import lru_cache

import cv2
import numpy as np


TRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM2_ROOT = os.path.join(TRACKER_ROOT, "sam2")
SAM2_FALLBACK_PYTHON = "/home/zjr/anaconda3/envs/any6d_sam3d/bin/python"
SAM2_SUBPROCESS_ENV = "TRACKER_SAM2_MASK_SUBPROCESS"


def _largest_component(mask):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask_u8.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def _fill_holes(mask_u8):
    binary = (mask_u8 > 0).astype(np.uint8)
    if binary.size == 0:
        return mask_u8

    flood = binary.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for x in range(w):
        if flood[0, x] == 0:
            cv2.floodFill(flood, flood_mask, (x, 0), 2)
        if flood[h - 1, x] == 0:
            cv2.floodFill(flood, flood_mask, (x, h - 1), 2)
    for y in range(h):
        if flood[y, 0] == 0:
            cv2.floodFill(flood, flood_mask, (0, y), 2)
        if flood[y, w - 1] == 0:
            cv2.floodFill(flood, flood_mask, (w - 1, y), 2)

    holes = flood == 0
    filled = np.logical_or(binary > 0, holes)
    return filled.astype(np.uint8) * 255


def _filter_components(mask_u8, min_area=32, keep_largest=False):
    binary = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return mask_u8
    if keep_largest:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8) * 255

    kept = np.zeros(labels.shape, dtype=np.uint8)
    for component_label in range(1, num_labels):
        if stats[component_label, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == component_label] = 255
    return kept


def _clean_mask(mask, keep_largest=False):
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    min_area = max(32, int(mask_u8.shape[0] * mask_u8.shape[1] * 0.00012))
    mask_u8 = _filter_components(mask_u8, min_area=min_area, keep_largest=keep_largest)
    return _fill_holes(mask_u8)


def _point_mask_label(labels, x, y, radius=8):
    h, w = labels.shape[:2]
    x0 = max(0, int(round(x)) - radius)
    x1 = min(w, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(h, int(round(y)) + radius + 1)
    window = labels[y0:y1, x0:x1]
    values = window[window > 0]
    if values.size == 0:
        return 0
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def _clean_prompt_mask(mask, image_shape, points):
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return mask_u8

    labels_with_positive = set()
    labels_with_negative = set()
    for point in points:
        pxy, label = _pixel_prompt(point, image_shape)
        component_label = _point_mask_label(labels, pxy[0], pxy[1])
        if component_label <= 0:
            continue
        if label == 1:
            labels_with_positive.add(component_label)
        else:
            labels_with_negative.add(component_label)

    image_area = float(max(1, image_shape[0] * image_shape[1]))
    min_component_area = max(32, int(image_area * 0.00012))
    keep_labels = set()

    for component_label in range(1, num_labels):
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        has_positive = component_label in labels_with_positive
        has_negative = component_label in labels_with_negative
        if has_negative and not has_positive:
            continue
        if has_positive or area >= min_component_area:
            keep_labels.add(component_label)

    if not keep_labels:
        valid_labels = [
            i for i in range(1, num_labels)
            if i not in labels_with_negative and stats[i, cv2.CC_STAT_AREA] >= min_component_area
        ]
        if not valid_labels:
            return np.zeros(labels.shape, dtype=np.uint8)
        largest = max(valid_labels, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        keep_labels = {largest}

    cleaned = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255
    return _fill_holes(cleaned)


def _select_prompt_mask(masks, scores, image_shape, points, score_thresh=0.0):
    best_idx = 0
    best_score = -1e9
    image_area = float(max(1, image_shape[0] * image_shape[1]))

    for idx, candidate in enumerate(masks):
        cleaned = _clean_prompt_mask(candidate > score_thresh, image_shape, points)
        binary = cleaned > 0
        area_ratio = float(binary.sum()) / image_area

        prompt_score = float(scores[idx]) if idx < len(scores) else 0.0
        for point in points:
            pxy, label = _pixel_prompt(point, image_shape)
            x = int(round(pxy[0]))
            y = int(round(pxy[1]))
            x0 = max(0, x - 5)
            x1 = min(binary.shape[1], x + 6)
            y0 = max(0, y - 5)
            y1 = min(binary.shape[0], y + 6)
            hit = bool(binary[y0:y1, x0:x1].any())
            if label == 1:
                prompt_score += 3.0 if hit else -8.0
            else:
                prompt_score += -6.0 if hit else 2.0

        if area_ratio < 0.001:
            prompt_score -= 4.0
        if area_ratio > 0.65:
            prompt_score -= 4.0

        if prompt_score > best_score:
            best_score = prompt_score
            best_idx = idx

    return _clean_prompt_mask(masks[best_idx] > score_thresh, image_shape, points)


def _negative_points(points):
    return [point for point in points if int(point.get("label", 1)) == 0]


def _positive_points(points):
    return [point for point in points if int(point.get("label", 1)) == 1]


def _foreground_from_preview(preview_bgra):
    if preview_bgra is None:
        return None, None

    if preview_bgra.ndim == 3 and preview_bgra.shape[2] == 4:
        bgr = preview_bgra[:, :, :3]
        fg = preview_bgra[:, :, 3] > 10
    elif preview_bgra.ndim == 3:
        bgr = preview_bgra
        gray = cv2.cvtColor(preview_bgra, cv2.COLOR_BGR2GRAY)
        fg = gray > 10
    else:
        return None, preview_bgra > 10

    if int(fg.sum()) < 32:
        return None, fg
    return bgr[fg], fg


def _hsv_hist_from_pixels(bgr_pixels):
    if bgr_pixels is None or len(bgr_pixels) < 32:
        return None
    pixels = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _mask_hist_similarity(image_bgr, mask, ref_hist):
    if ref_hist is None:
        return 0.0
    pixels = image_bgr[mask.astype(bool)]
    if len(pixels) < 32:
        return 0.0
    hist = _hsv_hist_from_pixels(pixels)
    if hist is None:
        return 0.0
    distance = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - distance))


def _bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _expanded_bbox_contains(box, other, image_shape, rel_pad=0.18):
    h, w = image_shape[:2]
    pad_x = rel_pad * w
    pad_y = rel_pad * h
    expanded = np.array(
        [
            max(0.0, box[0] - pad_x),
            max(0.0, box[1] - pad_y),
            min(float(w - 1), box[2] + pad_x),
            min(float(h - 1), box[3] + pad_y),
        ],
        dtype=np.float32,
    )
    cx = 0.5 * (other[0] + other[2])
    cy = 0.5 * (other[1] + other[3])
    return expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]


def _bbox_gap(box_a, box_b):
    dx = max(box_a[0] - box_b[2], box_b[0] - box_a[2], 0.0)
    dy = max(box_a[1] - box_b[3], box_b[1] - box_a[3], 0.0)
    return float(dx), float(dy)


def _bbox_overlap_ratio_1d(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1.0, min(a1 - a0, b1 - b0))
    return float(inter / denom)


def _is_spatially_related(anchor_box, box, image_shape):
    h, w = image_shape[:2]
    dx, dy = _bbox_gap(anchor_box, box)
    x_overlap = _bbox_overlap_ratio_1d(anchor_box[0], anchor_box[2], box[0], box[2])
    y_overlap = _bbox_overlap_ratio_1d(anchor_box[1], anchor_box[3], box[1], box[3])

    if dx <= 0.04 * w and dy <= 0.12 * h:
        return True
    if x_overlap > 0.25 and dy <= 0.22 * h:
        return True
    if y_overlap > 0.25 and dx <= 0.16 * w:
        return True
    return False


def _is_background_like(image_bgr, seg, area_ratio):
    pixels = image_bgr[seg.astype(bool)]
    if len(pixels) < 32:
        return True
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat_mean = float(hsv[:, 1].mean())
    val_mean = float(hsv[:, 2].mean())

    # Large low-saturation bright masks are usually table/wall/background.
    if area_ratio > 0.18 and sat_mean < 35.0 and val_mean > 150.0:
        return True
    # Large dark masks are usually keyboard/chair.
    if area_ratio > 0.12 and val_mean < 90.0:
        return True
    return False


def _touches_image_border(box, image_shape, margin=3):
    h, w = image_shape[:2]
    return (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= w - 1 - margin
        or box[3] >= h - 1 - margin
    )


@lru_cache(maxsize=1)
def _load_sam2_generator():
    if SAM2_ROOT not in sys.path:
        sys.path.insert(0, SAM2_ROOT)

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint,
        device=device,
        apply_postprocessing=True,
    )
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.85,
        min_mask_region_area=128,
        output_mode="binary_mask",
    )


@lru_cache(maxsize=1)
def _load_sam2_video_predictor():
    if SAM2_ROOT not in sys.path:
        sys.path.insert(0, SAM2_ROOT)

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    return build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint,
        device=device,
        apply_postprocessing=True,
        hydra_overrides_extra=[
            "++model.non_overlap_masks=true",
            "++model.fill_hole_area=128",
        ],
    )


@lru_cache(maxsize=1)
def _load_sam2_image_predictor():
    if SAM2_ROOT not in sys.path:
        sys.path.insert(0, SAM2_ROOT)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint,
        device=device,
        apply_postprocessing=True,
    )
    return SAM2ImagePredictor(model)


def generate_sam2_reference_mask(image_bgr, preview_bgra):
    ref_pixels, _ = _foreground_from_preview(preview_bgra)
    ref_hist = _hsv_hist_from_pixels(ref_pixels)
    if ref_hist is None:
        raise ValueError("Preview foreground is empty; cannot select a SAM2 mask")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = _load_sam2_generator().generate(image_rgb)
    if not masks:
        raise RuntimeError("SAM2 returned no masks")

    h, w = image_bgr.shape[:2]
    image_area = float(h * w)
    candidates = []

    for item in masks:
        seg = np.asarray(item["segmentation"]).astype(bool)
        area = float(seg.sum())
        area_ratio = area / image_area
        if area_ratio < 0.004 or area_ratio > 0.75:
            continue

        hist_sim = _mask_hist_similarity(image_bgr, seg, ref_hist)
        if hist_sim <= 0.0:
            continue

        bbox = _bbox_from_mask(seg)
        if bbox is None:
            continue
        if area_ratio > 0.015 and _touches_image_border(bbox, image_bgr.shape):
            continue

        if _is_background_like(image_bgr, seg, area_ratio):
            continue

        quality = float(item.get("predicted_iou", 0.0)) + float(item.get("stability_score", 0.0))
        score = 3.0 * hist_sim + 0.25 * quality
        candidates.append((score, hist_sim, area_ratio, seg, bbox, quality))

    if not candidates:
        raise RuntimeError("No usable SAM2 mask candidates")

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_hist, _, best_seg, best_box, _ = candidates[0]

    union = best_seg.copy()
    union_box = best_box.copy()

    for _ in range(2):
        changed = False
        for score, hist_sim, area_ratio, seg, box, quality in candidates[1:]:
            if np.any(union & seg):
                related = True
            else:
                related = _expanded_bbox_contains(union_box, box, image_bgr.shape) or _is_spatially_related(
                    union_box, box, image_bgr.shape
                )

            if not related:
                continue
            if area_ratio > 0.35:
                continue

            color_match = hist_sim >= max(0.16, best_hist * 0.38)
            strong_sam_piece = quality >= 1.55 and area_ratio <= 0.16
            if not color_match and not strong_sam_piece:
                continue

            before = int(union.sum())
            union |= seg
            if int(union.sum()) != before:
                updated_box = _bbox_from_mask(union)
                if updated_box is not None:
                    union_box = updated_box
                changed = True
        if not changed:
            break

    for score, hist_sim, area_ratio, seg, box, quality in candidates[1:]:
        if hist_sim < max(0.24, best_hist * 0.55):
            continue
        if area_ratio > 0.22:
            continue
        if _expanded_bbox_contains(union_box, box, image_bgr.shape) or _is_spatially_related(
            union_box, box, image_bgr.shape
        ):
            union |= seg

    return _clean_mask(union)


def write_sam2_reference_mask(image_path, preview_path, mask_path):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    preview_bgra = cv2.imread(preview_path, cv2.IMREAD_UNCHANGED)
    if preview_bgra is None:
        raise FileNotFoundError(f"Preview image not found: {preview_path}")

    try:
        mask = generate_sam2_reference_mask(image_bgr, preview_bgra)
    except Exception as exc:
        if (
            os.environ.get(SAM2_SUBPROCESS_ENV) != "1"
            and os.path.exists(SAM2_FALLBACK_PYTHON)
            and os.path.abspath(sys.executable) != os.path.abspath(SAM2_FALLBACK_PYTHON)
        ):
            env = os.environ.copy()
            env[SAM2_SUBPROCESS_ENV] = "1"
            try:
                subprocess.run(
                    [SAM2_FALLBACK_PYTHON, __file__, image_path, preview_path, mask_path],
                    check=True,
                    env=env,
                )
                return mask_path
            except Exception as sub_exc:
                raise RuntimeError(
                    f"SAM2 failed in current env ({exc}) and fallback env ({sub_exc})"
                ) from sub_exc
        raise

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask)
    return mask_path


def _conditioning_indices(num_frames):
    if num_frames <= 0:
        return []
    indices = {0, num_frames // 2, num_frames - 1}
    if num_frames >= 8:
        indices.add(num_frames // 4)
        indices.add((3 * num_frames) // 4)
    return sorted(indices)


def _run_video_consistent_masks(image_paths, preview_paths, mask_paths, score_thresh=0.0):
    import torch

    if not (len(image_paths) == len(preview_paths) == len(mask_paths)):
        raise ValueError("image_paths, preview_paths and mask_paths must have the same length")
    if len(image_paths) == 0:
        return []

    predictor = _load_sam2_video_predictor()

    with tempfile.TemporaryDirectory(prefix="tracker_sam2_video_") as tmp_dir:
        video_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(video_dir, exist_ok=True)

        seed_masks = {}
        for i, image_path in enumerate(image_paths):
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            cv2.imwrite(os.path.join(video_dir, f"{i:05d}.jpg"), image_bgr)

        for i in _conditioning_indices(len(image_paths)):
            image_bgr = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
            preview_bgra = cv2.imread(preview_paths[i], cv2.IMREAD_UNCHANGED)
            if preview_bgra is None:
                raise FileNotFoundError(f"Preview image not found: {preview_paths[i]}")
            seed_masks[i] = generate_sam2_reference_mask(image_bgr, preview_bgra) > 0

        inference_state = predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )

        for frame_idx, seed_mask in seed_masks.items():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                mask=seed_mask,
            )

        outputs = {}
        autocast_enabled = torch.cuda.is_available()
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast_enabled else torch.no_grad()
        with torch.inference_mode(), autocast_ctx:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                reverse=False,
            ):
                if 1 in out_obj_ids:
                    obj_idx = list(out_obj_ids).index(1)
                else:
                    obj_idx = 0
                mask = (out_mask_logits[obj_idx] > score_thresh).cpu().numpy()
                outputs[out_frame_idx] = _clean_mask(mask.squeeze())

        for i, mask_path in enumerate(mask_paths):
            mask = outputs.get(i)
            if mask is None:
                mask = (seed_masks.get(i, np.zeros(cv2.imread(image_paths[i]).shape[:2], dtype=bool)).astype(np.uint8) * 255)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, mask)

    return mask_paths


def _pixel_prompt(point, image_shape):
    h, w = image_shape[:2]
    x = float(point["x"])
    y = float(point["y"])
    if point.get("normalized", True):
        x *= w
        y *= h
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))
    return [x, y], int(point.get("label", 1))


def _predict_prompted_image_mask_array(image_bgr, frame_points, score_thresh=0.0):
    import torch

    if not frame_points:
        raise ValueError("At least one prompt point is required for this frame")

    predictor = _load_sam2_image_predictor()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    autocast_enabled = torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast_enabled else torch.no_grad()
    candidate_masks = []

    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image_rgb)

        prompt_points = []
        prompt_labels = []
        for point in frame_points:
            pxy, label = _pixel_prompt(point, image_bgr.shape)
            prompt_points.append(pxy)
            prompt_labels.append(label)
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(prompt_points, dtype=np.float32),
            point_labels=np.asarray(prompt_labels, dtype=np.int32),
            multimask_output=True,
            normalize_coords=True,
        )
        candidate_masks.append(_select_prompt_mask(masks, scores, image_bgr.shape, frame_points, score_thresh=score_thresh))

        background_points = _negative_points(frame_points)
        for positive_point in _positive_points(frame_points):
            local_points = [positive_point] + background_points
            prompt_points = []
            prompt_labels = []
            for point in local_points:
                pxy, label = _pixel_prompt(point, image_bgr.shape)
                prompt_points.append(pxy)
                prompt_labels.append(label)
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(prompt_points, dtype=np.float32),
                point_labels=np.asarray(prompt_labels, dtype=np.int32),
                multimask_output=True,
                normalize_coords=True,
            )
            candidate_masks.append(
                _select_prompt_mask(masks, scores, image_bgr.shape, local_points, score_thresh=score_thresh)
            )

    union = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for mask in candidate_masks:
        union = np.maximum(union, (mask > 0).astype(np.uint8) * 255)

    return _clean_prompt_mask(union > 0, image_bgr.shape, frame_points)


def _run_prompted_image_mask(image_path, mask_path, points, frame_index=None, score_thresh=0.0):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame_points = []
    for point in points:
        if frame_index is None or int(point.get("frame_index", -1)) == int(frame_index):
            frame_points.append(point)

    mask = _predict_prompted_image_mask_array(image_bgr, frame_points, score_thresh=score_thresh)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask)
    return mask_path


def write_prompted_image_mask(image_path, mask_path, points, frame_index=None):
    try:
        return _run_prompted_image_mask(image_path, mask_path, points, frame_index=frame_index)
    except Exception as exc:
        if (
            os.environ.get(SAM2_SUBPROCESS_ENV) != "1"
            and os.path.exists(SAM2_FALLBACK_PYTHON)
            and os.path.abspath(sys.executable) != os.path.abspath(SAM2_FALLBACK_PYTHON)
        ):
            env = os.environ.copy()
            env[SAM2_SUBPROCESS_ENV] = "1"
            payload_path = None
            try:
                fd, payload_path = tempfile.mkstemp(prefix="tracker_sam2_image_", suffix=".json")
                with os.fdopen(fd, "w") as f:
                    json.dump(
                        {
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "points": points,
                            "frame_index": frame_index,
                        },
                        f,
                    )
                subprocess.run(
                    [SAM2_FALLBACK_PYTHON, __file__, "--prompted-image", payload_path],
                    check=True,
                    env=env,
                )
                return mask_path
            except Exception as sub_exc:
                raise RuntimeError(
                    f"SAM2 prompted image failed in current env ({exc}) and fallback env ({sub_exc})"
                ) from sub_exc
            finally:
                if payload_path and os.path.exists(payload_path):
                    os.remove(payload_path)
        raise


def _run_prompted_video_masks(image_paths, mask_paths, points, seed_frame_indices=None, score_thresh=0.0):
    import torch

    if len(image_paths) != len(mask_paths):
        raise ValueError("image_paths and mask_paths must have the same length")
    if len(image_paths) == 0:
        return []
    if not points:
        raise ValueError("At least one foreground/background point is required")

    predictor = _load_sam2_video_predictor()

    all_points_by_frame = {}
    for point in points:
        frame_idx = int(point["frame_index"])
        if 0 <= frame_idx < len(image_paths):
            all_points_by_frame.setdefault(frame_idx, []).append(point)
    seed_frame_set = set(int(i) for i in seed_frame_indices) if seed_frame_indices is not None else set(all_points_by_frame)
    points_by_frame = {
        frame_idx: frame_points
        for frame_idx, frame_points in all_points_by_frame.items()
        if frame_idx in seed_frame_set
    }
    if not points_by_frame:
        raise ValueError("No approved seed frames with prompt points for video propagation")

    with tempfile.TemporaryDirectory(prefix="tracker_sam2_prompted_") as tmp_dir:
        video_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(video_dir, exist_ok=True)

        image_shapes = []
        for i, image_path in enumerate(image_paths):
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            image_shapes.append(image_bgr.shape)
            cv2.imwrite(os.path.join(video_dir, f"{i:05d}.jpg"), image_bgr)

        seed_masks = {}
        for frame_idx, frame_points in points_by_frame.items():
            image_bgr = cv2.imread(image_paths[frame_idx], cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Image not found: {image_paths[frame_idx]}")
            seed_masks[frame_idx] = _predict_prompted_image_mask_array(
                image_bgr,
                frame_points,
                score_thresh=score_thresh,
            )

        inference_state = predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )

        for frame_idx in sorted(seed_masks):
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                mask=seed_masks[frame_idx] > 0,
            )

        outputs = {}
        autocast_enabled = torch.cuda.is_available()
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast_enabled else torch.no_grad()
        start_idx = min(points_by_frame)
        with torch.inference_mode(), autocast_ctx:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start_idx,
                reverse=False,
            ):
                obj_idx = list(out_obj_ids).index(1) if 1 in out_obj_ids else 0
                mask = (out_mask_logits[obj_idx] > score_thresh).cpu().numpy().squeeze()
                if out_frame_idx in points_by_frame:
                    outputs[out_frame_idx] = _clean_prompt_mask(
                        mask,
                        image_shapes[out_frame_idx],
                        points_by_frame[out_frame_idx],
                    )
                else:
                    outputs[out_frame_idx] = _clean_mask(mask)

            if start_idx > 0:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_idx,
                    reverse=True,
                ):
                    obj_idx = list(out_obj_ids).index(1) if 1 in out_obj_ids else 0
                    mask = (out_mask_logits[obj_idx] > score_thresh).cpu().numpy().squeeze()
                    if out_frame_idx in points_by_frame:
                        outputs[out_frame_idx] = _clean_prompt_mask(
                            mask,
                            image_shapes[out_frame_idx],
                            points_by_frame[out_frame_idx],
                        )
                    else:
                        outputs[out_frame_idx] = _clean_mask(mask)

        for frame_idx, seed_mask in seed_masks.items():
            outputs[frame_idx] = seed_mask

        for i, mask_path in enumerate(mask_paths):
            mask = outputs.get(i)
            if mask is None:
                mask = np.zeros(image_shapes[i][:2], dtype=np.uint8)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, mask)

    return mask_paths


def write_prompted_video_masks(image_paths, mask_paths, points, seed_frame_indices=None):
    try:
        return _run_prompted_video_masks(image_paths, mask_paths, points, seed_frame_indices=seed_frame_indices)
    except Exception as exc:
        if (
            os.environ.get(SAM2_SUBPROCESS_ENV) != "1"
            and os.path.exists(SAM2_FALLBACK_PYTHON)
            and os.path.abspath(sys.executable) != os.path.abspath(SAM2_FALLBACK_PYTHON)
        ):
            env = os.environ.copy()
            env[SAM2_SUBPROCESS_ENV] = "1"
            payload_path = None
            try:
                fd, payload_path = tempfile.mkstemp(prefix="tracker_sam2_prompted_", suffix=".json")
                with os.fdopen(fd, "w") as f:
                    json.dump(
                        {
                            "image_paths": image_paths,
                            "mask_paths": mask_paths,
                            "points": points,
                            "seed_frame_indices": seed_frame_indices,
                        },
                        f,
                    )
                subprocess.run(
                    [SAM2_FALLBACK_PYTHON, __file__, "--prompted-video", payload_path],
                    check=True,
                    env=env,
                )
                return mask_paths
            except Exception as sub_exc:
                raise RuntimeError(
                    f"SAM2 prompted video failed in current env ({exc}) and fallback env ({sub_exc})"
                ) from sub_exc
            finally:
                if payload_path and os.path.exists(payload_path):
                    os.remove(payload_path)
        raise


def write_video_consistent_masks(image_paths, preview_paths, mask_paths):
    try:
        return _run_video_consistent_masks(image_paths, preview_paths, mask_paths)
    except Exception as exc:
        if (
            os.environ.get(SAM2_SUBPROCESS_ENV) != "1"
            and os.path.exists(SAM2_FALLBACK_PYTHON)
            and os.path.abspath(sys.executable) != os.path.abspath(SAM2_FALLBACK_PYTHON)
        ):
            env = os.environ.copy()
            env[SAM2_SUBPROCESS_ENV] = "1"
            list_path = None
            try:
                fd, list_path = tempfile.mkstemp(prefix="tracker_sam2_batch_", suffix=".txt")
                with os.fdopen(fd, "w") as f:
                    for image_path, preview_path, mask_path in zip(image_paths, preview_paths, mask_paths):
                        f.write(f"{image_path}\t{preview_path}\t{mask_path}\n")
                subprocess.run(
                    [SAM2_FALLBACK_PYTHON, __file__, "--video-list", list_path],
                    check=True,
                    env=env,
                )
                return mask_paths
            except Exception as sub_exc:
                raise RuntimeError(
                    f"SAM2 video failed in current env ({exc}) and fallback env ({sub_exc})"
                ) from sub_exc
            finally:
                if list_path and os.path.exists(list_path):
                    os.remove(list_path)
        raise


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--video-list":
        triples = []
        with open(sys.argv[2], "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                triples.append(line.split("\t"))
        image_paths = [t[0] for t in triples]
        preview_paths = [t[1] for t in triples]
        mask_paths = [t[2] for t in triples]
        write_video_consistent_masks(image_paths, preview_paths, mask_paths)
        return

    if len(sys.argv) == 3 and sys.argv[1] == "--prompted-video":
        with open(sys.argv[2], "r") as f:
            payload = json.load(f)
        write_prompted_video_masks(
            payload["image_paths"],
            payload["mask_paths"],
            payload["points"],
            seed_frame_indices=payload.get("seed_frame_indices"),
        )
        return

    if len(sys.argv) == 3 and sys.argv[1] == "--prompted-image":
        with open(sys.argv[2], "r") as f:
            payload = json.load(f)
        write_prompted_image_mask(
            payload["image_path"],
            payload["mask_path"],
            payload["points"],
            frame_index=payload.get("frame_index"),
        )
        return

    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: sam2_mask.py IMAGE_PATH PREVIEW_PATH MASK_PATH\n"
            "   or: sam2_mask.py --video-list TSV_PATH\n"
            "   or: sam2_mask.py --prompted-video PAYLOAD_JSON\n"
            "   or: sam2_mask.py --prompted-image PAYLOAD_JSON"
        )
    write_sam2_reference_mask(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
