from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


def _json_loads_maybe(text: str | None) -> Any:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    return json.loads(text)


def _iter_point_rows(payload: Any):
    if payload is None:
        return
    if isinstance(payload, dict):
        points = payload.get("points", [])
    else:
        points = payload
    if not isinstance(points, list):
        return
    for item in points:
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
            z = item.get("z")
            confidence = item.get("confidence", item.get("conf", 1.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            x, y, z = item[:3]
            confidence = item[3] if len(item) > 3 else 1.0
        else:
            continue
        try:
            point = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "confidence": float(confidence),
            }
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(float(point[k])) for k in ("x", "y", "z", "confidence")):
            yield point


def parse_slam_points_payload(payload_text: str | None) -> dict[str, Any]:
    payload = _json_loads_maybe(payload_text)
    points = list(_iter_point_rows(payload))
    source_count = 0
    if isinstance(payload, dict):
        source_count = int(payload.get("point_count", len(points)) or len(points))
    elif isinstance(payload, list):
        source_count = len(payload)
    return {
        "schema": "arpose_tracker_frame_points_v1",
        "source_point_count": int(source_count),
        "point_count": int(len(points)),
        "points": points,
    }


def append_slam_points_from_upload(
    form: Any,
    data_dir: str | Path,
    *,
    frame_name: str,
    frame_index: int | None = None,
    output_name: str = "slam_points.jsonl",
) -> dict[str, Any]:
    """Persist ARFoundation point-cloud points from a Flask request form.

    This helper is intentionally standalone so `CoarseModel/connect/server.py` can
    call it with a small optional hook while the actual smoke code lives under
    `trellis_point_prior_mv`.
    """

    payload_text = None
    if hasattr(form, "get"):
        payload_text = form.get("slam_points_json") or form.get("ar_slam_points_json")
    if payload_text is None or not str(payload_text).strip():
        return {
            "schema": "arpose_tracker_frame_points_v1",
            "frame_name": frame_name,
            "frame_index": int(frame_index) if frame_index is not None else None,
            "skipped": True,
            "reason": "no_slam_points_json",
            "point_count": 0,
        }
    parsed = parse_slam_points_payload(payload_text)
    row = {
        "schema": parsed["schema"],
        "frame_name": frame_name,
        "frame_index": int(frame_index) if frame_index is not None else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_point_count": int(parsed["source_point_count"]),
        "point_count": int(parsed["point_count"]),
        "coordinate_frame": "unity_world",
        "points": parsed["points"],
    }
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    with (data_path / output_name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return row
