#!/usr/bin/env python3
"""Persistent SAM2.1 Tiny video-predictor worker for phone alignment.

The reconstruction server runs in the ``reconviagen`` environment, whereas
SAM2 and Hydra live in ``any6d_sam3d``.  Starting a new interpreter for every
mask request costs most of the latency, so this localhost-only worker owns one
video predictor for its whole process lifetime.  Requests contain only paths
to an immutable calibration clip plus sparse box/point prompts derived from
the current Mesh projection; the worker writes one observed mask per frame.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import cv2
import numpy as np


TRACKER_ROOT = Path(__file__).resolve().parents[1]
FORMAT = "manual_mesh_reconstruction.sam2_tiny_video_worker.v1"
MODEL_NAME = "sam2.1_hiera_tiny"
_MODEL_LOCK = threading.Lock()
_PREDICTOR = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from CoarseModel.connect.sam2_mask import _load_sam2_video_predictor

        _PREDICTOR = _load_sam2_video_predictor()
    return _PREDICTOR


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    from CoarseModel.connect.sam2_mask import _clean_mask as clean

    return clean(np.asarray(mask, dtype=bool), keep_largest=True)


def _validate_paths(values: Any, *, name: str, must_exist: bool) -> list[Path]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    paths = [Path(str(value)).expanduser().resolve() for value in values]
    if must_exist:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name} contains missing files: {missing[:4]}")
    return paths


def segment_video(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    images = _validate_paths(payload.get("image_paths"), name="image_paths", must_exist=True)
    outputs = _validate_paths(payload.get("mask_paths"), name="mask_paths", must_exist=False)
    if len(images) != len(outputs):
        raise ValueError("image_paths and mask_paths must have equal length")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("at least one sparse Mesh-derived prompt is required")

    parsed_prompts: list[dict[str, Any]] = []
    for raw in prompts:
        if not isinstance(raw, dict):
            raise ValueError("each prompt must be an object")
        frame_index = int(raw.get("frame_index", -1))
        if frame_index < 0 or frame_index >= len(images):
            raise ValueError(f"prompt frame index is out of range: {frame_index}")
        points = np.asarray(raw.get("points", []), dtype=np.float32)
        labels = np.asarray(raw.get("labels", []), dtype=np.int32)
        box_value = raw.get("box")
        box = None if box_value is None else np.asarray(box_value, dtype=np.float32)
        if points.size:
            points = points.reshape(-1, 2)
            labels = labels.reshape(-1)
            if len(points) != len(labels):
                raise ValueError("prompt point/label counts differ")
        else:
            points = None
            labels = None
        if box is not None:
            box = box.reshape(4)
        if points is None and box is None:
            raise ValueError("prompt must contain points or a box")
        parsed_prompts.append(
            {
                "frame_index": frame_index,
                "points": points,
                "labels": labels,
                "box": box,
            }
        )

    predictor = _predictor()
    with tempfile.TemporaryDirectory(prefix="tracker_sam2_tiny_refine_") as temporary:
        video_dir = Path(temporary) / "frames"
        video_dir.mkdir()
        image_shapes: list[tuple[int, int]] = []
        for index, image_path in enumerate(images):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV could not decode calibration RGB: {image_path}")
            image_shapes.append((int(image.shape[0]), int(image.shape[1])))
            if not cv2.imwrite(str(video_dir / f"{index:05d}.jpg"), image):
                raise RuntimeError(f"failed to materialize SAM2 frame: {image_path}")

        state = predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        prompt_outputs: dict[int, np.ndarray] = {}
        try:
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                for prompt in sorted(parsed_prompts, key=lambda row: row["frame_index"]):
                    _frame, object_ids, logits = predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=prompt["frame_index"],
                        obj_id=1,
                        points=prompt["points"],
                        labels=prompt["labels"],
                        box=prompt["box"],
                        clear_old_points=True,
                        normalize_coords=True,
                    )
                    ids = list(object_ids)
                    object_index = ids.index(1) if 1 in ids else 0
                    prompt_outputs[prompt["frame_index"]] = (
                        logits[object_index] > 0.0
                    ).detach().cpu().numpy().squeeze()

                start = min(row["frame_index"] for row in parsed_prompts)
                masks: dict[int, np.ndarray] = dict(prompt_outputs)
                for frame_index, object_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=False
                ):
                    ids = list(object_ids)
                    object_index = ids.index(1) if 1 in ids else 0
                    masks[int(frame_index)] = (
                        logits[object_index] > 0.0
                    ).detach().cpu().numpy().squeeze()
                if start > 0:
                    for frame_index, object_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=start, reverse=True
                    ):
                        ids = list(object_ids)
                        object_index = ids.index(1) if 1 in ids else 0
                        masks[int(frame_index)] = (
                            logits[object_index] > 0.0
                        ).detach().cpu().numpy().squeeze()
        finally:
            predictor.reset_state(state)
            # Keep model weights resident but release per-video tensors before
            # the reconstruction/pose-refinement process uses the same GPU.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows = []
        for index, output_path in enumerate(outputs):
            mask = masks.get(index)
            if mask is None:
                mask = np.zeros(image_shapes[index], dtype=bool)
            cleaned = _clean_mask(mask)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_name(f".{output_path.name}.tmp.png")
            if not cv2.imwrite(str(temporary_path), cleaned):
                raise RuntimeError(f"failed to write SAM2 mask: {output_path}")
            os.replace(temporary_path, output_path)
            rows.append(
                {
                    "frame_index": index,
                    "mask": str(output_path),
                    "foreground_pixels": int((cleaned > 0).sum()),
                    "image_pixels": int(cleaned.size),
                    "foreground_ratio": float((cleaned > 0).mean()),
                }
            )
    return {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "model": MODEL_NAME,
        "predictor_kind": "SAM2VideoPredictor",
        "predictor_process_persistent": True,
        "image_count": len(images),
        "prompt_frame_indices": sorted(
            {row["frame_index"] for row in parsed_prompts}
        ),
        "masks": rows,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "TrackerSAM2Tiny/1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self._send(404, {"status": "error", "message": "not found"})
            return
        self._send(
            200,
            {
                "status": "ok",
                "format": FORMAT,
                "model": MODEL_NAME,
                "model_loaded": _PREDICTOR is not None,
                "pid": os.getpid(),
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/segment":
            self._send(404, {"status": "error", "message": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8 * 1024 * 1024:
                raise ValueError("invalid JSON request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with _MODEL_LOCK:
                result = segment_video(payload)
            self._send(200, result)
        except Exception as error:
            import traceback

            traceback.print_exc()
            self._send(
                500,
                {
                    "status": "error",
                    "format": FORMAT,
                    "message": str(error),
                    "error_type": type(error).__name__,
                },
            )

    def log_message(self, pattern: str, *args: Any) -> None:
        print(
            f"[{utc_now()}] {self.client_address[0]} " + pattern % args,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5091)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if args.preload:
        _predictor()
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(
        json.dumps(
            {
                "format": FORMAT,
                "status": "ready",
                "host": args.host,
                "port": int(args.port),
                "model": MODEL_NAME,
                "model_loaded": _PREDICTOR is not None,
                "pid": os.getpid(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
