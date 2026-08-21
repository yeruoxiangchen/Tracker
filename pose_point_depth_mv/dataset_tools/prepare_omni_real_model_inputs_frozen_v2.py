#!/usr/bin/env python3
"""Compatibility entrypoint for the frozen Aug-11 runtime-input v2 artifact.

Only the manifest-version gate differs.  Encoding, native VGGT/DINO model
loading, tensor construction and output schema remain owned by the current
``prepare_omni_real_model_inputs`` implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from pose_point_depth_mv.dataset_tools import prepare_omni_real_model_inputs as _base


FROZEN_MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_manifest.v2"
FROZEN_OBJECT_FORMAT = "pose_point_depth_mv.omni_real_runtime_input_object.v2"


def _manifest_argument(argv: list[str]) -> Path:
    try:
        position = argv.index("--runtime_input_manifest")
        value = argv[position + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError("--runtime_input_manifest is required") from error
    return Path(value).expanduser().resolve(strict=True)


def main() -> None:
    path = _manifest_argument(sys.argv[1:])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects", [])
    if (
        payload.get("format") != FROZEN_MANIFEST_FORMAT
        or payload.get("passed") is not True
        or len(rows) != 1
        or rows[0].get("format") != FROZEN_OBJECT_FORMAT
        or rows[0].get("object_key") != "plant:plant_012"
        or rows[0].get("selected_source_view_indices")
        != [0, 9, 18, 27, 36, 45, 54, 63]
        or rows[0].get("selected_frame_names")
        != [
            "00000.jpg",
            "00084.jpg",
            "00171.jpg",
            "00255.jpg",
            "00342.jpg",
            "00426.jpg",
            "00515.jpg",
            "00599.jpg",
        ]
        or rows[0].get("view_selection", {}).get("fallback_used") is not False
    ):
        raise RuntimeError(f"frozen Aug-11 Omni v2 identity differs: {path}")
    _base.RUNTIME_MANIFEST_FORMAT = FROZEN_MANIFEST_FORMAT
    _base.main()


if __name__ == "__main__":
    main()
