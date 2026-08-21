#!/usr/bin/env python3
"""Original ReconViaGen entrypoint for the frozen Aug-11 runtime v2 input."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from pose_point_depth_mv import infer_omni_real_reconviagen as _base


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
    ):
        raise RuntimeError(f"frozen Aug-11 Omni v2 identity differs: {path}")
    _base.IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS.add(FROZEN_MANIFEST_FORMAT)
    _base.main()


if __name__ == "__main__":
    main()
