#!/usr/bin/env python3
"""Compatibility entrypoint; implementation moved to pose_point_depth_mv."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parents[2]
    / "pose_point_depth_mv"
    / "dataset_tools"
    / Path(__file__).name
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
else:
    globals().update(
        {
            key: value
            for key, value in runpy.run_path(str(TARGET)).items()
            if not key.startswith("__")
        }
    )
