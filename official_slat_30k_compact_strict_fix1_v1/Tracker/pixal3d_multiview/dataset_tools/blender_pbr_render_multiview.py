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

runpy.run_path(str(TARGET), run_name="__main__")
