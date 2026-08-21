#!/usr/bin/env python3
"""Compatibility entry point for the AR capture and reconstruction server."""

from pose_point_depth_mv.server import *  # noqa: F401,F403
from pose_point_depth_mv.server import main


if __name__ == "__main__":
    main()
