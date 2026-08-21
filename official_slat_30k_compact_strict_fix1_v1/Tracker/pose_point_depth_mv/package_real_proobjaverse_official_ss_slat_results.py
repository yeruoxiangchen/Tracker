#!/usr/bin/env python3
"""Package official-SS/trained-SLat real inference in runtime-O/world frames."""

from __future__ import annotations

from pose_point_depth_mv import package_coarsemodel_real_no_vggt_results as _base
from pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat import (
    MANIFEST_FORMAT,
)


def main() -> None:
    # The base packager owns the audited T_O2W mesh conversion and input sheet.
    # Extend only its explicit manifest-format allowlist for this new inference
    # producer; no coordinate conversion is reimplemented here.
    _base.INFERENCE_MANIFEST_FORMATS.add(MANIFEST_FORMAT)
    _base.main()


if __name__ == "__main__":
    main()
