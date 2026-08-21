#!/usr/bin/env python3
"""Official Train2000 with-VGGT Native-SLat condition+LoRA entrypoint."""

from __future__ import annotations

from pose_point_depth_mv import train_native_slat_genrecon as _base
from pose_point_depth_mv import train_native_slat_genrecon_with_vggt_official as _arm
from pose_point_depth_mv.proobjaverse_official_slat_training import (
    validate_official_decoder_audit,
)


def main() -> None:
    _base.validate_decoder_audit = validate_official_decoder_audit
    _arm.main()


if __name__ == "__main__":
    main()
