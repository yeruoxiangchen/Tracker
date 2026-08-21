#!/usr/bin/env python3
"""Train the exact-v2 condition+small-LoRA official-SLat diagnostic arm."""

from __future__ import annotations

from pose_point_depth_mv import train_native_slat_genrecon as _base
from pose_point_depth_mv import train_native_slat_genrecon_no_vggt as _arm
from pose_point_depth_mv.proobjaverse_official_slat_training import (
    validate_official_decoder_audit,
)


def main() -> None:
    _base.validate_decoder_audit = validate_official_decoder_audit
    _arm.main()


if __name__ == "__main__":
    main()
