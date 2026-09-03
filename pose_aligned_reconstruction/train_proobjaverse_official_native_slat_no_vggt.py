#!/usr/bin/env python3
"""Train the paper's no-VGGT Native-SLat v2 on official SLat targets."""

from __future__ import annotations

from pose_aligned_reconstruction import train_native_slat_genrecon as _base
from pose_aligned_reconstruction import train_native_slat_genrecon_no_vggt as _arm
from pose_aligned_reconstruction.proobjaverse_official_slat_training import (
    validate_official_decoder_audit,
)


def main() -> None:
    # The shared trainer also supports older direct-SLat experiments. Replace
    # only its target-audit validator for this explicit official-target entry.
    _base.validate_decoder_audit = validate_official_decoder_audit
    _arm.main()


if __name__ == "__main__":
    main()
