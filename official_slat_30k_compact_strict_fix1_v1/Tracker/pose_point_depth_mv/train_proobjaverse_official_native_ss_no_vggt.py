#!/usr/bin/env python3
"""Train no-VGGT Native SS on audited official ProObjaverse SS targets."""

from __future__ import annotations

from pose_point_depth_mv import train_native_ss_genrecon as _trainer
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.proobjaverse_official_ss import (
    validate_official_ss_cache_contract,
)


def main() -> None:
    # Reuse the proven DDP/EMA/optimizer loop, but fail closed unless every row
    # is rebound from the old placeholder to an audited official SS latent.
    _trainer.NATIVE_SS_GENRECON_VERSION = NATIVE_SS_NO_VGGT_VERSION
    _trainer.validate_genrecon_cache_contract = validate_official_ss_cache_contract
    _trainer.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _trainer.build_native_ss_genrecon_components = build_native_ss_no_vggt_components
    _trainer.main()


if __name__ == "__main__":
    main()
