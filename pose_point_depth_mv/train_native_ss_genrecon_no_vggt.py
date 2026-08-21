#!/usr/bin/env python3
"""Train no-VGGT Native SS without changing the frozen v2 implementation."""

from __future__ import annotations

from pose_point_depth_mv import train_native_ss_genrecon as _v2_train
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
    validate_no_vggt_cache_contract,
)


def main() -> None:
    # The v2 trainer owns the proven optimization loop.  Only its injected
    # protocol hooks are replaced in this new entrypoint.
    _v2_train.NATIVE_SS_GENRECON_VERSION = NATIVE_SS_NO_VGGT_VERSION
    _v2_train.validate_genrecon_cache_contract = validate_no_vggt_cache_contract
    _v2_train.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _v2_train.build_native_ss_genrecon_components = (
        build_native_ss_no_vggt_components
    )
    _v2_train.main()


if __name__ == "__main__":
    main()
