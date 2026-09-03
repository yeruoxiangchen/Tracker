#!/usr/bin/env python3
"""Train no-VGGT Native SS on audited official ProObjaverse SS targets."""

from __future__ import annotations

from pose_aligned_reconstruction import train_native_ss_genrecon as _trainer
from pose_aligned_reconstruction.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_aligned_reconstruction.proobjaverse_official_ss import (
    validate_official_ss_cache_contract,
)


def load_official_training_dataset(manifest: str, *, indices: str = "all"):
    """Load either a legacy lifting cache or the paper's compact 30K cache."""

    from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
    from pose_aligned_reconstruction.proobjaverse_official_ss_compact import (
        CompactOfficialSSDataset,
        is_official_ss_compact_manifest,
    )

    if is_official_ss_compact_manifest(manifest):
        return CompactOfficialSSDataset(manifest, indices=indices)
    return PoseLiftingCacheDataset(manifest, indices=indices)


def main() -> None:
    # Reuse the proven DDP/EMA/optimizer loop, but fail closed unless every row
    # is rebound from the old placeholder to an audited official SS latent.
    _trainer.NATIVE_SS_GENRECON_VERSION = NATIVE_SS_NO_VGGT_VERSION
    _trainer.validate_genrecon_cache_contract = validate_official_ss_cache_contract
    _trainer.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _trainer.build_native_ss_genrecon_components = build_native_ss_no_vggt_components
    _trainer.PoseLiftingCacheDataset = load_official_training_dataset
    _trainer.main()


if __name__ == "__main__":
    main()
