from __future__ import annotations

import unittest
from unittest import mock

from pose_aligned_reconstruction import train_native_slat_genrecon as base_trainer
from pose_aligned_reconstruction import train_native_slat_genrecon_no_vggt as no_vggt_trainer
from pose_aligned_reconstruction.proobjaverse_official_slat_training import (
    validate_official_decoder_audit,
)
from pose_aligned_reconstruction.train_proobjaverse_official_native_slat_no_vggt import (
    main,
)


class OfficialSLatTrainingEntrypointTest(unittest.TestCase):
    def test_entrypoint_installs_official_audit_validator(self) -> None:
        original = base_trainer.validate_decoder_audit
        try:
            with mock.patch.object(no_vggt_trainer, "main") as delegated:
                main()
            delegated.assert_called_once_with()
            self.assertIs(
                base_trainer.validate_decoder_audit,
                validate_official_decoder_audit,
            )
        finally:
            base_trainer.validate_decoder_audit = original


if __name__ == "__main__":
    unittest.main()
