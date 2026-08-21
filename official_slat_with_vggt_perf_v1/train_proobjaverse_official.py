"""Official ProObjaverse with-VGGT SLat strict-performance entrypoint."""

from __future__ import annotations

from pose_point_depth_mv.proobjaverse_official_slat_training import (
    validate_official_decoder_audit,
)

from . import train as _arm


def main() -> None:
    _arm.main(decoder_validator=validate_official_decoder_audit)


if __name__ == "__main__":
    main()
