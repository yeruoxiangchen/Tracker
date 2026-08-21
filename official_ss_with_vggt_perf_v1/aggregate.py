"""Aggregate disjoint official with-VGGT Native-SS evaluation shards."""

from official_ss_with_vggt_perf_v1.model import (
    EVAL_AGGREGATE_FORMAT,
    EVAL_FORMAT,
)
from pose_point_depth_mv import aggregate_proobjaverse_official_native_ss_eval as _base


def main() -> None:
    _base.OFFICIAL_SS_EVAL = EVAL_FORMAT
    _base.OFFICIAL_SS_EVAL_AGGREGATE = EVAL_AGGREGATE_FORMAT
    _base.main()


if __name__ == "__main__":
    main()

