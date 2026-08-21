"""Standalone multi-view pose-point-depth evidence and local Flow probes.

This package is intentionally separate from :mod:`ar_ss_flow`. It only reads
existing pose-lifting caches. Legacy PPD-2 can audit an existing residual;
PPD-3A trains a separate low-rank same-voxel learnability probe.
"""

PACKAGE_VERSION = "pose_point_depth_mv.v3"
