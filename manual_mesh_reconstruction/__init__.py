"""Isolated manual Mesh-reconstruction and review package.

The package owns the human-facing runtime-O, inference, rendering, projection,
and orchestration code.  It intentionally imports the already validated model
architectures from :mod:`pose_point_depth_mv` instead of forking a second copy
of the trainable network definitions.
"""

PACKAGE_VERSION = "manual_mesh_reconstruction.20260819.v3_official_z_up"
