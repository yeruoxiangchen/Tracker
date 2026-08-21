# official 30K compact-v2 + strict-fix1 integration v1

This is an independent, versioned integration tree. It does not overwrite the
legacy official no-VGGT tree, `Tracker_perf_v1`, or `Tracker_perf_v1_fix1`.

Base compatibility contract:

- exact compact-v2 SLat/lifting matched-pair validation;
- strict `paired_combined.v1` 30K protocol support;
- legacy manifest and protocol behavior retained;
- compact cache scientific identity retained.

Imported, previously CUDA-validated strict runtime behavior:

- DDP with `device_ids=None` and no `output_device`;
- `broadcast_buffers=False`, `find_unused_parameters=False`, and
  `gradient_as_bucket_view=True`;
- DataLoader workers, persistent workers, prefetch and pinned-memory controls;
- complete lifting payload remains on CPU across DDP forward;
- view selection happens on CPU and only selected DINO/K/T tensors transfer to
  the rank-local CUDA device;
- performance controls remain outside checkpoint scientific identity.

The compact format stores `image_size` as two Python integers rather than an
all-zero `predicted_depth` tensor. The projection path therefore reads this
metadata for H/W while retaining the exact legacy `predicted_depth.shape`
fallback. It does not change projection values, condition tensors, target,
sampling order, random draws, loss, optimizer hyperparameters, EMA definition,
or model architecture.

Source-server validation is CPU/static only. The dedicated integration suite,
the compact compatibility/equivalence suite, and the strict runtime/DDP suite
pass. CUDA status remains **READY FOR F39 CUDA RETEST**, not CUDA PASS.

F39 commands are in:

`pose_point_depth_mv/F39官方30K_compact_strict_fix1_1_2_8GPU_CUDA_smoke命令_20260816.txt`
