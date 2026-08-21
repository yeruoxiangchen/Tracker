# Pixal3D Multi-View Wrapper

The current mixed-10k data construction implementations are centralized in
`pose_point_depth_mv/dataset_tools/`; the legacy paths under this package remain
compatible.

This directory is an independent multi-view wrapper around `/home/zjr/Tracker/Pixal3D`.
It does not modify Pixal3D source code.

## Goal

Input:

- multi-view RGB images
- masks
- camera intrinsics
- camera extrinsics

Output:

- textured GLB mesh from Pixal3D
- optional geometry-only OBJ
- projection / visual-hull stats JSON

The wrapper keeps Pixal3D's original projection-attention model structure, but replaces single fixed-front-camera projection with multi-view projection aggregation:

```text
canonical grid point -> temporary visual-hull volume in AR/world -> camera_i -> image_i
```

Projected DINO/NAF image features are averaged over valid views and passed into Pixal3D's existing `proj` attention.

The default aggregation is not plain mask averaging. It additionally builds an approximate per-view front-depth map from the temporary visual hull and downweights query points that are behind or far away from the front visible surface:

```text
weight_i(point) = mask_i(point) * visibility_depth_weight_i(point)
```

This is a z-buffer approximation. It does not estimate the final object pose, but it prevents internal/back-side query points from freely sampling front-surface image features.

## Coordinate Strategy

The generation model does not know the true `T_M2W`. That transform is estimated later by `CoarseModel`.

For real AR/world camera poses, this wrapper estimates a temporary coarse object volume from masks and camera rays:

1. Compute mask-centroid rays in world space.
2. Estimate a rough object center from ray intersections.
3. Build a world-space cube around that center.
4. Shrink it with visual-hull mask voting.
5. Map Pixal3D's canonical cube `[-0.5, 0.5]^3` into this temporary volume only for projection feature sampling.

The exported mesh remains in Pixal3D/canonical object coordinates. The temporary visual-hull transform is written to `*.stats.json` for debugging, but it is not applied to the mesh and should not be treated as `T_M2W`.

`--object_to_world_json` and `--world_to_object_json` are only debug overrides for this internal projection volume. Normal AR inference should leave them empty.

## GOOD_MESH_TEST Example

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pixal3d_multiview/run_multiview.py \
  --manifest /home/zjr/Tracker/ar_pose_trellis/test_manifests/GOOD_MESH_TEST_arpose.json \
  --image_root /home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/images \
  --mask_root /home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/masks \
  --output /home/zjr/Tracker/pixal3d_multiview/outputs/good_mesh_test_network.glb \
  --obj_output /home/zjr/Tracker/pixal3d_multiview/outputs/good_mesh_test_network.obj \
  --stats_output /home/zjr/Tracker/pixal3d_multiview/outputs/good_mesh_test_network.stats.json \
  --max_frames 8 \
  --coords_source network \
  --vh_volume_resolution 48 \
  --vh_visibility_resolution 48 \
  --visibility_depth_tolerance_ratio 0.15 \
  --low_vram \
  --resolution 1024
```

## Sparse Training

`train_sparse_multiview.py` is the first training entry point for this wrapper.
It trains only the Pixal3D sparse-structure flow stage with the independent multi-view condition implemented here:

```text
multi-view RGB/mask/K/camera pose
  -> temporary visual-hull volume
  -> front-depth visibility weighted projection
  -> Pixal3D sparse_structure_flow_model
  -> Pixal3D sparse latent target z
```

This script does not train shape or texture yet. That is intentional: if sparse structure does not become pose-sensitive and stable, full mesh training is not useful.

### Build Training Data

`pixal3d_multiview` does not train from TRELLIS `target_coords` directly. It needs Pixal3D sparse latent targets:

```text
Objaverse/ObjaverseXL textured GLB
  -> normalized Pixal3D sparse-structure occupancy
  -> TRELLIS/Pixal3D sparse encoder
  -> ss_latent z
  -> AR-like multi-view RGB/mask/K/c2w manifest
```

Build a smoke subset first and inspect `outputs/data_previews/.../vis`. The formal dataset should not use the old flat vertex-color renderer. Prefer Blender/PBR RGB with transparent masks; if Blender is unavailable, use the nvdiffrast renderer with normal-based shading as a fallback.

Install or locate Blender before using `--renderer blender`:

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python -c \
  "from Pixal3D.data_toolkit.render_cond import _install_blender, BLENDER_PATH; _install_blender(); print(BLENDER_PATH)"
```

Smoke-test the PBR dataset builder:

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
ATTN_BACKEND=flash_attn \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u \
  pose_point_depth_mv/dataset_tools/build_objaverse_multiview_sparse_data.py \
  --renderer blender \
  --blender_path /tmp/blender-3.0.1-linux-x64/blender \
  --blender_engine CYCLES \
  --objaverse_manifest /data/Objaverse/manifest_0_5000.json \
  --output_dir /data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_smoke20 \
  --code_output_dir /home/zjr/Tracker/pixal3d_multiview/outputs/data_previews/objaverse_sparse_mv_artraj_pbr_smoke20 \
  --max_objects 20 \
  --shuffle_objects \
  --sequences_per_object 2 \
  --num_views 12 \
  --image_size 512 \
  --trajectory_mode ar_random \
  --surface_points 160000 \
  --voxel_resolution 64 \
  --min_voxels 1500 \
  --radius_min 1.8 \
  --radius_max 3.2 \
  --focal_ratio 0.95 \
  --target_jitter 0.04 \
  --lookat_jitter 0.03 \
  --camera_lateral_jitter 0.02 \
  --roll_jitter 5 \
  --trajectory_resample_attempts 24 \
  --min_complete_view_fraction 0.60 \
  --min_usable_view_fraction 0.85 \
  --max_clipped_view_fraction 0.40 \
  --min_complete_in_frame_ratio 0.95 \
  --min_usable_in_frame_ratio 0.45 \
  --min_bbox_margin_px 12 \
  --max_border_touch_views 4 \
  --val_count 4 \
  --vis_count 40
```

Then build the formal dataset under `/data`, following the storage convention used by `ar_pose_trellis`:

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
ATTN_BACKEND=flash_attn \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u \
  pose_point_depth_mv/dataset_tools/build_objaverse_multiview_sparse_data.py \
  --renderer blender \
  --blender_path /tmp/blender-3.0.1-linux-x64/blender \
  --blender_engine CYCLES \
  --objaverse_manifest /data/Objaverse/manifest_0_5000.json \
  --output_dir /data/pixal3d_multiview/objaverse_sparse_mv_artraj_5000_s2_pbr \
  --code_output_dir /home/zjr/Tracker/pixal3d_multiview/outputs/data_previews/objaverse_sparse_mv_artraj_5000_s2_pbr \
  --max_objects 5000 \
  --shuffle_objects \
  --sequences_per_object 2 \
  --num_views 12 \
  --image_size 512 \
  --trajectory_mode ar_random \
  --surface_points 160000 \
  --voxel_resolution 64 \
  --min_voxels 1500 \
  --radius_min 1.8 \
  --radius_max 3.2 \
  --focal_ratio 0.95 \
  --target_jitter 0.04 \
  --lookat_jitter 0.03 \
  --camera_lateral_jitter 0.02 \
  --roll_jitter 5 \
  --trajectory_resample_attempts 24 \
  --min_complete_view_fraction 0.60 \
  --min_usable_view_fraction 0.85 \
  --max_clipped_view_fraction 0.40 \
  --min_complete_in_frame_ratio 0.95 \
  --min_usable_in_frame_ratio 0.45 \
  --min_bbox_margin_px 12 \
  --max_border_touch_views 4 \
  --val_count 64 \
  --vis_count 96 \
  --max_low_texture_ratio 0.22
```

If Blender is unavailable, use the faster fallback. This still adds normal-based shading and applies the same quality gate:

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
ATTN_BACKEND=flash_attn \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u \
  pose_point_depth_mv/dataset_tools/build_objaverse_multiview_sparse_data.py \
  --renderer nvdiffrast \
  --shading_mode normal \
  --objaverse_manifest /data/Objaverse/manifest_0_5000.json \
  --output_dir /data/pixal3d_multiview/objaverse_sparse_mv_artraj_5000_s2_shaded \
  --code_output_dir /home/zjr/Tracker/pixal3d_multiview/outputs/data_previews/objaverse_sparse_mv_artraj_5000_s2_shaded \
  --max_objects 5000 \
  --shuffle_objects \
  --sequences_per_object 2 \
  --num_views 12 \
  --image_size 512 \
  --trajectory_mode ar_random \
  --surface_points 160000 \
  --voxel_resolution 64 \
  --min_voxels 1500 \
  --radius_min 1.8 \
  --radius_max 3.2 \
  --focal_ratio 0.95 \
  --target_jitter 0.04 \
  --lookat_jitter 0.03 \
  --camera_lateral_jitter 0.02 \
  --roll_jitter 5 \
  --trajectory_resample_attempts 24 \
  --min_complete_view_fraction 0.60 \
  --min_usable_view_fraction 0.85 \
  --max_clipped_view_fraction 0.40 \
  --min_complete_in_frame_ratio 0.95 \
  --min_usable_in_frame_ratio 0.45 \
  --min_bbox_margin_px 12 \
  --max_border_touch_views 4 \
  --val_count 64 \
  --vis_count 96 \
  --max_low_texture_ratio 0.22
```

Validate the generated manifest before training:

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pose_point_depth_mv/dataset_tools/validate_multiview_sparse_data.py \
  --manifest /data/pixal3d_multiview/objaverse_sparse_mv_artraj_5000_s2/train.json \
  --max_samples 200 \
  --expected_z_shape 8,16,16,16
```

### Training Manifest Format

Each sample needs multi-view inputs and a Pixal3D sparse latent target. The sparse latent is an `.npz` with key `z`, shaped `[C, D, H, W]`, matching the selected Pixal3D sparse flow model.

```json
{
  "extrinsics_type": "c2w",
  "image_root": "/path/to/images",
  "mask_root": "/path/to/masks",
  "latent_root": "/path/to/pixal3d_ss_latents",
  "samples": [
    {
      "uid": "object_0001_seq000",
      "ss_latent": "object_0001.npz",
      "frames": [
        {
          "image": "object_0001/view_000.png",
          "mask": "object_0001/view_000.png",
          "intrinsic": [[1000, 0, 512], [0, 1000, 512], [0, 0, 1]],
          "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        }
      ]
    }
  ]
}
```

### Sparse Training Command

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pixal3d_multiview/train_sparse_multiview.py \
  --train_manifest /path/to/pixal3d_multiview_sparse_train.json \
  --output_dir /home/zjr/Tracker/pixal3d_multiview/runs/sparse_mv_v001 \
  --sparse_flow_model TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16 \
  --max_frames 8 \
  --batch_size 1 \
  --num_workers 2 \
  --max_epochs 4 \
  --max_steps 20000 \
  --lr 2e-5 \
  --trainable proj_only \
  --amp_dtype bf16 \
  --vh_volume_resolution 48 \
  --vh_visibility_resolution 48 \
  --visibility_depth_tolerance_ratio 0.15
```

If your environment cannot import `DINOv3ViTModel` from `transformers`, use the Pixal3D-compatible environment or install the Pixal3D-required transformers build before training.

Visual-hull sparse coords variant:

```bash
cd /home/zjr/Tracker

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  pixal3d_multiview/run_multiview.py \
  --manifest /home/zjr/Tracker/ar_pose_trellis/test_manifests/GOOD_MESH_TEST_arpose.json \
  --image_root /home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/images \
  --mask_root /home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/masks \
  --output /home/zjr/Tracker/pixal3d_multiview/outputs/good_mesh_test_visual_hull.glb \
  --obj_output /home/zjr/Tracker/pixal3d_multiview/outputs/good_mesh_test_visual_hull.obj \
  --stats_output /home/zjr/Tracker/pixal3d_multiview/outputs/good_mesh_test_visual_hull.stats.json \
  --max_frames 8 \
  --coords_source visual_hull \
  --vh_volume_resolution 48 \
  --vh_visibility_resolution 48 \
  --vh_min_support_views 2 \
  --vh_min_support_ratio 0.6 \
  --vh_max_coords 12000 \
  --low_vram \
  --resolution 1024
```
