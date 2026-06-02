# AR-Pose TRELLIS Benchmark

This directory contains lightweight wrappers for comparing:

- `ar_pose_trellis`: the AR-pose-conditioned TRELLIS sparse-structure generator.
- `ReconViaGen`: direct multi-view image-to-mesh generation, without `server.py`, `run_local`, or CoarseModel refinement.

Keep real captures conservative. The default real test list only contains `GOOD_MESH_TEST`; most older `CoarseModel/datasets/*` captures are not clean enough for generator-level evaluation.

## 1. Export Synthetic Holdout Cases

Use a few held-out Objaverse pose samples as controlled sanity tests. These cases contain rendered images, masks, camera poses, and sparse target coordinates for fast reference-shape evaluation.

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/benchmark/export_objaverse_holdout_cases.py \
  --data_root /data/ar_pose_trellis/objaverse_pose_1000_artraj_s2 \
  --split val \
  --max_cases 4 \
  --max_views 8 \
  --output_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_holdout_cases \
  --testsets_out /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_holdout_testsets.json
```

## 2. Run AR-Pose TRELLIS

Use `--dry_run` first if you only want to inspect the generated commands.

```bash
cd /home/zjr/Tracker

ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/benchmark/run_arpose_batch.py \
  --testsets /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_holdout_testsets.json \
  --checkpoint /home/zjr/Tracker/ar_pose_trellis/runs/ss_arpose_objaverse_1000_e8_v4_finish/last.ckpt \
  --output_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/arpose_holdout \
  --max_frames 8 \
  --ss_steps 12 \
  --slat_steps 12 \
  --cond_fp16
```

For the real `GOOD_MESH_TEST` case, change `--testsets` and `--output_root`:

```bash
--testsets ar_pose_trellis/benchmark/testsets.json \
--output_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/arpose_real
```

## 3. Run ReconViaGen Directly

This wrapper feeds multi-view images and masks directly into ReconViaGen and writes outputs under a controlled benchmark directory.

```bash
cd /home/zjr/Tracker

ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/benchmark/run_reconviagen_batch.py \
  --testsets /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_holdout_testsets.json \
  --output_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/reconviagen_holdout \
  --max_frames 8 \
  --seeds 0 \
  --skip_video
```

For `GOOD_MESH_TEST`, use `ar_pose_trellis/benchmark/testsets.json` and a separate output root such as `benchmark_outputs/reconviagen_real`.

## 4. Evaluate Meshes

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/benchmark/evaluate_meshes.py \
  --testsets /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/objaverse_holdout_testsets.json \
  --arpose_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/arpose_holdout \
  --reconviagen_root /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/reconviagen_holdout \
  --output_dir /home/zjr/Tracker/ar_pose_trellis/benchmark_outputs/eval_holdout \
  --sample_points 4000
```

The evaluator computes normalized shape statistics, Chamfer/F-score between available methods, and a projection contact sheet. For synthetic holdout cases, the reference is `target_coords.npy`, not the full Objaverse GLB, so this is a sparse-structure sanity metric. For `GOOD_MESH_TEST`, the reference is the normalized model under `CoarseModel/datasets/GOOD_MESH_TEST/models`.
