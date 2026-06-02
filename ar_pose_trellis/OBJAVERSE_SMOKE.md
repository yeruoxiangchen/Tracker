# Objaverse AR-Pose Sparse-Structure Smoke

This directory keeps code, logs, checkpoints, and visualizations. Generated
training data lives under `/data/ar_pose_trellis`.

## Build A Small Objaverse Pose Dataset

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/dataset_tools/build_objaverse_pose_data.py \
  --objaverse_manifest /data/Objaverse/manifest_0_5000.json \
  --output_dir /data/ar_pose_trellis/objaverse_pose_smoke \
  --code_output_dir /home/zjr/Tracker/ar_pose_trellis/smoke_outputs/data_build \
  --max_objects 3 \
  --num_views 4 \
  --image_size 128 \
  --surface_points 30000 \
  --point_radius 1 \
  --val_count 1
```

Outputs:

```text
/data/ar_pose_trellis/objaverse_pose_smoke/{manifest,train,val}.json
/data/ar_pose_trellis/objaverse_pose_smoke/samples/*/*.npz
/home/zjr/Tracker/ar_pose_trellis/smoke_outputs/data_build/vis/*.jpg
```

The current builder is a CPU software point-splat renderer. It is intended for
pipeline smoke tests. For final-quality training data, replace it with a real
mesh renderer that outputs RGB/mask/depth/normal/camera.

## CPU Smoke Training

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/train_ss_ar_pose_smoke.py \
  --data_root /data/ar_pose_trellis/objaverse_pose_smoke \
  --split train \
  --output_dir /home/zjr/Tracker/ar_pose_trellis/smoke_runs/objaverse_pose_smoke_cpu \
  --num_views 4 \
  --batch_size 1 \
  --steps 5 \
  --patch_side 8 \
  --channels 32 \
  --tokens 64 \
  --lr 1e-3
```

Outputs:

```text
/home/zjr/Tracker/ar_pose_trellis/smoke_runs/objaverse_pose_smoke_cpu/ckpt_last.pt
/home/zjr/Tracker/ar_pose_trellis/smoke_runs/objaverse_pose_smoke_cpu/train_log.csv
/home/zjr/Tracker/ar_pose_trellis/smoke_runs/objaverse_pose_smoke_cpu/vis/*.jpg
```

This CPU smoke trains `ARDinoRayCond` with a tiny target head. It validates data
reading, intrinsics/extrinsics handling, mask-conditioned rays, backprop, and
artifact writing. It is not the final TRELLIS sparse-structure flow finetune.

## Full Sparse-Structure Training Entry

`train_ss_ar_pose.py` now supports the generated Objaverse pose dataset:

```bash
cd /home/zjr/Tracker

ATTN_BACKEND=flash_attn \
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  ar_pose_trellis/train_ss_ar_pose.py \
  --dataset_format objaverse_pose \
  --data_root /data/ar_pose_trellis/objaverse_pose_smoke \
  --split train \
  --weights /home/zjr/.cache/huggingface/hub/models--Stable-X--trellis-vggt-v0-2/snapshots/647659a5ad5fbf67e22793e7b5e2cee4b30c5d13 \
  --save_dir /home/zjr/Tracker/ar_pose_trellis/runs/ss_arpose_objaverse \
  --num_views 4 \
  --batch_size 1 \
  --max_epochs 1 \
  --limit_train_batches 2
```

The full entry requires CUDA and `pytorch_lightning` in the active environment.
