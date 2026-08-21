#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

RUN_DIR=/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/ss_arpose_artraj_1000_s2_e4_v2
NUM_WORKERS=${NUM_WORKERS:-0}

mkdir -p "$RUN_DIR"

exec env \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}" \
  SPCONV_ALGO="${SPCONV_ALGO:-native}" \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  /home/zjr/anaconda3/envs/reconviagen/bin/python \
    ar_pose_trellis/train_ss_ar_pose.py \
    --dataset_format objaverse_pose \
    --data_root /data/ar_pose_trellis/objaverse_pose_1000_artraj_s2 \
    --weights microsoft/TRELLIS-image-large \
    --save_dir "$RUN_DIR" \
    --num_views 6 \
    --batch_size 1 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 4 \
    --lr 5e-5 \
    --cfg_drop_prob 0.1 \
    --ckpt_every_n_steps 500 \
    "$@"
