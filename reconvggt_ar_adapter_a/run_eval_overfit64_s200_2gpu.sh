#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

ROOT=/data/reconvggt_pointpose_v9_ssfixed_odsplit_20260712
RUN=/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/pointpose_ss_lora_ssfixed_overfit64_s200_2gpu_fp16
EVAL_OUTPUT=${RUN}/eval_overfit64_noise_t1_seeds424344_v2
GPU=${GPU:-1}

/home/zjr/anaconda3/envs/reconviagen/bin/python -u \
  reconvggt_ar_adapter_a/audit_pointpose_training_run.py \
  --train_report "${RUN}/train_report.json" \
  --checkpoint "${RUN}/checkpoints/last.pt" \
  --expected_updates 200 \
  --max_nonfinite_attempts 0 \
  --output "${RUN}/finite_run_audit.json"

if [[ -e "${EVAL_OUTPUT}/report.json" ]]; then
  echo "evaluation report already exists: ${EVAL_OUTPUT}/report.json" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zjr/anaconda3/envs/reconviagen/bin/python -u \
  reconvggt_ar_adapter_a/eval_pointpose_ss_lora.py \
  --cache_manifest "${ROOT}/cache/train/overfit64.json" \
  --checkpoint "${RUN}/checkpoints/last.pt" \
  --output_dir "${EVAL_OUTPUT}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --indices all \
  --max_samples 0 \
  --device cuda \
  --seeds 42,43,44 \
  --steps 30 \
  --cfg_strength 7.5 \
  --guidance_rescale 0.5 \
  --rescale_t 3.0 \
  --physical_scale 1.0 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --physical_hidden_dim 256 \
  --physical_heads 8 \
  --bridge_train_last_blocks 0 \
  2>&1 | tee "${EVAL_OUTPUT}.log"
