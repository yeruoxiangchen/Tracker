#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-1}

INPUT_TYPE=${INPUT_TYPE:-coarse_dataset}
DATASET_DIR=${DATASET_DIR:-$ROOT/CoarseModel/datasets/reconviagen_20260617_073549}
PHONE_DATA_DIR=${PHONE_DATA_DIR:-}
PHONE_MASK_DIR=${PHONE_MASK_DIR:-}
CASE_NAME=${CASE_NAME:-$(basename "$DATASET_DIR")_phone_pose_guided}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/CoarseModel/pose_guided_recon/outputs}

RUN_RECON=${RUN_RECON:-1}
SEEDS=${SEEDS:-0}
EXISTING_MESHES=${EXISTING_MESHES:-}
MAX_INPUT_FRAMES=${MAX_INPUT_FRAMES:-18}
MIN_INPUT_FRAMES=${MIN_INPUT_FRAMES:-8}
SCORE_ALL_FRAMES=${SCORE_ALL_FRAMES:-1}
MAX_SCORE_POINTS=${MAX_SCORE_POINTS:-12000}
YAW_STEPS=${YAW_STEPS:-24}
SCALE_FACTORS=${SCALE_FACTORS:-0.75,0.9,1.0,1.1,1.25}
POINT_DILATION=${POINT_DILATION:-9}
APPLY_VISUAL_HULL_FILTER=${APPLY_VISUAL_HULL_FILTER:-0}
VH_INSIDE_RATIO_THRESHOLD=${VH_INSIDE_RATIO_THRESHOLD:-0.45}
VH_MIN_VISIBLE_VIEWS=${VH_MIN_VISIBLE_VIEWS:-3}
VH_MASK_DILATION=${VH_MASK_DILATION:-7}

CMD=(
  "$PY" -u "$ROOT/CoarseModel/pose_guided_recon/pose_guided_recon.py"
  --input_type "$INPUT_TYPE"
  --pose_source phone_ar
  --case_name "$CASE_NAME"
  --output_root "$OUTPUT_ROOT"
  --python_bin "$PY"
  --run_recon "$RUN_RECON"
  --seeds "$SEEDS"
  --max_input_frames "$MAX_INPUT_FRAMES"
  --min_input_frames "$MIN_INPUT_FRAMES"
  --score_all_frames "$SCORE_ALL_FRAMES"
  --max_score_points "$MAX_SCORE_POINTS"
  --yaw_steps "$YAW_STEPS"
  --scale_factors "$SCALE_FACTORS"
  --point_dilation "$POINT_DILATION"
  --apply_visual_hull_filter "$APPLY_VISUAL_HULL_FILTER"
  --vh_inside_ratio_threshold "$VH_INSIDE_RATIO_THRESHOLD"
  --vh_min_visible_views "$VH_MIN_VISIBLE_VIEWS"
  --vh_mask_dilation "$VH_MASK_DILATION"
)

if [[ "$INPUT_TYPE" == "phone_ar_session" ]]; then
  if [[ -z "$PHONE_DATA_DIR" ]]; then
    echo "PHONE_DATA_DIR is required when INPUT_TYPE=phone_ar_session" >&2
    exit 2
  fi
  CMD+=(--phone_data_dir "$PHONE_DATA_DIR")
  if [[ -n "$PHONE_MASK_DIR" ]]; then
    CMD+=(--phone_mask_dir "$PHONE_MASK_DIR")
  fi
else
  CMD+=(--dataset_dir "$DATASET_DIR")
fi

if [[ -n "$EXISTING_MESHES" ]]; then
  CMD+=(--existing_meshes "$EXISTING_MESHES")
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${CMD[@]}"
