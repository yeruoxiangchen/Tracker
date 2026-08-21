#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-1}
MODE=${MODE:-smoke}
RUN_NAME=${RUN_NAME:-pointprior_pixal_v9_${MODE}}

DATA_ROOT=${DATA_ROOT:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8}
TRAIN_SOURCE=${TRAIN_SOURCE:-$DATA_ROOT/train.json}
VAL_SOURCE=${VAL_SOURCE:-$DATA_ROOT/val.json}
WEIGHTS=${WEIGHTS:-microsoft/TRELLIS-image-large}
OUT_ROOT=${OUT_ROOT:-$ROOT/trellis_point_prior_mv/outputs}

RUN_BUILD=${RUN_BUILD:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_INDICES=${TRAIN_INDICES:-0-63}
  VAL_INDICES=${VAL_INDICES:-0-31}
  MAX_STEPS=${MAX_STEPS:-20}
  EVAL_INDICES=${EVAL_INDICES:-0-7}
  EVAL_STEPS=${EVAL_STEPS:-8}
elif [[ "$MODE" == "s200" ]]; then
  TRAIN_INDICES=${TRAIN_INDICES:-0-511}
  VAL_INDICES=${VAL_INDICES:-0-63}
  MAX_STEPS=${MAX_STEPS:-200}
  EVAL_INDICES=${EVAL_INDICES:-0-31}
  EVAL_STEPS=${EVAL_STEPS:-12}
else
  TRAIN_INDICES=${TRAIN_INDICES:-all}
  VAL_INDICES=${VAL_INDICES:-0-127}
  MAX_STEPS=${MAX_STEPS:-1000}
  EVAL_INDICES=${EVAL_INDICES:-0-63}
  EVAL_STEPS=${EVAL_STEPS:-12}
fi

MAX_FRAMES=${MAX_FRAMES:-8}
POINT_COUNT_CHOICES=${POINT_COUNT_CHOICES:-50,100,300,800,1500}
NUM_PRIOR_VIEWS_CHOICES=${NUM_PRIOR_VIEWS_CHOICES:-1,2,4,8}
MIN_SUPPORT=${MIN_SUPPORT:-1.0}
MIN_SUPPORT_RATIO=${MIN_SUPPORT_RATIO:-0.45}
DROPOUT_MIN=${DROPOUT_MIN:-0.0}
DROPOUT_MAX=${DROPOUT_MAX:-0.65}
COORD_JITTER=${COORD_JITTER:-1}
OUTLIER_RATIO=${OUTLIER_RATIO:-0.03}
FRONT_DEPTH_EPSILON=${FRONT_DEPTH_EPSILON:-0.02}
NO_FRONT_DEPTH=${NO_FRONT_DEPTH:-0}
ALLOW_SUPPORT_FALLBACK=${ALLOW_SUPPORT_FALLBACK:-0}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-1e-4}
CFG_DROP_PROB=${CFG_DROP_PROB:-0.1}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
SAVE_EVERY=${SAVE_EVERY:-100}

PRIOR_MODES=${PRIOR_MODES:-correct,empty,shuffle,random,jitter}
FIXED_TOPK=${FIXED_TOPK:-4096,8192,target_unique}
GUIDANCE_STRENGTH=${GUIDANCE_STRENGTH:-1.0}

RUN_DIR="$OUT_ROOT/$RUN_NAME"
TRAIN_PRIOR_DIR="$RUN_DIR/data/train"
VAL_PRIOR_DIR="$RUN_DIR/data/val"
CKPT_DIR="$RUN_DIR/checkpoints"
EVAL_DIR="$RUN_DIR/eval"

COMMON_ENV=(
  CUDA_VISIBLE_DEVICES="$GPU"
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  ATTN_BACKEND=flash_attn
  SPCONV_ALGO=native
  MPLCONFIGDIR=/tmp/matplotlib
  NUMBA_CACHE_DIR=/tmp/numba_cache
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

BUILD_EXTRA_ARGS=(--front_depth_epsilon "$FRONT_DEPTH_EPSILON")
if [[ "$NO_FRONT_DEPTH" == "1" ]]; then
  BUILD_EXTRA_ARGS+=(--no_front_depth)
fi
if [[ "$ALLOW_SUPPORT_FALLBACK" == "1" ]]; then
  BUILD_EXTRA_ARGS+=(--allow_support_fallback)
fi

cd "$ROOT"
mkdir -p "$RUN_DIR"

if [[ "$RUN_BUILD" == "1" ]]; then
  echo "[pointprior_stage1] build train priors -> $TRAIN_PRIOR_DIR"
  "$PY" -u trellis_point_prior_mv/build_point_prior_dataset.py \
    --source_manifest "$TRAIN_SOURCE" \
    --output_dir "$TRAIN_PRIOR_DIR" \
    --indices "$TRAIN_INDICES" \
    --max_frames "$MAX_FRAMES" \
    --grid_transform pixal3d_rotation \
    --num_prior_views_choices "$NUM_PRIOR_VIEWS_CHOICES" \
    --point_count_choices "$POINT_COUNT_CHOICES" \
    --min_support "$MIN_SUPPORT" \
    --min_support_ratio "$MIN_SUPPORT_RATIO" \
    --dropout_min "$DROPOUT_MIN" \
    --dropout_max "$DROPOUT_MAX" \
    --coord_jitter "$COORD_JITTER" \
    --outlier_ratio "$OUTLIER_RATIO" \
    "${BUILD_EXTRA_ARGS[@]}"

  echo "[pointprior_stage1] build val priors -> $VAL_PRIOR_DIR"
  "$PY" -u trellis_point_prior_mv/build_point_prior_dataset.py \
    --source_manifest "$VAL_SOURCE" \
    --output_dir "$VAL_PRIOR_DIR" \
    --indices "$VAL_INDICES" \
    --max_frames "$MAX_FRAMES" \
    --grid_transform pixal3d_rotation \
    --num_prior_views_choices "$NUM_PRIOR_VIEWS_CHOICES" \
    --point_count_choices "$POINT_COUNT_CHOICES" \
    --min_support "$MIN_SUPPORT" \
    --min_support_ratio "$MIN_SUPPORT_RATIO" \
    --dropout_min "$DROPOUT_MIN" \
    --dropout_max "$DROPOUT_MAX" \
    --coord_jitter "$COORD_JITTER" \
    --outlier_ratio "$OUTLIER_RATIO" \
    "${BUILD_EXTRA_ARGS[@]}"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[pointprior_stage1] train max_steps=$MAX_STEPS -> $CKPT_DIR"
  env "${COMMON_ENV[@]}" "$PY" -u trellis_point_prior_mv/train_sparse_inpaint.py \
    --manifest "$TRAIN_PRIOR_DIR/manifest.json" \
    --weights "$WEIGHTS" \
    --save_dir "$CKPT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 100 \
    --max_steps "$MAX_STEPS" \
    --lr "$LR" \
    --cfg_drop_prob "$CFG_DROP_PROB" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --ckpt_every_n_steps "$SAVE_EVERY"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  CKPT=$(find "$CKPT_DIR" -maxdepth 1 -name "*.ckpt" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)
  test -f "$CKPT" || { echo "No checkpoint found in $CKPT_DIR" >&2; exit 1; }
  echo "[pointprior_stage1] eval ckpt=$CKPT -> $EVAL_DIR"
  env "${COMMON_ENV[@]}" "$PY" -u trellis_point_prior_mv/eval_sparse_inpaint.py \
    --manifest "$VAL_PRIOR_DIR/manifest.json" \
    --checkpoint "$CKPT" \
    --output_dir "$EVAL_DIR" \
    --weights "$WEIGHTS" \
    --indices "$EVAL_INDICES" \
    --prior_modes "$PRIOR_MODES" \
    --fixed_topk "$FIXED_TOPK" \
    --steps "$EVAL_STEPS" \
    --guidance_strength "$GUIDANCE_STRENGTH" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA"
fi

echo "[pointprior_stage1] done: $RUN_DIR"
