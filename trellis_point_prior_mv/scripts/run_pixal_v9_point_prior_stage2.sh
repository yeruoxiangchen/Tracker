#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:-1}
MODE=${MODE:-smoke}
RUN_NAME=${RUN_NAME:-pointprior_pixal_v9_stage2_${MODE}}

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
  EVAL_STEPS=${EVAL_STEPS:-12}
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
POINT_COUNT_CHOICES=${POINT_COUNT_CHOICES:-1500}
NUM_PRIOR_VIEWS_CHOICES=${NUM_PRIOR_VIEWS_CHOICES:-8}
MIN_SUPPORT=${MIN_SUPPORT:-1.0}
MIN_SUPPORT_RATIO=${MIN_SUPPORT_RATIO:-0.45}
DROPOUT_MIN=${DROPOUT_MIN:-0.0}
DROPOUT_MAX=${DROPOUT_MAX:-0.0}
COORD_JITTER=${COORD_JITTER:-0}
OUTLIER_RATIO=${OUTLIER_RATIO:-0.0}
FRONT_DEPTH_EPSILON=${FRONT_DEPTH_EPSILON:-0.02}
NO_FRONT_DEPTH=${NO_FRONT_DEPTH:-0}
ALLOW_SUPPORT_FALLBACK=${ALLOW_SUPPORT_FALLBACK:-0}
BUILD_SEED=${BUILD_SEED:-42}
TRAIN_SEED=${TRAIN_SEED:-42}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-1e-4}
CFG_DROP_PROB=${CFG_DROP_PROB:-0.05}
KNOWN_FLOW_LOSS_WEIGHT=${KNOWN_FLOW_LOSS_WEIGHT:-2.0}
KNOWN_X0_LOSS_WEIGHT=${KNOWN_X0_LOSS_WEIGHT:-1.0}
KNOWN_CONF_POWER=${KNOWN_CONF_POWER:-1.0}
KNOWN_USE_CONFIDENCE=${KNOWN_USE_CONFIDENCE:-0}
ANTI_OVERFILL_LOSS_WEIGHT=${ANTI_OVERFILL_LOSS_WEIGHT:-0.0}
ANTI_OVERFILL_MARGIN=${ANTI_OVERFILL_MARGIN:-0.0}
RANKING_LOSS_WEIGHT=${RANKING_LOSS_WEIGHT:-0.0}
RANKING_MARGIN=${RANKING_MARGIN:-0.05}
RANKING_NEGATIVE_MODES=${RANKING_NEGATIVE_MODES:-shuffle,random}
RANKING_OUTSIDE_WEIGHT=${RANKING_OUTSIDE_WEIGHT:-1.0}
RANKING_OBSERVED_WEIGHT=${RANKING_OBSERVED_WEIGHT:-1.0}
RANKING_WRONG_SUPPORT_WEIGHT=${RANKING_WRONG_SUPPORT_WEIGHT:-1.0}
RANKING_TARGET_SUPPORT_WEIGHT=${RANKING_TARGET_SUPPORT_WEIGHT:-0.0}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
SAVE_EVERY=${SAVE_EVERY:-100}

PRIOR_MODES=${PRIOR_MODES:-correct,empty,shuffle,random,jitter}
FIXED_TOPK=${FIXED_TOPK:-4096,8192,target_unique}
GUIDANCE_STRENGTH=${GUIDANCE_STRENGTH:-1.0}
KNOWN_LATENT_CLAMP_STRENGTH=${KNOWN_LATENT_CLAMP_STRENGTH:-1.0}
KNOWN_CLAMP_START_T=${KNOWN_CLAMP_START_T:-1.0}
KNOWN_LOGIT_BOOST=${KNOWN_LOGIT_BOOST:-0.0}
CLAMP_INITIAL_NOISE=${CLAMP_INITIAL_NOISE:-1}

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

EVAL_EXTRA_ARGS=()
if [[ "$CLAMP_INITIAL_NOISE" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--clamp_initial_noise)
else
  EVAL_EXTRA_ARGS+=(--no_clamp_initial_noise)
fi
if [[ "$KNOWN_USE_CONFIDENCE" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--known_use_confidence)
fi

TRAIN_EXTRA_ARGS=()
if [[ "$KNOWN_USE_CONFIDENCE" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--known_use_confidence)
fi

cd "$ROOT"
mkdir -p "$RUN_DIR"

if [[ "$RUN_BUILD" == "1" ]]; then
  echo "[pointprior_stage2] build train priors -> $TRAIN_PRIOR_DIR"
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
    --seed "$BUILD_SEED" \
    "${BUILD_EXTRA_ARGS[@]}"

  echo "[pointprior_stage2] build val priors -> $VAL_PRIOR_DIR"
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
    --seed "$BUILD_SEED" \
    "${BUILD_EXTRA_ARGS[@]}"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[pointprior_stage2] train max_steps=$MAX_STEPS -> $CKPT_DIR"
  env "${COMMON_ENV[@]}" "$PY" -u trellis_point_prior_mv/train_sparse_inpaint_stage2.py \
    --manifest "$TRAIN_PRIOR_DIR/manifest.json" \
    --weights "$WEIGHTS" \
    --save_dir "$CKPT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 100 \
    --max_steps "$MAX_STEPS" \
    --lr "$LR" \
    --seed "$TRAIN_SEED" \
    --cfg_drop_prob "$CFG_DROP_PROB" \
    --known_flow_loss_weight "$KNOWN_FLOW_LOSS_WEIGHT" \
    --known_x0_loss_weight "$KNOWN_X0_LOSS_WEIGHT" \
    --known_conf_power "$KNOWN_CONF_POWER" \
    --anti_overfill_loss_weight "$ANTI_OVERFILL_LOSS_WEIGHT" \
    --anti_overfill_margin "$ANTI_OVERFILL_MARGIN" \
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT" \
    --ranking_margin "$RANKING_MARGIN" \
    --ranking_negative_modes "$RANKING_NEGATIVE_MODES" \
    --ranking_outside_weight "$RANKING_OUTSIDE_WEIGHT" \
    --ranking_observed_weight "$RANKING_OBSERVED_WEIGHT" \
    --ranking_wrong_support_weight "$RANKING_WRONG_SUPPORT_WEIGHT" \
    --ranking_target_support_weight "$RANKING_TARGET_SUPPORT_WEIGHT" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --ckpt_every_n_steps "$SAVE_EVERY" \
    "${TRAIN_EXTRA_ARGS[@]}"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  CKPT=$(find "$CKPT_DIR" -maxdepth 1 -name "*.ckpt" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)
  test -f "$CKPT" || { echo "No checkpoint found in $CKPT_DIR" >&2; exit 1; }
  echo "[pointprior_stage2] eval ckpt=$CKPT -> $EVAL_DIR"
  env "${COMMON_ENV[@]}" "$PY" -u trellis_point_prior_mv/eval_sparse_inpaint_stage2.py \
    --manifest "$VAL_PRIOR_DIR/manifest.json" \
    --checkpoint "$CKPT" \
    --output_dir "$EVAL_DIR" \
    --weights "$WEIGHTS" \
    --indices "$EVAL_INDICES" \
    --prior_modes "$PRIOR_MODES" \
    --fixed_topk "$FIXED_TOPK" \
    --steps "$EVAL_STEPS" \
    --guidance_strength "$GUIDANCE_STRENGTH" \
    --known_latent_clamp_strength "$KNOWN_LATENT_CLAMP_STRENGTH" \
    --known_clamp_start_t "$KNOWN_CLAMP_START_T" \
    --known_logit_boost "$KNOWN_LOGIT_BOOST" \
    --known_conf_power "$KNOWN_CONF_POWER" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    "${EVAL_EXTRA_ARGS[@]}"
fi

echo "[pointprior_stage2] done: $RUN_DIR"
