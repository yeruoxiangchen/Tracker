#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zjr/Tracker}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
GPU="${GPU:-1}"
MODE="${MODE:-smoke}"

DATA_ROOT="${DATA_ROOT:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$DATA_ROOT/train.json}"
VAL_SOURCE="${VAL_SOURCE:-$DATA_ROOT/val.json}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
OUT_ROOT="${OUT_ROOT:-$ROOT/trellis_point_prior_mv/outputs}"

SWEEP_NAME="${SWEEP_NAME:-pointprior_pixal_v9_stage2_antioverfill_weight_sweep_${MODE}}"
SWEEP_WEIGHTS="${SWEEP_WEIGHTS:-0.005,0.01,0.02,0.04}"
TOPK_SPECS="${TOPK_SPECS:-r0.35_cap4096,r0.50_cap8192,target_unique}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_MESH_EVAL="${RUN_MESH_EVAL:-1}"

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_INDICES="${TRAIN_INDICES:-0-63}"
  VAL_INDICES="${VAL_INDICES:-0-31}"
  MAX_STEPS="${MAX_STEPS:-20}"
  EVAL_INDICES="${EVAL_INDICES:-0-7}"
elif [[ "$MODE" == "s200" ]]; then
  TRAIN_INDICES="${TRAIN_INDICES:-0-511}"
  VAL_INDICES="${VAL_INDICES:-0-63}"
  MAX_STEPS="${MAX_STEPS:-200}"
  EVAL_INDICES="${EVAL_INDICES:-0-31}"
else
  echo "Unsupported MODE=${MODE}. Use MODE=smoke or MODE=s200." >&2
  exit 2
fi

MAX_FRAMES="${MAX_FRAMES:-8}"
POINT_COUNT_CHOICES="${POINT_COUNT_CHOICES:-1500}"
NUM_PRIOR_VIEWS_CHOICES="${NUM_PRIOR_VIEWS_CHOICES:-8}"
MIN_SUPPORT="${MIN_SUPPORT:-1.0}"
MIN_SUPPORT_RATIO="${MIN_SUPPORT_RATIO:-0.45}"
DROPOUT_MIN="${DROPOUT_MIN:-0.0}"
DROPOUT_MAX="${DROPOUT_MAX:-0.0}"
OUTLIER_RATIO="${OUTLIER_RATIO:-0.0}"
COORD_JITTER="${COORD_JITTER:-0}"
FRONT_DEPTH_EPSILON="${FRONT_DEPTH_EPSILON:-0.02}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-4}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0.05}"
KNOWN_FLOW_LOSS_WEIGHT="${KNOWN_FLOW_LOSS_WEIGHT:-2.0}"
KNOWN_X0_LOSS_WEIGHT="${KNOWN_X0_LOSS_WEIGHT:-0.5}"
KNOWN_USE_CONFIDENCE="${KNOWN_USE_CONFIDENCE:-0}"
KNOWN_CONF_POWER="${KNOWN_CONF_POWER:-1.0}"
ANTI_OVERFILL_MARGIN="${ANTI_OVERFILL_MARGIN:-0.0}"
KNOWN_CLAMP_START_T="${KNOWN_CLAMP_START_T:-0.5}"
CLAMP_INITIAL_NOISE="${CLAMP_INITIAL_NOISE:-0}"
KNOWN_LATENT_CLAMP_STRENGTH="${KNOWN_LATENT_CLAMP_STRENGTH:-1.0}"
KNOWN_LOGIT_BOOST="${KNOWN_LOGIT_BOOST:-0.0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_STEPS="${EVAL_STEPS:-12}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-4000}"

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

cd "$ROOT"

SWEEP_ROOT="$OUT_ROOT/$SWEEP_NAME"
TRAIN_PRIOR_DIR="$SWEEP_ROOT/data/train"
VAL_PRIOR_DIR="$SWEEP_ROOT/data/val"
mkdir -p "$SWEEP_ROOT"

if [[ "$RUN_BUILD" == "1" ]]; then
  echo "[stage2_weight_sweep] build train priors -> $TRAIN_PRIOR_DIR"
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
    --front_depth_epsilon "$FRONT_DEPTH_EPSILON"

  echo "[stage2_weight_sweep] build val priors -> $VAL_PRIOR_DIR"
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
    --front_depth_epsilon "$FRONT_DEPTH_EPSILON"
fi

IFS=',' read -ra WEIGHT_LIST <<< "$SWEEP_WEIGHTS"
for WEIGHT in "${WEIGHT_LIST[@]}"; do
  WEIGHT="$(echo "$WEIGHT" | xargs)"
  [[ -n "$WEIGHT" ]] || continue
  LABEL="w${WEIGHT//./p}"
  RUN_DIR="$OUT_ROOT/${SWEEP_NAME}_${LABEL}"
  CKPT_DIR="$RUN_DIR/checkpoints"
  EVAL_DIR="$RUN_DIR/eval"
  MESH_RUN_NAME="${SWEEP_NAME}_${LABEL}_mesh_relcap"

  mkdir -p "$RUN_DIR"
  echo "[stage2_weight_sweep] weight=$WEIGHT run=$RUN_DIR"

  if [[ "$RUN_TRAIN" == "1" ]]; then
    env "${COMMON_ENV[@]}" "$PY" -u trellis_point_prior_mv/train_sparse_inpaint_stage2.py \
      --manifest "$TRAIN_PRIOR_DIR/manifest.json" \
      --weights "$WEIGHTS" \
      --save_dir "$CKPT_DIR" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --max_epochs 100 \
      --max_steps "$MAX_STEPS" \
      --lr "$LR" \
      --cfg_drop_prob "$CFG_DROP_PROB" \
      --known_flow_loss_weight "$KNOWN_FLOW_LOSS_WEIGHT" \
      --known_x0_loss_weight "$KNOWN_X0_LOSS_WEIGHT" \
      --known_conf_power "$KNOWN_CONF_POWER" \
      --anti_overfill_loss_weight "$WEIGHT" \
      --anti_overfill_margin "$ANTI_OVERFILL_MARGIN" \
      --lora_rank "$LORA_RANK" \
      --lora_alpha "$LORA_ALPHA" \
      --ckpt_every_n_steps "$SAVE_EVERY" \
      $( [[ "$KNOWN_USE_CONFIDENCE" == "1" ]] && printf '%s' "--known_use_confidence" )
  fi

  CKPT=$(find "$CKPT_DIR" -maxdepth 1 -name "*.ckpt" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)
  test -f "$CKPT" || { echo "No checkpoint found in $CKPT_DIR" >&2; exit 1; }

  if [[ "$RUN_EVAL" == "1" ]]; then
    env "${COMMON_ENV[@]}" "$PY" -u trellis_point_prior_mv/eval_sparse_inpaint_stage2.py \
      --manifest "$VAL_PRIOR_DIR/manifest.json" \
      --checkpoint "$CKPT" \
      --output_dir "$EVAL_DIR" \
      --weights "$WEIGHTS" \
      --indices "$EVAL_INDICES" \
      --prior_modes correct,empty,shuffle,random,jitter \
      --fixed_topk 4096,8192,target_unique \
      --steps "$EVAL_STEPS" \
      --guidance_strength 1.0 \
      --known_latent_clamp_strength "$KNOWN_LATENT_CLAMP_STRENGTH" \
      --known_clamp_start_t "$KNOWN_CLAMP_START_T" \
      --known_logit_boost "$KNOWN_LOGIT_BOOST" \
      --known_conf_power "$KNOWN_CONF_POWER" \
      --lora_rank "$LORA_RANK" \
      --lora_alpha "$LORA_ALPHA" \
      $( [[ "$CLAMP_INITIAL_NOISE" == "1" ]] && printf '%s' "--clamp_initial_noise" || printf '%s' "--no_clamp_initial_noise" ) \
      $( [[ "$KNOWN_USE_CONFIDENCE" == "1" ]] && printf '%s' "--known_use_confidence" )
  fi

  if [[ "$RUN_MESH_EVAL" == "1" ]]; then
    GPU="$GPU" \
    MODE="val8" \
    RUN_NAME="$MESH_RUN_NAME" \
    MANIFEST="$VAL_PRIOR_DIR/manifest.json" \
    STAGE2_CHECKPOINT="$CKPT" \
    KNOWN_CLAMP_START_T="$KNOWN_CLAMP_START_T" \
    TOPK_SPECS="$TOPK_SPECS" \
    MESH_EVAL_SAMPLES="$MESH_EVAL_SAMPLES" \
    bash trellis_point_prior_mv/scripts/run_mesh_frozen_topk_sweep.sh
  fi
done

echo "[stage2_weight_sweep] done: $SWEEP_ROOT"
