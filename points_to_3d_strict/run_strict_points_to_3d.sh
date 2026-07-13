#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

GPU="${GPU:-4}"
PY="${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
MODE="${MODE:-overfit1}"
RUN_NAME="${RUN_NAME:-points_to_3d_strict_${MODE}}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_MESH="${RUN_MESH:-0}"

WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
ROOT="${ROOT:-/home/zjr/Tracker/points_to_3d_strict/outputs/${RUN_NAME}}"

case "${MODE}" in
  overfit1)
    LATENT_ROOT="${LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_smoke}"
    TRAIN_INDICES="${TRAIN_INDICES:-0}"
    EVAL_INDICES="${EVAL_INDICES:-0}"
    MAX_STEPS="${MAX_STEPS:-500}"
    SAVE_EVERY="${SAVE_EVERY:-100}"
    ;;
  train64_val32)
    TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_train64}"
    VAL_LATENT_ROOT="${VAL_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val32}"
    TRAIN_INDICES="${TRAIN_INDICES:-all}"
    EVAL_INDICES="${EVAL_INDICES:-0-31}"
    MAX_STEPS="${MAX_STEPS:-1000}"
    SAVE_EVERY="${SAVE_EVERY:-200}"
    ;;
  train512_val128)
    TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_train512}"
    VAL_LATENT_ROOT="${VAL_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val128}"
    TRAIN_INDICES="${TRAIN_INDICES:-all}"
    EVAL_INDICES="${EVAL_INDICES:-0-127}"
    MAX_STEPS="${MAX_STEPS:-2000}"
    SAVE_EVERY="${SAVE_EVERY:-500}"
    ;;
  train1488_val128)
    TRAIN_LATENT_ROOT="${TRAIN_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_train1488}"
    VAL_LATENT_ROOT="${VAL_LATENT_ROOT:-/home/zjr/Tracker/trellis_point_prior_mv/outputs/latent_inpaint/latent_inpaint_val128}"
    TRAIN_INDICES="${TRAIN_INDICES:-all}"
    EVAL_INDICES="${EVAL_INDICES:-0-127}"
    MAX_STEPS="${MAX_STEPS:-5000}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use overfit1, train64_val32, train512_val128, or train1488_val128." >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "overfit1" ]]; then
  TRAIN_MANIFEST="${TRAIN_MANIFEST:-${LATENT_ROOT}/manifest.json}"
  EVAL_MANIFEST="${EVAL_MANIFEST:-${LATENT_ROOT}/manifest.json}"
else
  TRAIN_MANIFEST="${TRAIN_MANIFEST:-${TRAIN_LATENT_ROOT}/manifest.json}"
  EVAL_MANIFEST="${EVAL_MANIFEST:-${VAL_LATENT_ROOT}/manifest.json}"
fi

CKPT="${CKPT:-${ROOT}/checkpoints/last.ckpt}"
if [[ "${CKPT}" == "none" || "${CKPT}" == "stock" || "${CKPT}" == "base" ]]; then
  CKPT=""
fi
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${ROOT}}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${ROOT}/eval}"
MESH_OUTPUT="${MESH_OUTPUT:-${ROOT}/mesh_eval}"

MASK_DILATE64="${MASK_DILATE64:-0}"
MASK_DILATE16="${MASK_DILATE16:-0}"
IMAGE_MAX_VIEWS="${IMAGE_MAX_VIEWS:-1}"
IMAGE_FRAME_SELECT="${IMAGE_FRAME_SELECT:-first}"
IMAGE_SELECT_SEED="${IMAGE_SELECT_SEED:-0}"
IMAGE_COND_AGGREGATION="${IMAGE_COND_AGGREGATION:-first}"
IMAGE_USE_SOURCE_MASK="${IMAGE_USE_SOURCE_MASK:-1}"
IMAGE_MASK_CROP_RESOLUTION="${IMAGE_MASK_CROP_RESOLUTION:-518}"
PROJECTION_GRID_TRANSFORM="${PROJECTION_GRID_TRANSFORM:-pixal3d_rotation}"
CONDITION_ADAPTER_DIM="${CONDITION_ADAPTER_DIM:-1024}"
CONDITION_ADAPTER_HIDDEN_DIM="${CONDITION_ADAPTER_HIDDEN_DIM:-256}"
CONDITION_ADAPTER_MAX_VIEWS="${CONDITION_ADAPTER_MAX_VIEWS:-8}"
CONDITION_ADAPTER_POSE_DIM="${CONDITION_ADAPTER_POSE_DIM:-32}"
LATENT_CHANNELS="${LATENT_CHANNELS:-8}"
CONDITION_PROJECTION_TOKEN_MAX_CELLS="${CONDITION_PROJECTION_TOKEN_MAX_CELLS:-512}"
CONDITION_ADAPTER_USE_VIEW_EMBED="${CONDITION_ADAPTER_USE_VIEW_EMBED:-1}"
STRICT_INPUT_PROJECTION_GRID="${STRICT_INPUT_PROJECTION_GRID:-0}"
STRICT_INPUT_PROJECTION_CHANNELS="${STRICT_INPUT_PROJECTION_CHANNELS:-6}"
LR="${LR:-1e-5}"
ADAM_EPS="${ADAM_EPS:-1e-4}"
FINITE_CHECK_EVERY="${FINITE_CHECK_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-5}"
SEED="${SEED:-42}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0.0}"
TRAIN_T_MIN="${TRAIN_T_MIN:-0.0}"
TRAIN_T_MAX="${TRAIN_T_MAX:-1.0}"
UNKNOWN_FLOW_LOSS_WEIGHT="${UNKNOWN_FLOW_LOSS_WEIGHT:-1.0}"
KNOWN_FLOW_LOSS_WEIGHT="${KNOWN_FLOW_LOSS_WEIGHT:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TRAIN_INPUT_LAYER_ONLY="${TRAIN_INPUT_LAYER_ONLY:-0}"
TRAIN_CONDITION_ADAPTER_ONLY="${TRAIN_CONDITION_ADAPTER_ONLY:-0}"
TRAIN_FLOW_PARTIAL="${TRAIN_FLOW_PARTIAL:-0}"
TRAIN_FLOW_LORA="${TRAIN_FLOW_LORA:-0}"
FLOW_LORA="${FLOW_LORA:-${TRAIN_FLOW_LORA}}"
PARTIAL_TRAIN_BLOCKS="${PARTIAL_TRAIN_BLOCKS:-0}"
PARTIAL_TRAIN_INPUT_LAYER="${PARTIAL_TRAIN_INPUT_LAYER:-1}"
PARTIAL_TRAIN_T_EMBEDDER="${PARTIAL_TRAIN_T_EMBEDDER:-1}"
PARTIAL_TRAIN_OUT_LAYER="${PARTIAL_TRAIN_OUT_LAYER:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_BLOCKS="${LORA_TARGET_BLOCKS:-8}"
LORA_TRAIN_INPUT_LAYER="${LORA_TRAIN_INPUT_LAYER:-0}"
LORA_TRAIN_T_EMBEDDER="${LORA_TRAIN_T_EMBEDDER:-0}"
LORA_TRAIN_OUT_LAYER="${LORA_TRAIN_OUT_LAYER:-0}"
TRAIN_BLOCKWISE_GLOBAL_MODULATION="${TRAIN_BLOCKWISE_GLOBAL_MODULATION:-0}"
BLOCKWISE_GLOBAL_MODULATION="${BLOCKWISE_GLOBAL_MODULATION:-${TRAIN_BLOCKWISE_GLOBAL_MODULATION}}"
BLOCKWISE_GLOBAL_TARGET_BLOCKS="${BLOCKWISE_GLOBAL_TARGET_BLOCKS:-4}"
BLOCKWISE_GLOBAL_HIDDEN_DIM="${BLOCKWISE_GLOBAL_HIDDEN_DIM:-256}"
BLOCKWISE_GLOBAL_MODE="${BLOCKWISE_GLOBAL_MODE:-bias}"
TRAIN_BLOCKWISE_TOKEN_MODULATION="${TRAIN_BLOCKWISE_TOKEN_MODULATION:-0}"
BLOCKWISE_TOKEN_MODULATION="${BLOCKWISE_TOKEN_MODULATION:-${TRAIN_BLOCKWISE_TOKEN_MODULATION}}"
BLOCKWISE_TOKEN_TARGET_BLOCKS="${BLOCKWISE_TOKEN_TARGET_BLOCKS:-4}"
BLOCKWISE_TOKEN_HIDDEN_DIM="${BLOCKWISE_TOKEN_HIDDEN_DIM:-128}"
BLOCKWISE_TOKEN_MODE="${BLOCKWISE_TOKEN_MODE:-film}"
FREEZE_CONDITION_ADAPTER="${FREEZE_CONDITION_ADAPTER:-0}"
NO_AMP="${NO_AMP:-0}"
NO_GRAD_SCALER="${NO_GRAD_SCALER:-0}"
RESUME="${RESUME:-}"
PARTIAL_FLOW_LABEL="disabled"
if [[ "${TRAIN_FLOW_PARTIAL}" == "1" ]]; then
  if [[ "${PARTIAL_TRAIN_BLOCKS}" == "0" ]]; then
    PARTIAL_FLOW_LABEL="b0_peripheral_input_t_out"
  else
    PARTIAL_FLOW_LABEL="peripheral_plus_last_${PARTIAL_TRAIN_BLOCKS}_blocks"
  fi
fi

EVAL_STEPS="${EVAL_STEPS:-25}"
BOUNDARY_REFINE_STEPS="${BOUNDARY_REFINE_STEPS:-0}"
STRICT_INIT_MODE="${STRICT_INIT_MODE:-noise}"
START_T="${START_T:-1.0}"
Q_SPLICE_NOISE_SCALE="${Q_SPLICE_NOISE_SCALE:-0.05}"
INCLUDE_STOCK_SOURCE="${INCLUDE_STOCK_SOURCE:-0}"
STOCK_STEPS="${STOCK_STEPS:-0}"
STOCK_CFG_STRENGTH="${STOCK_CFG_STRENGTH:-0}"
STOCK_RESCALE_T="${STOCK_RESCALE_T:-0}"
RESCALE_T="${RESCALE_T:-1.0}"
CFG_STRENGTH="${CFG_STRENGTH:-1.0}"
TEACHER_FORCED_T="${TEACHER_FORCED_T:-0.25,0.5,0.75}"
TOPK="${TOPK:-4096,8192,target_unique}"
EVAL_PROGRESS_EVERY="${EVAL_PROGRESS_EVERY:-1}"
MESH_MODES="${MESH_MODES:-q_pred,q_splice}"
COORD_DECODE="${COORD_DECODE:-threshold}"
SLAT_STEPS="${SLAT_STEPS:-12}"
SLAT_GUIDANCE_STRENGTH="${SLAT_GUIDANCE_STRENGTH:-7.5}"
SLAT_GUIDANCE_RESCALE="${SLAT_GUIDANCE_RESCALE:-0.5}"
SLAT_RESCALE_T="${SLAT_RESCALE_T:-3.0}"
MESH_EVAL_SAMPLES="${MESH_EVAL_SAMPLES:-4000}"

echo "[points_to_3d_strict] mode=${MODE} run=${RUN_NAME}"
echo "[points_to_3d_strict] train_manifest=${TRAIN_MANIFEST}"
echo "[points_to_3d_strict] eval_manifest=${EVAL_MANIFEST}"
echo "[points_to_3d_strict] root=${ROOT}"
echo "[points_to_3d_strict] image_views=${IMAGE_MAX_VIEWS} aggregation=${IMAGE_COND_AGGREGATION} use_mask=${IMAGE_USE_SOURCE_MASK} projection_grid_transform=${PROJECTION_GRID_TRANSFORM}"
echo "[points_to_3d_strict] condition_adapter hidden=${CONDITION_ADAPTER_HIDDEN_DIM} max_views=${CONDITION_ADAPTER_MAX_VIEWS} pose_dim=${CONDITION_ADAPTER_POSE_DIM} projection_token_max=${CONDITION_PROJECTION_TOKEN_MAX_CELLS} train_adapter_only=${TRAIN_CONDITION_ADAPTER_ONLY}"
echo "[points_to_3d_strict] strict_input_projection_grid=${STRICT_INPUT_PROJECTION_GRID} strict_input_projection_channels=${STRICT_INPUT_PROJECTION_CHANNELS}"
echo "[points_to_3d_strict] flow_train input_only=${TRAIN_INPUT_LAYER_ONLY} partial=${TRAIN_FLOW_PARTIAL} partial_label=${PARTIAL_FLOW_LABEL} partial_blocks=${PARTIAL_TRAIN_BLOCKS} partial_input=${PARTIAL_TRAIN_INPUT_LAYER} partial_t=${PARTIAL_TRAIN_T_EMBEDDER} partial_out=${PARTIAL_TRAIN_OUT_LAYER} freeze_adapter=${FREEZE_CONDITION_ADAPTER}"
echo "[points_to_3d_strict] flow_lora train=${TRAIN_FLOW_LORA} eval=${FLOW_LORA} rank=${LORA_RANK} alpha=${LORA_ALPHA} target_blocks=${LORA_TARGET_BLOCKS} lora_input=${LORA_TRAIN_INPUT_LAYER} lora_t=${LORA_TRAIN_T_EMBEDDER} lora_out=${LORA_TRAIN_OUT_LAYER}"
echo "[points_to_3d_strict] blockwise_global train=${TRAIN_BLOCKWISE_GLOBAL_MODULATION} eval=${BLOCKWISE_GLOBAL_MODULATION} target_blocks=${BLOCKWISE_GLOBAL_TARGET_BLOCKS} hidden=${BLOCKWISE_GLOBAL_HIDDEN_DIM} mode=${BLOCKWISE_GLOBAL_MODE}"
echo "[points_to_3d_strict] blockwise_token train=${TRAIN_BLOCKWISE_TOKEN_MODULATION} eval=${BLOCKWISE_TOKEN_MODULATION} target_blocks=${BLOCKWISE_TOKEN_TARGET_BLOCKS} hidden=${BLOCKWISE_TOKEN_HIDDEN_DIM} mode=${BLOCKWISE_TOKEN_MODE}"
echo "[points_to_3d_strict] cfg_strength=${CFG_STRENGTH} boundary_refine_steps=${BOUNDARY_REFINE_STEPS}"
echo "[points_to_3d_strict] strict_init_mode=${STRICT_INIT_MODE} start_t=${START_T} stock_steps=${STOCK_STEPS}"
echo "[points_to_3d_strict] loss_weights unknown=${UNKNOWN_FLOW_LOSS_WEIGHT} known=${KNOWN_FLOW_LOSS_WEIGHT} cfg_drop_prob=${CFG_DROP_PROB} t_range=[${TRAIN_T_MIN},${TRAIN_T_MAX}]"
echo "[points_to_3d_strict] optimizer lr=${LR} adam_eps=${ADAM_EPS} finite_check_every=${FINITE_CHECK_EVERY}"
echo "[points_to_3d_strict] progress log_every=${LOG_EVERY} eval_progress_every=${EVAL_PROGRESS_EVERY}"
echo "[points_to_3d_strict] run_train=${RUN_TRAIN} run_eval=${RUN_EVAL} run_mesh=${RUN_MESH}"

COMMON_ENV=(
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  ATTN_BACKEND=flash_attn
  SPCONV_ALGO=native
  MPLCONFIGDIR=/tmp/matplotlib
  NUMBA_CACHE_DIR=/tmp/numba_cache
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
if [[ -n "${GPU}" && "${GPU}" != "all" && "${GPU}" != "none" ]]; then
  COMMON_ENV=(CUDA_VISIBLE_DEVICES="${GPU}" "${COMMON_ENV[@]}")
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  TRAIN_ARGS=(
    -u points_to_3d_strict/train_strict_inpaint.py
    --manifest "${TRAIN_MANIFEST}"
    --output_dir "${TRAIN_OUTPUT}"
    --weights "${WEIGHTS}"
    --indices "${TRAIN_INDICES}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_steps "${MAX_STEPS}"
    --save_every "${SAVE_EVERY}"
    --log_every "${LOG_EVERY}"
    --lr "${LR}"
    --adam_eps "${ADAM_EPS}"
    --finite_check_every "${FINITE_CHECK_EVERY}"
    --seed "${SEED}"
    --cfg_drop_prob "${CFG_DROP_PROB}"
    --t_min "${TRAIN_T_MIN}"
    --t_max "${TRAIN_T_MAX}"
    --unknown_flow_loss_weight "${UNKNOWN_FLOW_LOSS_WEIGHT}"
    --known_flow_loss_weight "${KNOWN_FLOW_LOSS_WEIGHT}"
    --mask_dilate64 "${MASK_DILATE64}"
    --mask_dilate16 "${MASK_DILATE16}"
    --image_max_views "${IMAGE_MAX_VIEWS}"
    --image_frame_select "${IMAGE_FRAME_SELECT}"
    --image_select_seed "${IMAGE_SELECT_SEED}"
    --image_cond_aggregation "${IMAGE_COND_AGGREGATION}"
    --image_mask_crop_resolution "${IMAGE_MASK_CROP_RESOLUTION}"
    --projection_grid_transform "${PROJECTION_GRID_TRANSFORM}"
    --condition_adapter_dim "${CONDITION_ADAPTER_DIM}"
    --condition_adapter_hidden_dim "${CONDITION_ADAPTER_HIDDEN_DIM}"
    --condition_adapter_max_views "${CONDITION_ADAPTER_MAX_VIEWS}"
    --condition_adapter_pose_dim "${CONDITION_ADAPTER_POSE_DIM}"
    --latent_channels "${LATENT_CHANNELS}"
    --condition_projection_token_max_cells "${CONDITION_PROJECTION_TOKEN_MAX_CELLS}"
    --strict_input_projection_channels "${STRICT_INPUT_PROJECTION_CHANNELS}"
    --grad_clip "${GRAD_CLIP}"
    --partial_train_blocks "${PARTIAL_TRAIN_BLOCKS}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_blocks "${LORA_TARGET_BLOCKS}"
    --blockwise_global_target_blocks "${BLOCKWISE_GLOBAL_TARGET_BLOCKS}"
    --blockwise_global_hidden_dim "${BLOCKWISE_GLOBAL_HIDDEN_DIM}"
    --blockwise_global_mode "${BLOCKWISE_GLOBAL_MODE}"
    --blockwise_token_target_blocks "${BLOCKWISE_TOKEN_TARGET_BLOCKS}"
    --blockwise_token_hidden_dim "${BLOCKWISE_TOKEN_HIDDEN_DIM}"
    --blockwise_token_mode "${BLOCKWISE_TOKEN_MODE}"
  )
  if [[ -n "${RESUME}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME}")
  fi
  if [[ "${IMAGE_USE_SOURCE_MASK}" == "1" ]]; then
    TRAIN_ARGS+=(--image_use_source_mask)
  else
    TRAIN_ARGS+=(--no-image_use_source_mask)
  fi
  if [[ "${TRAIN_INPUT_LAYER_ONLY}" == "1" ]]; then
    TRAIN_ARGS+=(--train_input_layer_only)
  fi
  if [[ "${STRICT_INPUT_PROJECTION_GRID}" == "1" ]]; then
    TRAIN_ARGS+=(--strict_input_projection_grid)
  fi
  if [[ "${TRAIN_CONDITION_ADAPTER_ONLY}" == "1" ]]; then
    TRAIN_ARGS+=(--train_condition_adapter_only)
  fi
  if [[ "${TRAIN_FLOW_PARTIAL}" == "1" ]]; then
    TRAIN_ARGS+=(--train_flow_partial)
  fi
  if [[ "${TRAIN_FLOW_LORA}" == "1" ]]; then
    TRAIN_ARGS+=(--train_flow_lora)
  fi
  if [[ "${TRAIN_BLOCKWISE_GLOBAL_MODULATION}" == "1" ]]; then
    TRAIN_ARGS+=(--train_blockwise_global_modulation)
  fi
  if [[ "${TRAIN_BLOCKWISE_TOKEN_MODULATION}" == "1" ]]; then
    TRAIN_ARGS+=(--train_blockwise_token_modulation)
  fi
  if [[ "${PARTIAL_TRAIN_INPUT_LAYER}" == "1" ]]; then
    TRAIN_ARGS+=(--partial_train_input_layer)
  else
    TRAIN_ARGS+=(--no-partial_train_input_layer)
  fi
  if [[ "${PARTIAL_TRAIN_T_EMBEDDER}" == "1" ]]; then
    TRAIN_ARGS+=(--partial_train_t_embedder)
  else
    TRAIN_ARGS+=(--no-partial_train_t_embedder)
  fi
  if [[ "${PARTIAL_TRAIN_OUT_LAYER}" == "1" ]]; then
    TRAIN_ARGS+=(--partial_train_out_layer)
  else
    TRAIN_ARGS+=(--no-partial_train_out_layer)
  fi
  if [[ "${LORA_TRAIN_INPUT_LAYER}" == "1" ]]; then
    TRAIN_ARGS+=(--lora_train_input_layer)
  else
    TRAIN_ARGS+=(--no-lora_train_input_layer)
  fi
  if [[ "${LORA_TRAIN_T_EMBEDDER}" == "1" ]]; then
    TRAIN_ARGS+=(--lora_train_t_embedder)
  else
    TRAIN_ARGS+=(--no-lora_train_t_embedder)
  fi
  if [[ "${LORA_TRAIN_OUT_LAYER}" == "1" ]]; then
    TRAIN_ARGS+=(--lora_train_out_layer)
  else
    TRAIN_ARGS+=(--no-lora_train_out_layer)
  fi
  if [[ "${FREEZE_CONDITION_ADAPTER}" == "1" ]]; then
    TRAIN_ARGS+=(--freeze_condition_adapter)
  fi
  if [[ "${CONDITION_ADAPTER_USE_VIEW_EMBED}" == "1" ]]; then
    TRAIN_ARGS+=(--condition_adapter_use_view_embed)
  else
    TRAIN_ARGS+=(--no-condition_adapter_use_view_embed)
  fi
  if [[ "${NO_AMP}" == "1" ]]; then
    TRAIN_ARGS+=(--no_amp)
  fi
  if [[ "${NO_GRAD_SCALER}" == "1" ]]; then
    TRAIN_ARGS+=(--no_grad_scaler)
  fi
  env "${COMMON_ENV[@]}" "${PY}" "${TRAIN_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  EVAL_ARGS=(
    -u points_to_3d_strict/eval_strict_inpaint.py
    --manifest "${EVAL_MANIFEST}"
    --checkpoint "${CKPT}"
    --output_dir "${EVAL_OUTPUT}"
    --weights "${WEIGHTS}"
    --indices "${EVAL_INDICES}"
    --mask_dilate64 "${MASK_DILATE64}"
    --mask_dilate16 "${MASK_DILATE16}"
    --image_max_views "${IMAGE_MAX_VIEWS}"
    --image_frame_select "${IMAGE_FRAME_SELECT}"
    --image_select_seed "${IMAGE_SELECT_SEED}"
    --image_cond_aggregation "${IMAGE_COND_AGGREGATION}"
    --image_mask_crop_resolution "${IMAGE_MASK_CROP_RESOLUTION}"
    --projection_grid_transform "${PROJECTION_GRID_TRANSFORM}"
    --condition_adapter_dim "${CONDITION_ADAPTER_DIM}"
    --condition_adapter_hidden_dim "${CONDITION_ADAPTER_HIDDEN_DIM}"
    --condition_adapter_max_views "${CONDITION_ADAPTER_MAX_VIEWS}"
    --condition_adapter_pose_dim "${CONDITION_ADAPTER_POSE_DIM}"
    --latent_channels "${LATENT_CHANNELS}"
    --condition_projection_token_max_cells "${CONDITION_PROJECTION_TOKEN_MAX_CELLS}"
    --strict_input_projection_channels "${STRICT_INPUT_PROJECTION_CHANNELS}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_blocks "${LORA_TARGET_BLOCKS}"
    --blockwise_global_target_blocks "${BLOCKWISE_GLOBAL_TARGET_BLOCKS}"
    --blockwise_global_hidden_dim "${BLOCKWISE_GLOBAL_HIDDEN_DIM}"
    --blockwise_global_mode "${BLOCKWISE_GLOBAL_MODE}"
    --blockwise_token_target_blocks "${BLOCKWISE_TOKEN_TARGET_BLOCKS}"
    --blockwise_token_hidden_dim "${BLOCKWISE_TOKEN_HIDDEN_DIM}"
    --blockwise_token_mode "${BLOCKWISE_TOKEN_MODE}"
    --steps "${EVAL_STEPS}"
    --boundary_refine_steps "${BOUNDARY_REFINE_STEPS}"
    --init_mode "${STRICT_INIT_MODE}"
    --start_t "${START_T}"
    --q_splice_noise_scale "${Q_SPLICE_NOISE_SCALE}"
    --stock_steps "${STOCK_STEPS}"
    --stock_cfg_strength "${STOCK_CFG_STRENGTH}"
    --stock_rescale_t "${STOCK_RESCALE_T}"
    --rescale_t "${RESCALE_T}"
    --cfg_strength "${CFG_STRENGTH}"
    --teacher_forced_t "${TEACHER_FORCED_T}"
    --topk "${TOPK}"
    --progress_every "${EVAL_PROGRESS_EVERY}"
    --seed "${SEED}"
  )
  if [[ "${IMAGE_USE_SOURCE_MASK}" == "1" ]]; then
    EVAL_ARGS+=(--image_use_source_mask)
  else
    EVAL_ARGS+=(--no-image_use_source_mask)
  fi
  if [[ "${STRICT_INPUT_PROJECTION_GRID}" == "1" ]]; then
    EVAL_ARGS+=(--strict_input_projection_grid)
  fi
  if [[ "${FLOW_LORA}" == "1" ]]; then
    EVAL_ARGS+=(--flow_lora)
  fi
  if [[ "${BLOCKWISE_GLOBAL_MODULATION}" == "1" ]]; then
    EVAL_ARGS+=(--blockwise_global_modulation)
  fi
  if [[ "${BLOCKWISE_TOKEN_MODULATION}" == "1" ]]; then
    EVAL_ARGS+=(--blockwise_token_modulation)
  fi
  if [[ "${INCLUDE_STOCK_SOURCE}" == "1" ]]; then
    EVAL_ARGS+=(--include_stock_source)
  fi
  if [[ "${CONDITION_ADAPTER_USE_VIEW_EMBED}" == "1" ]]; then
    EVAL_ARGS+=(--condition_adapter_use_view_embed)
  else
    EVAL_ARGS+=(--no-condition_adapter_use_view_embed)
  fi
  env "${COMMON_ENV[@]}" "${PY}" "${EVAL_ARGS[@]}"
  echo "[points_to_3d_strict] report=${EVAL_OUTPUT}/report.json"
fi

if [[ "${RUN_MESH}" == "1" ]]; then
  MESH_ARGS=(
    -u points_to_3d_strict/eval_strict_mesh.py
    --manifest "${EVAL_MANIFEST}"
    --checkpoint "${CKPT}"
    --output_dir "${MESH_OUTPUT}"
    --weights "${WEIGHTS}"
    --indices "${EVAL_INDICES}"
    --modes "${MESH_MODES}"
    --mask_dilate64 "${MASK_DILATE64}"
    --mask_dilate16 "${MASK_DILATE16}"
    --image_max_views "${IMAGE_MAX_VIEWS}"
    --image_frame_select "${IMAGE_FRAME_SELECT}"
    --image_select_seed "${IMAGE_SELECT_SEED}"
    --image_cond_aggregation "${IMAGE_COND_AGGREGATION}"
    --image_mask_crop_resolution "${IMAGE_MASK_CROP_RESOLUTION}"
    --projection_grid_transform "${PROJECTION_GRID_TRANSFORM}"
    --condition_adapter_dim "${CONDITION_ADAPTER_DIM}"
    --condition_adapter_hidden_dim "${CONDITION_ADAPTER_HIDDEN_DIM}"
    --condition_adapter_max_views "${CONDITION_ADAPTER_MAX_VIEWS}"
    --condition_adapter_pose_dim "${CONDITION_ADAPTER_POSE_DIM}"
    --latent_channels "${LATENT_CHANNELS}"
    --condition_projection_token_max_cells "${CONDITION_PROJECTION_TOKEN_MAX_CELLS}"
    --strict_input_projection_channels "${STRICT_INPUT_PROJECTION_CHANNELS}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_blocks "${LORA_TARGET_BLOCKS}"
    --blockwise_global_target_blocks "${BLOCKWISE_GLOBAL_TARGET_BLOCKS}"
    --blockwise_global_hidden_dim "${BLOCKWISE_GLOBAL_HIDDEN_DIM}"
    --blockwise_global_mode "${BLOCKWISE_GLOBAL_MODE}"
    --blockwise_token_target_blocks "${BLOCKWISE_TOKEN_TARGET_BLOCKS}"
    --blockwise_token_hidden_dim "${BLOCKWISE_TOKEN_HIDDEN_DIM}"
    --blockwise_token_mode "${BLOCKWISE_TOKEN_MODE}"
    --steps "${EVAL_STEPS}"
    --boundary_refine_steps "${BOUNDARY_REFINE_STEPS}"
    --init_mode "${STRICT_INIT_MODE}"
    --start_t "${START_T}"
    --q_splice_noise_scale "${Q_SPLICE_NOISE_SCALE}"
    --stock_steps "${STOCK_STEPS}"
    --stock_cfg_strength "${STOCK_CFG_STRENGTH}"
    --stock_rescale_t "${STOCK_RESCALE_T}"
    --rescale_t "${RESCALE_T}"
    --cfg_strength "${CFG_STRENGTH}"
    --coord_decode "${COORD_DECODE}"
    --slat_steps "${SLAT_STEPS}"
    --slat_guidance_strength "${SLAT_GUIDANCE_STRENGTH}"
    --slat_guidance_rescale "${SLAT_GUIDANCE_RESCALE}"
    --slat_rescale_t "${SLAT_RESCALE_T}"
    --mesh_eval_samples "${MESH_EVAL_SAMPLES}"
    --seed "${SEED}"
  )
  if [[ "${IMAGE_USE_SOURCE_MASK}" == "1" ]]; then
    MESH_ARGS+=(--image_use_source_mask)
  else
    MESH_ARGS+=(--no-image_use_source_mask)
  fi
  if [[ "${STRICT_INPUT_PROJECTION_GRID}" == "1" ]]; then
    MESH_ARGS+=(--strict_input_projection_grid)
  fi
  if [[ "${FLOW_LORA}" == "1" ]]; then
    MESH_ARGS+=(--flow_lora)
  fi
  if [[ "${BLOCKWISE_GLOBAL_MODULATION}" == "1" ]]; then
    MESH_ARGS+=(--blockwise_global_modulation)
  fi
  if [[ "${BLOCKWISE_TOKEN_MODULATION}" == "1" ]]; then
    MESH_ARGS+=(--blockwise_token_modulation)
  fi
  if [[ "${CONDITION_ADAPTER_USE_VIEW_EMBED}" == "1" ]]; then
    MESH_ARGS+=(--condition_adapter_use_view_embed)
  else
    MESH_ARGS+=(--no-condition_adapter_use_view_embed)
  fi
  env "${COMMON_ENV[@]}" "${PY}" "${MESH_ARGS[@]}"
  echo "[points_to_3d_strict] mesh_report=${MESH_OUTPUT}/report.json"
fi
