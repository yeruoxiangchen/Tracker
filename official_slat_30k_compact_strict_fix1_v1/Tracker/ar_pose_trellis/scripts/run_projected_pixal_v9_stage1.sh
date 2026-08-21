#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

MODE="${MODE:-all}"
GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/pixal3d_multiview/objaverse_sparse_mv_artraj_pbr_5000_v9_select8}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"

SMOKE_RUN_DIR="${SMOKE_RUN_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_smoke_v2}"
SMOKE_EVAL_DIR="${SMOKE_EVAL_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_smoke_v2_val8}"
S200_RUN_DIR="${S200_RUN_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_s200_v2}"
S200_EVAL_DIR="${S200_EVAL_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32}"
S200_FIXED_EVAL_DIR="${S200_FIXED_EVAL_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32_fixed_topk}"
S200_VH_FIXED_ROOT="${S200_VH_FIXED_ROOT:-/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_v2_val32_fixed_topk_vh_sweep}"
RANK_SMOKE_RUN_DIR="${RANK_SMOKE_RUN_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_s200_rank_smoke_v1}"
RANK_SMOKE_EVAL_DIR="${RANK_SMOKE_EVAL_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_rank_smoke_v1_val8_fixed_topk}"
RANK_S200_RUN_DIR="${RANK_S200_RUN_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/ss_projected_pixal_v9_rot_s200_rank_s200_v1}"
RANK_S200_EVAL_DIR="${RANK_S200_EVAL_DIR:-/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/ss_projected_pixal_v9_rot_s200_rank_s200_v1_val32_fixed_topk}"
DIAG_JSON="${DIAG_JSON:-/home/zjr/Tracker/ar_pose_trellis/outputs/diagnostics/projected_alignment_pixal_v9_train_0_31_rot_strict_v2.json}"
FIXED_TOPK="${FIXED_TOPK:-4096,8192,target_unique}"
VH_WEIGHTS="${VH_WEIGHTS:-5 10 20 40}"
VH_MASK_THRESHOLD="${VH_MASK_THRESHOLD:-0.5}"
VH_MIN_VISIBLE_VIEWS="${VH_MIN_VISIBLE_VIEWS:-1}"
RANK_RESUME_CKPT="${RANK_RESUME_CKPT:-}"
RANKING_WEIGHT="${RANKING_WEIGHT:-0.2}"
RANKING_MARGIN="${RANKING_MARGIN:-0.05}"
RANKING_MODES="${RANKING_MODES:-identity,shuffle,large_noise,noise}"
RANKING_NUM_NEGATIVES="${RANKING_NUM_NEGATIVES:-1}"
RANKING_BACKGROUND_SAMPLES="${RANKING_BACKGROUND_SAMPLES:-4096}"
RANKING_EVERY_N_STEPS="${RANKING_EVERY_N_STEPS:-1}"

COMMON_ENV=(
  CUDA_VISIBLE_DEVICES="$GPU"
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
  SPCONV_ALGO="${SPCONV_ALGO:-native}"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
  TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
)

latest_ckpt() {
  local run_dir="$1"
  local ckpt
  ckpt=$(find "$run_dir" -maxdepth 1 -name "*.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [[ -z "${ckpt:-}" || ! -f "$ckpt" ]]; then
    echo "No checkpoint found in $run_dir" >&2
    exit 1
  fi
  printf "%s" "$ckpt"
}

run_diagnostic() {
  mkdir -p "$(dirname "$DIAG_JSON")"
  echo "[stage1] diagnostic -> $DIAG_JSON"
  "$PYTHON_BIN" -u ar_pose_trellis/diagnose_projected_sparse_alignment.py \
    --data_root "$DATA_ROOT" \
    --split train \
    --indices 0-31 \
    --num_views 8 \
    --max_points 5000 \
    --extrinsics_type c2w \
    --camera_forward_sign 1.0 \
    --grid_transform pixal3d_rotation \
    --min_support 2.0 \
    --min_support_ratio 0.5 \
    --output_json "$DIAG_JSON"
  test -f "$DIAG_JSON"
}

run_train() {
  local run_dir="$1"
  local max_steps="$2"
  local workers="$3"
  local save_every="$4"
  mkdir -p "$run_dir"
  echo "[stage1] train steps=$max_steps -> $run_dir"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" -u ar_pose_trellis/train_ss_ar_pose.py \
    --dataset_format pixal3d_multiview_manifest \
    --data_root "$DATA_ROOT" \
    --split train \
    --weights "$WEIGHTS" \
    --save_dir "$run_dir" \
    --condition_mode projected \
    --projected_grid_resolution 16 \
    --projected_grid_transform pixal3d_rotation \
    --projected_min_support 2.0 \
    --projected_min_support_ratio 0.5 \
    --num_views 8 \
    --batch_size 1 \
    --num_workers "$workers" \
    --max_epochs 100 \
    --max_steps "$max_steps" \
    --lr 5e-5 \
    --cfg_drop_prob 0.1 \
    --ckpt_every_n_steps "$save_every" \
    --extrinsics_type c2w \
    --camera_forward_sign 1.0 \
    --absolute_pose_condition
  latest_ckpt "$run_dir" >/dev/null
  echo "[stage1] checkpoint $(latest_ckpt "$run_dir")"
}

run_eval() {
  local run_dir="$1"
  local eval_dir="$2"
  local indices="$3"
  local steps="$4"
  local modes="$5"
  local ckpt
  ckpt=$(latest_ckpt "$run_dir")
  rm -f "$eval_dir/failure.json"
  mkdir -p "$eval_dir"
  echo "[stage1] eval ckpt=$ckpt -> $eval_dir"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" -u ar_pose_trellis/evaluate_projected_sparse_pixal_manifest.py \
    --manifest "$DATA_ROOT/val.json" \
    --checkpoint "$ckpt" \
    --output_dir "$eval_dir" \
    --weights "$WEIGHTS" \
    --condition_mode projected \
    --projected_grid_resolution 16 \
    --projected_grid_transform pixal3d_rotation \
    --projected_min_support 2.0 \
    --projected_min_support_ratio 0.5 \
    --indices "$indices" \
    --pose_modes "$modes" \
    --max_frames 8 \
    --ss_steps "$steps" \
    --ss_guidance_strength 1.0 \
    --ss_min_coords 0 \
    --camera_forward_sign 1.0 \
    --cond_fp16
  test -f "$eval_dir/report.json"
  echo "[stage1] report $eval_dir/report.json"
}

run_fixed_eval() {
  local run_dir="$1"
  local eval_dir="$2"
  local indices="$3"
  local steps="$4"
  local modes="$5"
  local visual_hull_weight="${6:-0}"
  local ckpt
  ckpt=$(latest_ckpt "$run_dir")
  rm -f "$eval_dir/failure.json"
  mkdir -p "$eval_dir"
  echo "[stage1] fixed-topk eval ckpt=$ckpt topk=$FIXED_TOPK vh_weight=$visual_hull_weight -> $eval_dir"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" -u ar_pose_trellis/evaluate_projected_sparse_pixal_manifest.py \
    --manifest "$DATA_ROOT/val.json" \
    --checkpoint "$ckpt" \
    --output_dir "$eval_dir" \
    --weights "$WEIGHTS" \
    --condition_mode projected \
    --projected_grid_resolution 16 \
    --projected_grid_transform pixal3d_rotation \
    --projected_min_support 2.0 \
    --projected_min_support_ratio 0.5 \
    --indices "$indices" \
    --pose_modes "$modes" \
    --max_frames 8 \
    --ss_steps "$steps" \
    --ss_guidance_strength 1.0 \
    --ss_min_coords 0 \
    --fixed_topk "$FIXED_TOPK" \
    --camera_forward_sign 1.0 \
    --visual_hull_prior_weight "$visual_hull_weight" \
    --visual_hull_mask_threshold "$VH_MASK_THRESHOLD" \
    --visual_hull_min_visible_views "$VH_MIN_VISIBLE_VIEWS" \
    --cond_fp16
  test -f "$eval_dir/report.json"
  echo "[stage1] report $eval_dir/report.json"
}

run_vh_fixed_sweep() {
  local run_dir="$1"
  local root_dir="$2"
  local indices="$3"
  local steps="$4"
  local modes="$5"
  local weight
  mkdir -p "$root_dir"
  for weight in $VH_WEIGHTS; do
    run_fixed_eval "$run_dir" "$root_dir/vh_w${weight}" "$indices" "$steps" "$modes" "$weight"
  done
}

rank_resume_ckpt() {
  if [[ -n "${RANK_RESUME_CKPT:-}" ]]; then
    printf "%s" "$RANK_RESUME_CKPT"
  else
    latest_ckpt "$S200_RUN_DIR"
  fi
}

run_rank_train() {
  local run_dir="$1"
  local max_steps="$2"
  local workers="$3"
  local save_every="$4"
  local resume_ckpt
  resume_ckpt=$(rank_resume_ckpt)
  mkdir -p "$run_dir"
  echo "[stage1] rank train steps=$max_steps resume=$resume_ckpt -> $run_dir"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" -u ar_pose_trellis/train_ss_ar_pose.py \
    --dataset_format pixal3d_multiview_manifest \
    --data_root "$DATA_ROOT" \
    --split train \
    --weights "$WEIGHTS" \
    --save_dir "$run_dir" \
    --resume "$resume_ckpt" \
    --condition_mode projected \
    --projected_grid_resolution 16 \
    --projected_grid_transform pixal3d_rotation \
    --projected_min_support 2.0 \
    --projected_min_support_ratio 0.5 \
    --num_views 8 \
    --batch_size 1 \
    --num_workers "$workers" \
    --max_epochs 100 \
    --max_steps "$max_steps" \
    --lr 2e-5 \
    --cfg_drop_prob 0.0 \
    --ckpt_every_n_steps "$save_every" \
    --extrinsics_type c2w \
    --camera_forward_sign 1.0 \
    --absolute_pose_condition \
    --ranking_weight "$RANKING_WEIGHT" \
    --ranking_margin "$RANKING_MARGIN" \
    --ranking_modes "$RANKING_MODES" \
    --ranking_num_negatives "$RANKING_NUM_NEGATIVES" \
    --ranking_background_samples "$RANKING_BACKGROUND_SAMPLES" \
    --ranking_every_n_steps "$RANKING_EVERY_N_STEPS"
  latest_ckpt "$run_dir" >/dev/null
  echo "[stage1] rank checkpoint $(latest_ckpt "$run_dir")"
}

case "$MODE" in
  diagnostic)
    run_diagnostic
    ;;
  smoke_train)
    run_train "$SMOKE_RUN_DIR" 20 0 20
    ;;
  smoke_eval)
    run_eval "$SMOKE_RUN_DIR" "$SMOKE_EVAL_DIR" 0-7 8 correct,identity,shuffle,noise
    ;;
  smoke)
    run_diagnostic
    run_train "$SMOKE_RUN_DIR" 20 0 20
    run_eval "$SMOKE_RUN_DIR" "$SMOKE_EVAL_DIR" 0-7 8 correct,identity,shuffle,noise
    ;;
  s200_train)
    run_train "$S200_RUN_DIR" 200 2 100
    ;;
  s200_eval)
    run_eval "$S200_RUN_DIR" "$S200_EVAL_DIR" 0-31 12 correct,identity,shuffle,noise,large_noise
    ;;
  s200_fixed_eval)
    run_fixed_eval "$S200_RUN_DIR" "$S200_FIXED_EVAL_DIR" 0-31 12 correct,identity,shuffle,noise,large_noise 0
    ;;
  s200_vh_fixed_sweep)
    run_vh_fixed_sweep "$S200_RUN_DIR" "$S200_VH_FIXED_ROOT" 0-31 12 correct,identity,shuffle,noise,large_noise
    ;;
  rank_smoke_train)
    run_rank_train "$RANK_SMOKE_RUN_DIR" 20 0 20
    ;;
  rank_smoke_eval)
    run_fixed_eval "$RANK_SMOKE_RUN_DIR" "$RANK_SMOKE_EVAL_DIR" 0-7 8 correct,identity,shuffle,noise,large_noise 0
    ;;
  rank_smoke)
    run_rank_train "$RANK_SMOKE_RUN_DIR" 20 0 20
    run_fixed_eval "$RANK_SMOKE_RUN_DIR" "$RANK_SMOKE_EVAL_DIR" 0-7 8 correct,identity,shuffle,noise,large_noise 0
    ;;
  rank_s200_train)
    run_rank_train "$RANK_S200_RUN_DIR" 200 2 100
    ;;
  rank_s200_eval)
    run_fixed_eval "$RANK_S200_RUN_DIR" "$RANK_S200_EVAL_DIR" 0-31 12 correct,identity,shuffle,noise,large_noise 0
    ;;
  rank_s200)
    run_rank_train "$RANK_S200_RUN_DIR" 200 2 100
    run_fixed_eval "$RANK_S200_RUN_DIR" "$RANK_S200_EVAL_DIR" 0-31 12 correct,identity,shuffle,noise,large_noise 0
    ;;
  s200)
    run_train "$S200_RUN_DIR" 200 2 100
    run_eval "$S200_RUN_DIR" "$S200_EVAL_DIR" 0-31 12 correct,identity,shuffle,noise,large_noise
    ;;
  all)
    run_diagnostic
    run_train "$SMOKE_RUN_DIR" 20 0 20
    run_eval "$SMOKE_RUN_DIR" "$SMOKE_EVAL_DIR" 0-7 8 correct,identity,shuffle,noise
    run_train "$S200_RUN_DIR" 200 2 100
    run_eval "$S200_RUN_DIR" "$S200_EVAL_DIR" 0-31 12 correct,identity,shuffle,noise,large_noise
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    echo "Valid: diagnostic smoke_train smoke_eval smoke s200_train s200_eval s200_fixed_eval s200_vh_fixed_sweep rank_smoke_train rank_smoke_eval rank_smoke rank_s200_train rank_s200_eval rank_s200 s200 all" >&2
    exit 2
    ;;
esac
