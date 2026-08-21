#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-both}"
if [[ "$MODE" != "sparse" && "$MODE" != "mesh" && "$MODE" != "both" ]]; then
  echo "Usage: $0 [sparse|mesh|both]" >&2
  exit 2
fi

cd /home/zjr/Tracker

PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
GPU="${GPU:-0}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
CHECKPOINT="${CHECKPOINT:-/home/zjr/Tracker/ar_pose_trellis/checkpoints/sparse_image_pose_meshrgb_s2_e4.ckpt}"
MANIFEST="${MANIFEST:-/home/zjr/Tracker/ar_pose_trellis/test_manifests/GOOD_MESH_TEST_arpose.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/images}"
MASK_ROOT="${MASK_ROOT:-/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/masks}"
OUT_ROOT="${OUT_ROOT:-/home/zjr/Tracker/ar_pose_trellis/outputs/good_mesh_tests/visual_hull_logits_sweep}"

MAX_FRAMES="${MAX_FRAMES:-8}"
SS_STEPS="${SS_STEPS:-12}"
SLAT_STEPS="${SLAT_STEPS:-12}"
SS_GUIDANCE="${SS_GUIDANCE:-1.0}"
SLAT_GUIDANCE="${SLAT_GUIDANCE:-3.0}"
SPARSE_MIN_COORDS="${SPARSE_MIN_COORDS:-0}"
MESH_MIN_COORDS="${MESH_MIN_COORDS:-4096}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
MIN_VISIBLE_VIEWS="${MIN_VISIBLE_VIEWS:-1}"
PREVIEW_FRAMES="${PREVIEW_FRAMES:-72}"
PREVIEW_RESOLUTION="${PREVIEW_RESOLUTION:-320}"
PREVIEW_FPS="${PREVIEW_FPS:-15}"
LAMBDA_LIST="${LAMBDA_LIST:-0 20 40 80}"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.tsv"
printf "mode\tlambda\tstatus\tcoords\ttopk_fallback\tlogits_min\tlogits_max\tlogits_mean\tvisual_hull_score_mean\toutput_dir\tlog\n" > "$SUMMARY"

declare -a FAILURES=()

append_summary() {
  local mode_name="$1"
  local weight="$2"
  local status_name="$3"
  local output_dir="$4"
  local log_path="$5"
  local stats_path="$output_dir/sparse_stats.json"

  "$PYTHON_BIN" - "$mode_name" "$weight" "$status_name" "$output_dir" "$log_path" "$stats_path" >> "$SUMMARY" <<'PY'
import json
import sys

mode_name, weight, status_name, output_dir, log_path, stats_path = sys.argv[1:]
stats = {}
try:
    with open(stats_path, "r") as f:
        stats = json.load(f)
except Exception:
    pass

def val(name):
    value = stats.get(name, "NA")
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

print("\t".join([
    mode_name,
    str(weight),
    status_name,
    val("num_coords"),
    val("used_topk_fallback"),
    val("logits_min"),
    val("logits_max"),
    val("logits_mean"),
    val("visual_hull_score_mean"),
    output_dir,
    log_path,
]))
PY
}

run_case() {
  local mode_name="$1"
  local weight="$2"
  local output_dir="$OUT_ROOT/${mode_name}_w${weight}"
  local log_path="$output_dir/run.log"
  local min_coords="$SPARSE_MIN_COORDS"
  local -a mode_args

  mkdir -p "$output_dir"
  if [[ "$mode_name" == "sparse" ]]; then
    mode_args=(--only_sparse)
    min_coords="$SPARSE_MIN_COORDS"
  else
    mode_args=(
      --slat_steps "$SLAT_STEPS"
      --slat_guidance_strength "$SLAT_GUIDANCE"
      --skip_glb
      --preview_frames "$PREVIEW_FRAMES"
      --preview_resolution "$PREVIEW_RESOLUTION"
      --preview_fps "$PREVIEW_FPS"
    )
    min_coords="$MESH_MIN_COORDS"
  fi

  echo "[start] mode=$mode_name lambda=$weight output=$output_dir"
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}" \
    SPCONV_ALGO="${SPCONV_ALGO:-native}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" \
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" \
      ar_pose_trellis/generate_ar_pose_mesh.py \
      --weights "$WEIGHTS" \
      --checkpoint "$CHECKPOINT" \
      --manifest "$MANIFEST" \
      --image_root "$IMAGE_ROOT" \
      --mask_root "$MASK_ROOT" \
      --output_dir "$output_dir" \
      --max_frames "$MAX_FRAMES" \
      --ss_steps "$SS_STEPS" \
      --ss_guidance_strength "$SS_GUIDANCE" \
      --ss_min_coords "$min_coords" \
      --visual_hull_prior_weight "$weight" \
      --visual_hull_mask_threshold "$MASK_THRESHOLD" \
      --visual_hull_min_visible_views "$MIN_VISIBLE_VIEWS" \
      --cond_fp16 \
      "${mode_args[@]}" \
      > "$log_path" 2>&1

  local status_code=$?
  if [[ "$status_code" -eq 0 ]]; then
    echo "[done] mode=$mode_name lambda=$weight log=$log_path"
    append_summary "$mode_name" "$weight" "ok" "$output_dir" "$log_path"
  else
    echo "[failed] mode=$mode_name lambda=$weight status=$status_code log=$log_path"
    append_summary "$mode_name" "$weight" "failed:$status_code" "$output_dir" "$log_path"
    FAILURES+=("$mode_name:w$weight:$status_code:$log_path")
  fi
}

if [[ "$MODE" == "sparse" || "$MODE" == "both" ]]; then
  for weight in $LAMBDA_LIST; do
    run_case sparse "$weight"
  done
fi

if [[ "$MODE" == "mesh" || "$MODE" == "both" ]]; then
  for weight in $LAMBDA_LIST; do
    run_case mesh "$weight"
  done
fi

echo "[summary] $SUMMARY"
if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo "[failures]"
  printf "  %s\n" "${FAILURES[@]}"
  exit 1
fi
