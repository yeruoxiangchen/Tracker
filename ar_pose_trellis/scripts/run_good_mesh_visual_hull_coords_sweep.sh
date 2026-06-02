#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker

PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}"
GPU="${GPU:-0}"
WEIGHTS="${WEIGHTS:-microsoft/TRELLIS-image-large}"
CHECKPOINT="${CHECKPOINT:-/home/zjr/Tracker/ar_pose_trellis/checkpoints/sparse_image_pose_meshrgb_s2_e4.ckpt}"
MANIFEST="${MANIFEST:-/home/zjr/Tracker/ar_pose_trellis/test_manifests/GOOD_MESH_TEST_arpose.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/images}"
MASK_ROOT="${MASK_ROOT:-/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST/masks}"
OUT_ROOT="${OUT_ROOT:-/home/zjr/Tracker/ar_pose_trellis/outputs/good_mesh_tests/direct_visual_hull_coords_sweep}"

MAX_FRAMES="${MAX_FRAMES:-8}"
SLAT_STEPS="${SLAT_STEPS:-12}"
SLAT_GUIDANCE="${SLAT_GUIDANCE:-3.0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
MIN_VISIBLE_VIEWS="${MIN_VISIBLE_VIEWS:-1}"
VH_MAX_COORDS="${VH_MAX_COORDS:-8192}"
PREVIEW_FRAMES="${PREVIEW_FRAMES:-72}"
PREVIEW_RESOLUTION="${PREVIEW_RESOLUTION:-320}"
PREVIEW_FPS="${PREVIEW_FPS:-15}"
CASES="${CASES:-s2_r04:2:0.4 s2_r06:2:0.6 s2_r08:2:0.8 s3_r06:3:0.6}"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.tsv"
printf "case\tstatus\tcoords\traw_coords\tsupport_views\tsupport_ratio\toutput_dir\tlog\n" > "$SUMMARY"

declare -a FAILURES=()

append_summary() {
  local case_name="$1"
  local status_name="$2"
  local output_dir="$3"
  local log_path="$4"
  local stats_path="$output_dir/sparse_stats.json"

  "$PYTHON_BIN" - "$case_name" "$status_name" "$output_dir" "$log_path" "$stats_path" >> "$SUMMARY" <<'PY'
import json
import sys

case_name, status_name, output_dir, log_path, stats_path = sys.argv[1:]
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
    case_name,
    status_name,
    val("num_coords"),
    val("visual_hull_raw_num_coords"),
    val("min_support_views"),
    val("min_support_ratio"),
    output_dir,
    log_path,
]))
PY
}

run_case() {
  local spec="$1"
  local case_name support_views support_ratio
  IFS=: read -r case_name support_views support_ratio <<< "$spec"
  local output_dir="$OUT_ROOT/$case_name"
  local log_path="$output_dir/run.log"

  mkdir -p "$output_dir"
  echo "[start] case=$case_name support_views=$support_views support_ratio=$support_ratio output=$output_dir"
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
      --coords_source visual_hull \
      --visual_hull_mask_threshold "$MASK_THRESHOLD" \
      --visual_hull_min_visible_views "$MIN_VISIBLE_VIEWS" \
      --visual_hull_min_support_views "$support_views" \
      --visual_hull_min_support_ratio "$support_ratio" \
      --visual_hull_max_coords "$VH_MAX_COORDS" \
      --slat_steps "$SLAT_STEPS" \
      --slat_guidance_strength "$SLAT_GUIDANCE" \
      --skip_glb \
      --preview_frames "$PREVIEW_FRAMES" \
      --preview_resolution "$PREVIEW_RESOLUTION" \
      --preview_fps "$PREVIEW_FPS" \
      --cond_fp16 \
      > "$log_path" 2>&1

  local status_code=$?
  if [[ "$status_code" -eq 0 ]]; then
    echo "[done] case=$case_name log=$log_path"
    append_summary "$case_name" "ok" "$output_dir" "$log_path"
  else
    echo "[failed] case=$case_name status=$status_code log=$log_path"
    append_summary "$case_name" "failed:$status_code" "$output_dir" "$log_path"
    FAILURES+=("$case_name:$status_code:$log_path")
  fi
}

for spec in $CASES; do
  run_case "$spec"
done

echo "[summary] $SUMMARY"
if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo "[failures]"
  printf "  %s\n" "${FAILURES[@]}"
  exit 1
fi
