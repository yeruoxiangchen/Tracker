#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
PIXAL_PY=/home/zjr/anaconda3/envs/pixal3d/bin/python
GPU=${REAL_S8_GPU:-5}
ROOT=/data/zjr/omni_real_video500_download_20260804_v2
ADAPT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2
SPLIT=${ROOT}/D6_novel500_dev64_holdout64_v3_pilotfree_eval/holdout.json
S7=${ROOT}/S7_benchmark32_fourway_realadapt_step1000_seed42_v1/report.json
INV=${ROOT}/S8A_holdout64_extraction_inventory_v1.json
RAW=${ROOT}/S8B_holdout64_raw_cache_v1
RUNTIME=${ROOT}/S8C_holdout64_runtime_o_v1
ALIGN=${ROOT}/S8D_holdout64_scan_to_colmap_w_v1
ADJ=${ALIGN}/alignment_adjudicated_v1.json
LABEL=${ROOT}/S8E_holdout64_mesh_o_labels_v1
MODEL=${ROOT}/S8F_holdout64_native_v2_model_inputs_v1
ADAPTED=${ROOT}/S8G_holdout64_native_v2_realadapt_step1000_seed42_v1
PARENT=${ROOT}/S8H_holdout64_native_v2_parent_seed42_v1
RECON=${ROOT}/S8I_holdout64_reconviagen_original_seed42_v1
PIXAL=${ROOT}/S8J_holdout64_pixal3d_official_seed42_v1
EVAL=${ROOT}/S8K_holdout64_fourway_realadapt_step1000_seed42_v1
SS_ADAPTED=${ADAPT}/ss_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SLAT_ADAPTED=${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SS_PARENT=/data/zjr/native3d_condition_ss_mixed1k_20260801_v1/ss868_sourceholdout_seed42_v1/checkpoints/step_002000.pt
SLAT_PARENT=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/train868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PIXAL_MODEL=/home/zjr/.cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots/0b31f9160aa400719af409098bff7936a932f726
NAF_ROOT=/data/zjr/models/valeoai_NAF_37f2dfc180f2de53d98bd601109c0da0dd6b0f43
STATE=${ADAPT}/logs/S8_holdout64_background.state
EXIT_CODE=${ADAPT}/logs/S8_holdout64_background.exit_code
LOCK=${ADAPT}/logs/S8_holdout64_background.lock

mkdir -p "${ADAPT}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "S8 start refused: another S8 job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "S8 background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${SPLIT}" "${S7}" "${SS_ADAPTED}" "${SLAT_ADAPTED}" \
                "${SS_PARENT}" "${SLAT_PARENT}" "${FREEZE}" \
                "${PIXAL_MODEL}/pipeline.json" "${NAF_ROOT}/repo/hubconf.py" \
                "${NAF_ROOT}/naf_release.pth" "${NAF_ROOT}/source_manifest.sha256"; do
  test -s "${REQUIRED}"
done

"${PY}" - "${S7}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True
assert report["formal"] is False and report["holdout64_consumed"] is False
assert report["adaptation_decision"]["primary_passed"] is True
print("S7 development gate passed; holdout64 is unlocked exactly once")
PY

if [ ! -s "${INV}" ]; then
  "${PY}" -m pose_point_depth_mv.dataset_tools.freeze_omni_real_raw_split \
    inventory --source_split "${SPLIT}" --output "${INV}"
fi
if [ ! -s "${RAW}/raw_cache_report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache \
    extract-cache --inventory "${INV}" --output_dir "${RAW}"
fi
if [ ! -s "${RUNTIME}/runtime_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${RUNTIME}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
    --raw_cache_report "${RAW}/raw_cache_report.json" \
    --output_dir "${RUNTIME}" \
    --selected_view_count 8 \
    "${RESUME[@]}"
fi
if [ ! -s "${ALIGN}/coarse_alignment_manifest.json" ]; then
  RESUME=()
  if [ -e "${ALIGN}" ]; then RESUME=(--resume); fi
  set +e
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.align_omni_real_mesh_to_colmap \
    --raw_cache_report "${RAW}/raw_cache_report.json" \
    --output_dir "${ALIGN}" \
    "${RESUME[@]}"
  ALIGN_RC=$?
  set -e
  if [ "${ALIGN_RC}" -ne 0 ] && [ "${ALIGN_RC}" -ne 2 ]; then
    exit "${ALIGN_RC}"
  fi
fi
if [ ! -s "${ADJ}" ]; then
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.adjudicate_omni_real_mesh_alignment \
    --source_alignment_manifest "${ALIGN}/coarse_alignment_manifest.json" \
    --output "${ADJ}" \
    --expected_objects 64
fi
"${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True and p["automatic_pass_count"]==64' "${ADJ}"

if [ ! -s "${LABEL}/runtime_o_label_manifest.json" ]; then
  RESUME=()
  if [ -e "${LABEL}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --alignment_manifest "${ADJ}" \
    --output_dir "${LABEL}" \
    "${RESUME[@]}"
fi
if [ ! -s "${MODEL}/model_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${MODEL}" ]; then RESUME=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${MODEL}" --device cuda \
    "${RESUME[@]}"
fi

run_native() {
  local SS=$1
  local SLAT=$2
  local OUT=$3
  if [ ! -s "${OUT}/inference_manifest.json" ]; then
    CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_v2 \
      --model_input_manifest "${MODEL}/model_input_manifest.json" \
      --native_ss_checkpoint "${SS}" \
      --native_slat_checkpoint "${SLAT}" \
      --stock_slat_freeze "${FREEZE}" \
      --output_dir "${OUT}" \
      --seeds 42 --weights ema --amp_dtype bf16 --device cuda
  fi
}
run_native "${SS_ADAPTED}" "${SLAT_ADAPTED}" "${ADAPTED}"
run_native "${SS_PARENT}" "${SLAT_PARENT}" "${PARENT}"

if [ ! -s "${RECON}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_reconviagen \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${RECON}" --seeds 42 --device cuda --low_vram
fi
if [ ! -s "${PIXAL}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PIXAL_PY}" -u -m pose_point_depth_mv.infer_omni_real_pixal3d \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${PIXAL}" \
    --model_path "${PIXAL_MODEL}" \
    --naf_repo "${NAF_ROOT}/repo" \
    --naf_checkpoint "${NAF_ROOT}/naf_release.pth" \
    --naf_source_manifest "${NAF_ROOT}/source_manifest.sha256" \
    --seeds 42 --device cuda --low_vram \
    --resolution 1024 --max_num_tokens 49152 --sampling_steps 12 \
    --isolate_objects --isolate_batch_size 1
fi

if [ ! -s "${EVAL}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_omni_real_native_adaptation \
    --label_manifest "${LABEL}/runtime_o_label_manifest.json" \
    --adapted_native_manifest "${ADAPTED}/inference_manifest.json" \
    --parent_native_manifest "${PARENT}/inference_manifest.json" \
    --reconviagen_manifest "${RECON}/inference_manifest.json" \
    --pixal3d_manifest "${PIXAL}/inference_manifest.json" \
    --output_dir "${EVAL}" \
    --protocol_scope formal_holdout64 \
    --frozen_split_manifest "${SPLIT}" \
    --expected_objects 64 \
    --surface_samples 20000
fi

"${PY}" - "${EVAL}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True and report["formal"] is True
assert report["holdout64_consumed"] is True
assert report["formal_holdout_binding"]["passed"] is True
print({
    "formal_protocol_passed": True,
    "adaptation_primary_passed": report["adaptation_decision"]["primary_passed"],
    "secondary_all_nonnegative": report["adaptation_decision"]["secondary_all_nonnegative"],
    "report": sys.argv[1],
})
PY
