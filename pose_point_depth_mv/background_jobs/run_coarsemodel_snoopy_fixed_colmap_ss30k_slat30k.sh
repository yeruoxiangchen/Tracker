#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/CoarseModel_snoopy_指定8帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v1}
EVAL_GPU=${EVAL_GPU:-0}

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT_CHECKPOINT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
ABC_R_EVIDENCE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

for path in "${PYTHON}" "${SS_REPORT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_ss30k_slat30k \
  prepare --output "${OUTPUT}"
mkdir -p "${OUTPUT}/logs"
"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_ss30k_slat30k \
  materialize-input --output "${OUTPUT}"

ROOT=${OUTPUT}/objects/snoopy
DATASET=${ROOT}/00_fixed_input/snoopy_fixed8
RAW=${ROOT}/01_raw_cache
RUNTIME=${ROOT}/02_runtime_o
MODEL=${ROOT}/03_dino_only_input
CURRENT=${ROOT}/04_current_ss30k_slat30k
RECON=${ROOT}/05_reconviagen
CONTOUR=${ROOT}/06_current_camera_contours
FRAMES=(00001.jpg 00021.jpg 00051.jpg 00089.jpg 00101.jpg 00125.jpg 00131.jpg 00150.jpg)
FRAME_ARGS=()
for frame in "${FRAMES[@]}"; do FRAME_ARGS+=(--frame_name "${frame}"); done

if [[ ! -s "${RAW}/raw_cache_report.json" ]]; then
  "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache \
    --dataset "${DATASET}" --sparse_override "snoopy_fixed8=${DATASET}/sparse/0" \
    --output_dir "${RAW}" --min_registered_pairs 8 --allow_empty_points --resume \
    "${FRAME_ARGS[@]}"
fi
if [[ ! -s "${RUNTIME}/runtime_input_manifest.json" ]]; then
  "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
    --raw_cache_report "${RAW}/raw_cache_report.json" --output_dir "${RUNTIME}" \
    --selected_view_count 8 --view_selection_policy lexical_even \
    --geometry_mode pose_mask --min_completed_objects 1 --resume
fi
if [[ ! -s "${MODEL}/model_input_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${MODEL}" --device cuda --resume
fi
if [[ ! -s "${CURRENT}/inference_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
    --model_input_manifest "${MODEL}/model_input_manifest.json" \
    --native_ss_report "${SS_REPORT}" --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
    --expected_slat_step 30000 --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
    --stock_slat_freeze "${STOCK_FREEZE}" --output_dir "${CURRENT}" \
    --seeds 42 --weights ema --device cuda --amp_dtype bf16
fi
if [[ ! -s "${RECON}/inference_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_reconviagen \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${RECON}" --seeds 42 --device cuda --low_vram
fi

json_value() {
  local path=$1 expression=$2
  "${PYTHON}" -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); ${expression}" "${path}"
}
if [[ ! -s "${CONTOUR}/report.json" ]]; then
  mesh=$(json_value "${CURRENT}/inference_manifest.json" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["mesh"])')
  result=$(json_value "${CURRENT}/inference_manifest.json" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["result"])')
  object_key=$(json_value "${RUNTIME}/runtime_input_manifest.json" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["object_key"])')
  test ! -e "${CONTOUR}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --mesh_o "${mesh}" --mesh_frame_report "${result}" --output_dir "${CONTOUR}" \
    --object "${object_key}" --contour_width 3 \
    --method_label "SS30K+SLat30K (native runtime-O)" \
    --overview_name "snoopy_SS30K_SLat30K_COLMAPPose_runtimeO正确轮廓总览.png"
fi

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_fixed_colmap_ss30k_slat30k \
  finalize --output "${OUTPUT}"
echo "COARSEMODEL SNOOPY FIXED COLMAP TEST COMPLETE: ${OUTPUT}"
