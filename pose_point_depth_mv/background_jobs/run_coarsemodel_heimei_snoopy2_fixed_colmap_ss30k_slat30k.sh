#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/CoarseModel_heimei_snoopy2_指定帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v3}
EVAL_GPUS=${EVAL_GPUS:-0,3}

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

IFS=, read -r -a GPUS <<<"${EVAL_GPUS}"
if [[ ${#GPUS[@]} -ne 2 ]] || [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "ERROR: EVAL_GPUS must contain exactly two distinct physical GPU ids" >&2
  exit 90
fi

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k \
  prepare --output "${OUTPUT}"
mkdir -p "${OUTPUT}/logs"

# snoopy2 has no source mask tree.  The generated masks stay inside this
# versioned output and are checked before either model sees them.
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -u -m \
  pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k \
  segment-snoopy --output "${OUTPUT}" \
  2>&1 | tee "${OUTPUT}/logs/00_snoopy2_sam2.log"

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k \
  materialize-inputs --output "${OUTPUT}"

json_value() {
  local path=$1 expression=$2
  "${PYTHON}" -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); ${expression}" "${path}"
}

render_contour() {
  local runtime_manifest=$1 current_manifest=$2 contour=$3 object_name=$4 gpu=$5
  local mesh result object_key
  mesh=$(json_value "${current_manifest}" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["mesh"])')
  result=$(json_value "${current_manifest}" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["result"])')
  object_key=$(json_value "${runtime_manifest}" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["object_key"])')
  if [[ ! -s "${contour}/report.json" ]]; then
    test ! -e "${contour}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
      --runtime_input_manifest "${runtime_manifest}" \
      --mesh_o "${mesh}" --mesh_frame_report "${result}" \
      --output_dir "${contour}" --object "${object_key}" --contour_width 3 \
      --method_label "SS30K+SLat30K (native runtime-O)" \
      --overview_name "${object_name}_SS30K_SLat30K_COLMAPPose_runtimeO正确轮廓总览.png"
  fi
}

run_one() {
  local name=$1 gpu=$2
  shift 2
  local frames=("$@")
  local view_count=${#frames[@]}
  local root=${OUTPUT}/objects/${name}
  local dataset=${root}/00_fixed_input/${name}_fixed${view_count}
  local raw=${root}/01_raw_cache
  local runtime=${root}/02_runtime_o
  local model=${root}/03_dino_only_input
  local current=${root}/04_current_ss30k_slat30k
  local recon=${root}/05_reconviagen
  local contour=${root}/06_current_camera_contours
  local frame_args=()
  local frame
  for frame in "${frames[@]}"; do
    frame_args+=(--frame_name "${frame}")
  done

  if [[ ! -s "${raw}/raw_cache_report.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache \
      --dataset "${dataset}" \
      --sparse_override "${name}_fixed${view_count}=${dataset}/sparse/0" \
      --output_dir "${raw}" --min_registered_pairs "${view_count}" \
      --allow_empty_points --resume "${frame_args[@]}"
  fi
  if [[ ! -s "${runtime}/runtime_input_manifest.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
      --raw_cache_report "${raw}/raw_cache_report.json" \
      --output_dir "${runtime}" --selected_view_count "${view_count}" \
      --view_selection_policy lexical_even --geometry_mode pose_mask \
      --min_completed_objects 1 --resume
  fi
  if [[ ! -s "${model}/model_input_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
      --runtime_input_manifest "${runtime}/runtime_input_manifest.json" \
      --output_dir "${model}" --device cuda --resume
  fi
  if [[ ! -s "${current}/inference_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
      --model_input_manifest "${model}/model_input_manifest.json" \
      --native_ss_report "${SS_REPORT}" \
      --native_slat_checkpoint "${SLAT_CHECKPOINT}" --expected_slat_step 30000 \
      --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
      --stock_slat_freeze "${STOCK_FREEZE}" --output_dir "${current}" \
      --seeds 42 --weights ema --device cuda --amp_dtype bf16
  fi
  if [[ ! -s "${recon}/inference_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.infer_omni_real_reconviagen \
      --runtime_input_manifest "${runtime}/runtime_input_manifest.json" \
      --output_dir "${recon}" --seeds 42 --device cuda --low_vram
  fi
  render_contour "${runtime}/runtime_input_manifest.json" \
    "${current}/inference_manifest.json" "${contour}" "${name}" "${gpu}"
  echo "OBJECT COMPLETE name=${name} gpu=${gpu} views=${view_count}"
}

HEIMEI=(00000.jpg 00021.jpg 00058.jpg 00070.jpg 00253.jpg 00237.jpg 00267.jpg)
SNOOPY2=(00001.jpg 00011.jpg 00031.jpg 00050.jpg 00061.jpg 00081.jpg 00131.jpg 00141.jpg)

(run_one heimei "${GPUS[0]}" "${HEIMEI[@]}") \
  >"${OUTPUT}/logs/01_heimei_gpu${GPUS[0]}.log" 2>&1 &
PID_HEIMEI=$!
(run_one snoopy2 "${GPUS[1]}" "${SNOOPY2[@]}") \
  >"${OUTPUT}/logs/02_snoopy2_gpu${GPUS[1]}.log" 2>&1 &
PID_SNOOPY2=$!
echo "heimei pid=${PID_HEIMEI} gpu=${GPUS[0]}"
echo "snoopy2 pid=${PID_SNOOPY2} gpu=${GPUS[1]}"

worker_rc=0
wait "${PID_HEIMEI}" || worker_rc=1
wait "${PID_SNOOPY2}" || worker_rc=1
if ((worker_rc != 0)); then
  echo "ERROR: one or more object workers failed; outputs are preserved" >&2
  exit 91
fi

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k \
  finalize --output "${OUTPUT}"
echo "COARSEMODEL FIXED COLMAP TWO-OBJECT TEST COMPLETE: ${OUTPUT}"
