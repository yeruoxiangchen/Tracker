#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/真实采集5组_AR_COLMAP六分支_SS30K_SLat30K_runtimeO正确轮廓_20260819_v2}
MATRIX_GPUS=${MATRIX_GPUS:-3,4,5,6,7}
# Loading five complete ReconViaGen pipelines at once can exhaust host RAM even
# though each pipeline is assigned to a different GPU.  Keep all five GPU
# identities available, but bound the number of dataset workers resident at
# the same time.  The workflow is artifact-resumable, so completed stages are
# skipped on restart.
MATRIX_MAX_PARALLEL=${MATRIX_MAX_PARALLEL:-2}

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
SS_CHECKPOINT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/ss/checkpoints/step_030000.pt
SLAT_CHECKPOINT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
ABC_R_EVIDENCE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

for path in "${PYTHON}" "${SS_REPORT}" "${SS_CHECKPOINT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done

IFS=, read -r -a GPUS <<<"${MATRIX_GPUS}"
if [[ ${#GPUS[@]} -ne 5 ]] || [[ $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) -ne 5 ]]; then
  echo "ERROR: MATRIX_GPUS must contain exactly five distinct physical GPU ids" >&2
  exit 90
fi
if ! [[ "${MATRIX_MAX_PARALLEL}" =~ ^[1-5]$ ]]; then
  echo "ERROR: MATRIX_MAX_PARALLEL must be an integer in [1,5]" >&2
  exit 91
fi

"${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_matrix_v2 \
  prepare --output "${OUTPUT}"
mkdir -p "${OUTPUT}/logs"

DATASETS=(
  20260816_035545_862_axisuv_v5
  20260812_171117_303
  20260816_040547_970_axisuv_v5
  20260811_064454_154
  20260811_090511_346
)

json_value() {
  local path=$1 expression=$2
  "${PYTHON}" -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); ${expression}" "${path}"
}

render_contour() {
  local runtime_manifest=$1 current_manifest=$2 contour=$3 slug=$4 gpu=$5
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
      --method_label "SS30K+SLat30K (runtime-O v2)" \
      --overview_name "SS30K_SLat30K_${slug}_runtimeO正确轮廓总览.png"
  fi
}

migrate_and_render() {
  local dataset_name=$1 slug=$2 old_slug=$3 gpu=$4
  local new_root=${OUTPUT}/objects/${dataset_name}/branches/${slug}
  local old_root=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1/objects/${dataset_name}/branches/${old_slug}
  "${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_matrix_v2 \
    migrate-legacy --output "${OUTPUT}" --dataset-name "${dataset_name}" --slug "${slug}"
  render_contour \
    "${old_root}/02_runtime_o/runtime_input_manifest.json" \
    "${new_root}/04_current_ss30k_slat30k/inference_manifest.json" \
    "${new_root}/06_current_camera_contours" "${slug}" "${gpu}"
  echo "CORRECTED branch=${slug} dataset=${dataset_name} source=${old_slug}"
}

run_fresh_ar_branch() {
  local dataset_name=$1 slug=$2 dataset=$3 gpu=$4
  local root=${OUTPUT}/objects/${dataset_name}/branches/${slug}
  local raw=${root}/01_raw_cache
  local runtime=${root}/02_runtime_o
  local model=${root}/03_dino_only_input
  local current=${root}/04_current_ss30k_slat30k
  local recon=${root}/05_reconviagen
  local contour=${root}/06_current_camera_contours
  mkdir -p "${root}"

  if [[ ! -s "${raw}/raw_cache_report.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache \
      --dataset "${dataset}" --output_dir "${raw}" \
      --min_registered_pairs 8 --allow_empty_points --resume
  fi
  if [[ ! -s "${runtime}/runtime_input_manifest.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
      --raw_cache_report "${raw}/raw_cache_report.json" \
      --output_dir "${runtime}" --selected_view_count 8 \
      --view_selection_policy lexical_even --geometry_mode pose_mask \
      --min_completed_objects 1 --gravity_up_w 0 1 0 --resume
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
    "${current}/inference_manifest.json" "${contour}" "${slug}" "${gpu}"
  echo "FRESH COMPLETE branch=${slug} dataset=${dataset_name} gpu=${gpu}"
}

run_one() {
  local index=$1 gpu=$2 dataset_name=${DATASETS[$1]}
  local ar_root=${OUTPUT}/objects/${dataset_name}/00_ar_pose/prepared_datasets

  echo "===== prepare frozen AR time/random selections: ${dataset_name} ====="
  "${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_matrix_v2 \
    prepare-ar --output "${OUTPUT}" --dataset-name "${dataset_name}"

  # Correct all already-computed branches first, so valid contours appear early.
  migrate_and_render "${dataset_name}" 03_ar_spherical8 01_ar_phone_spherical8 "${gpu}"
  migrate_and_render "${dataset_name}" 04_colmap_time_uniform8 02_colmap_time_uniform8 "${gpu}"
  migrate_and_render "${dataset_name}" 05_colmap_random8_seed20260819 03_colmap_random8_seed20260819 "${gpu}"
  migrate_and_render "${dataset_name}" 06_colmap_spherical8 04_colmap_spherical8 "${gpu}"

  # These two symmetric AR-pose branches were missing from v1 and require fresh inference.
  run_fresh_ar_branch "${dataset_name}" 01_ar_time_uniform8 \
    "${ar_root}/time_uniform8" "${gpu}"
  run_fresh_ar_branch "${dataset_name}" 02_ar_random8_seed20260819 \
    "${ar_root}/random8_seed20260819" "${gpu}"
}

worker_rc=0
for ((batch_start=0; batch_start<5; batch_start+=MATRIX_MAX_PARALLEL)); do
  pids=()
  batch_end=$((batch_start + MATRIX_MAX_PARALLEL))
  if ((batch_end > 5)); then batch_end=5; fi
  echo "===== resident-worker batch [${batch_start},${batch_end}) / 5 ====="
  for ((index=batch_start; index<batch_end; index++)); do
    log=${OUTPUT}/logs/worker_${index}_${DATASETS[$index]}_gpu${GPUS[$index]}.log
    (run_one "${index}" "${GPUS[$index]}") >>"${log}" 2>&1 &
    pids+=("$!")
    echo "worker=${index} dataset=${DATASETS[$index]} gpu=${GPUS[$index]} pid=${pids[-1]} log=${log}"
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then worker_rc=1; fi
  done
  if ((worker_rc != 0)); then break; fi
done
if ((worker_rc != 0)); then
  echo "ERROR: at least one corrected six-branch worker failed; outputs are preserved" >&2
  exit 91
fi

"${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_matrix_v2 \
  finalize --output "${OUTPUT}"
echo "REAL-CAPTURE SIX-BRANCH RUNTIME-O V2 COMPLETE: ${OUTPUT}"
