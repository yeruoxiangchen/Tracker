#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
COLMAP_BIN=${COLMAP_BIN:-/home/zjr/anaconda3/envs/foundpose/bin/colmap}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1}
MATRIX_GPUS=${MATRIX_GPUS:-3,4,5,6,7}

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
DATASET_ROOT=${PROJECT_ROOT}/pose_point_depth_mv/outputs/可视AR/datasets

for path in "${PYTHON}" "${COLMAP_BIN}" "${SS_REPORT}" "${SS_CHECKPOINT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done

IFS=, read -r -a GPUS <<<"${MATRIX_GPUS}"
if [[ ${#GPUS[@]} -ne 5 ]] || [[ $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) -ne 5 ]]; then
  echo "ERROR: MATRIX_GPUS must contain exactly five distinct physical GPU ids" >&2
  exit 90
fi

"${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_mask_batch \
  prepare --output "${OUTPUT}"
mkdir -p "${OUTPUT}/logs"

DATASETS=(
  20260816_035545_862_axisuv_v5
  20260812_171117_303
  20260816_040547_970_axisuv_v5
  20260811_064454_154
  20260811_090511_346
)

run_branch() {
  local dataset_name=$1 slug=$2 dataset=$3 policy=$4 gravity_mode=$5 gpu=$6
  local root=${OUTPUT}/objects/${dataset_name}/branches/${slug}
  local raw=${root}/01_raw_cache
  local runtime=${root}/02_runtime_o
  local model=${root}/03_dino_only_input
  local current=${root}/04_current_ss30k_slat30k
  local recon=${root}/05_reconviagen
  local contour=${root}/06_current_camera_contours

  mkdir -p "${root}"
  echo "===== branch=${slug} dataset=${dataset_name} gpu=${gpu} ====="
  if [[ ! -s "${raw}/raw_cache_report.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache \
      --dataset "${dataset}" --output_dir "${raw}" \
      --min_registered_pairs 8 --allow_empty_points --resume
  fi

  if [[ ! -s "${runtime}/runtime_input_manifest.json" ]]; then
    runtime_command=(
      "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs
      --raw_cache_report "${raw}/raw_cache_report.json"
      --output_dir "${runtime}"
      --selected_view_count 8
      --view_selection_policy "${policy}"
      --geometry_mode pose_mask
      --min_completed_objects 1
      --resume
    )
    if [[ "${gravity_mode}" == phone_y_up ]]; then
      runtime_command+=(--gravity_up_w 0 1 0)
    elif [[ "${gravity_mode}" != arbitrary_colmap_gauge ]]; then
      echo "ERROR: unsupported gravity_mode=${gravity_mode}" >&2
      return 93
    fi
    "${runtime_command[@]}"
  fi

  "${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_mask_batch \
    check-runtime --manifest "${runtime}/runtime_input_manifest.json" --allow-diagnostic

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

  local mesh result object_key
  mesh=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["mesh"])' "${current}/inference_manifest.json")
  result=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["result"])' "${current}/inference_manifest.json")
  object_key=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["object_key"])' "${runtime}/runtime_input_manifest.json")
  if [[ ! -s "${contour}/report.json" ]]; then
    test ! -e "${contour}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
      --runtime_input_manifest "${runtime}/runtime_input_manifest.json" \
      --mesh_o "${mesh}" --mesh_frame_report "${result}" \
      --output_dir "${contour}" --object "${object_key}" --contour_width 3 \
      --method_label "SS30K+SLat30K" \
      --overview_name "SS30K_SLat30K_${slug}_原始8帧相机轮廓总览.png"
  fi
  echo "COMPLETE branch=${slug} dataset=${dataset_name} gpu=${gpu}"
}

run_one() {
  local index=$1 gpu=$2 dataset_name=${DATASETS[$1]}
  local object_root=${OUTPUT}/objects/${dataset_name}
  local prepared=${object_root}/00_offline_colmap/prepared_datasets

  echo "===== offline COLMAP all-frame SfM/BA: dataset=${dataset_name} gpu=${gpu} ====="
  "${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_mask_batch \
    prepare-colmap --output "${OUTPUT}" --dataset-name "${dataset_name}" \
    --gpu "${gpu}" --colmap-bin "${COLMAP_BIN}" --resume

  run_branch "${dataset_name}" 01_ar_phone_spherical8 \
    "${DATASET_ROOT}/${dataset_name}" object_spherical_farthest_valid_mask \
    phone_y_up "${gpu}"
  run_branch "${dataset_name}" 02_colmap_time_uniform8 \
    "${prepared}/time_uniform8" lexical_even arbitrary_colmap_gauge "${gpu}"
  run_branch "${dataset_name}" 03_colmap_random8_seed20260819 \
    "${prepared}/random8_seed20260819" lexical_even arbitrary_colmap_gauge "${gpu}"
  run_branch "${dataset_name}" 04_colmap_spherical8 \
    "${prepared}/spherical_pool_all_registered" object_spherical_farthest_valid_mask \
    arbitrary_colmap_gauge "${gpu}"
}

pids=()
for index in 0 1 2 3 4; do
  log=${OUTPUT}/logs/worker_${index}_${DATASETS[$index]}_gpu${GPUS[$index]}.log
  (run_one "${index}" "${GPUS[$index]}") >"${log}" 2>&1 &
  pids+=("$!")
  echo "worker=${index} dataset=${DATASETS[$index]} gpu=${GPUS[$index]} pid=${pids[-1]} log=${log}"
done

worker_rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then worker_rc=1; fi
done
if ((worker_rc != 0)); then
  echo "ERROR: at least one AR/COLMAP real-capture worker failed; outputs are preserved" >&2
  exit 91
fi

"${PYTHON}" -u -m pose_point_depth_mv.real_capture_ss30k_slat30k_pose_mask_batch \
  finalize --output "${OUTPUT}"
echo "REAL-CAPTURE AR/COLMAP SS30K+SLAT30K BATCH COMPLETE: ${OUTPUT}"
