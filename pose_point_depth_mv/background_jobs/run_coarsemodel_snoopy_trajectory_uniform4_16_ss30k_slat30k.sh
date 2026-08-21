#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/CoarseModel_snoopy_轨迹均匀4帧16帧_ReconViaGen_vs_SS30K_SLat30K_20260819_v1}
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

IFS=, read -r -a GPUS <<<"${EVAL_GPUS}"
if [[ ${#GPUS[@]} -ne 2 ]] || [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "ERROR: EVAL_GPUS must contain two distinct physical GPU ids" >&2
  exit 90
fi
for path in "${PYTHON}" "${SS_REPORT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_trajectory_uniform_ss30k_slat30k \
  prepare --output "${OUTPUT}"
mkdir -p "${OUTPUT}/logs"
"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_trajectory_uniform_ss30k_slat30k \
  materialize --output "${OUTPUT}"

run_branch() {
  local branch=$1 gpu=$2
  shift 2
  local frames=("$@")
  local count=${#frames[@]}
  local root=${OUTPUT}/branches/${branch}
  local dataset=${root}/00_fixed_input/snoopy_${branch}_${count}views
  local raw=${root}/01_raw_cache
  local runtime=${root}/02_runtime_o
  local model=${root}/03_dino_only_input
  local current=${root}/04_current_ss30k_slat30k
  local recon=${root}/05_reconviagen
  local args=()
  local frame
  for frame in "${frames[@]}"; do args+=(--frame_name "${frame}"); done
  if [[ ! -s "${raw}/raw_cache_report.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache \
      --dataset "${dataset}" --sparse_override "$(basename "${dataset}")=${dataset}/sparse/0" \
      --output_dir "${raw}" --min_registered_pairs "${count}" --allow_empty_points --resume \
      "${args[@]}"
  fi
  if [[ ! -s "${runtime}/runtime_input_manifest.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
      --raw_cache_report "${raw}/raw_cache_report.json" --output_dir "${runtime}" \
      --selected_view_count "${count}" --view_selection_policy lexical_even \
      --geometry_mode pose_mask --min_completed_objects 1 --resume
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
      --native_ss_report "${SS_REPORT}" --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
      --expected_slat_step 30000 --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
      --stock_slat_freeze "${STOCK_FREEZE}" --output_dir "${current}" \
      --seeds 42 --weights ema --device cuda --amp_dtype bf16
  fi
  if [[ ! -s "${recon}/inference_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.infer_omni_real_reconviagen \
      --runtime_input_manifest "${runtime}/runtime_input_manifest.json" \
      --output_dir "${recon}" --seeds 42 --device cuda --low_vram
  fi
  echo "BRANCH COMPLETE branch=${branch} views=${count} gpu=${gpu}"
}

FRAMES4=(00001.jpg 00064.jpg 00106.jpg 00153.jpg)
FRAMES16=(00001.jpg 00018.jpg 00032.jpg 00045.jpg 00054.jpg 00064.jpg 00075.jpg 00087.jpg 00092.jpg 00098.jpg 00106.jpg 00115.jpg 00124.jpg 00134.jpg 00143.jpg 00153.jpg)

(run_branch trajectory_uniform4 "${GPUS[0]}" "${FRAMES4[@]}") >"${OUTPUT}/logs/trajectory_uniform4_gpu${GPUS[0]}.log" 2>&1 &
PID4=$!
(run_branch trajectory_uniform16 "${GPUS[1]}" "${FRAMES16[@]}") >"${OUTPUT}/logs/trajectory_uniform16_gpu${GPUS[1]}.log" 2>&1 &
PID16=$!
echo "trajectory_uniform4 pid=${PID4} gpu=${GPUS[0]}"
echo "trajectory_uniform16 pid=${PID16} gpu=${GPUS[1]}"
worker_rc=0
wait "${PID4}" || worker_rc=1
wait "${PID16}" || worker_rc=1
if ((worker_rc != 0)); then
  echo "ERROR: one or more trajectory branches failed; outputs are preserved" >&2
  exit 91
fi

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_trajectory_uniform_ss30k_slat30k \
  finalize --output "${OUTPUT}"
echo "COARSEMODEL SNOOPY TRAJECTORY 4/16 TEST COMPLETE: ${OUTPUT}"
