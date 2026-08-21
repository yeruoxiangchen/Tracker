#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATASET=${DATASET:-${PROJECT_ROOT}/yxc/datasets/Objectron_real_pose_2clips_20260819_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/Objectron_ShoeCamera_GT中心引导_训练一致球面FPS8_SS30K_SLat30K_vs_ReconViaGen_20260820_v1}
GT_CENTER_GPUS=${GT_CENTER_GPUS:-0,1,2,3}

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

MODULE=pose_point_depth_mv.objectron_ss30k_slat30k_frame_selection_matrix
SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT_CHECKPOINT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
ABC_R_EVIDENCE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

SHOE=${OUTPUT_ROOT}/shoe_batch14_30_obj0
CAMERA=${OUTPUT_ROOT}/camera_batch7_24_obj0
SLUG=01_gt_center_training_spherical_fps8

IFS=, read -r -a GPUS <<<"${GT_CENTER_GPUS}"
if [[ ${#GPUS[@]} -ne 4 ]] || [[ $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) -ne 4 ]]; then
  echo "ERROR: GT_CENTER_GPUS must contain exactly four distinct physical GPU ids" >&2
  exit 90
fi
for path in "${PYTHON}" "${SS_REPORT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done
test -d "${DATASET}/clips/shoe/batch-14/30/frames"
test -d "${DATASET}/clips/shoe/batch-14/30/masks"
test -d "${DATASET}/clips/camera/batch-7/24/frames"
test -d "${DATASET}/clips/camera/batch-7/24/masks"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

prepare_clip() {
  local output=$1 clip=$2 category=$3 object_id=$4 label=$5
  if [[ -s "${output}/experiment_plan.json" ]]; then
    echo "[reuse GT-centre plan] ${output}/experiment_plan.json"
    return 0
  fi
  if [[ -e "${output}" ]]; then
    echo "ERROR: partial preparation exists; preserve and inspect: ${output}" >&2
    return 91
  fi
  "${PYTHON}" -u -m "${MODULE}" prepare-gt-center \
    --dataset "${DATASET}" --output "${output}" \
    --clip-sequence "${clip}" --official-object-id 0 \
    --object-category "${category}" --object-id "${object_id}" \
    --experiment-label "${label}"
}

prepare_clip "${SHOE}" shoe/batch-14/30 \
  objectron_shoe shoe_batch14_30_obj0 "Objectron Shoe object_id=0"
prepare_clip "${CAMERA}" camera/batch-7/24 \
  objectron_camera camera_batch7_24_obj0 "Objectron Camera object_id=0"

build_pose_model_input() {
  local output=$1 gpu=$2 root=${output}/${SLUG}
  if [[ -s "${root}/03_model_input_pose_mask/model_input_manifest.json" ]]; then
    echo "[reuse DINO-only input] ${output}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${root}/01_runtime_pose_mask/runtime_input_manifest.json" \
    --output_dir "${root}/03_model_input_pose_mask" \
    --device cuda --resume
}

echo "===== PHASE 1: shoe/camera DINO-only inputs ====="
(build_pose_model_input "${SHOE}" "${GPUS[0]}") \
  >"${OUTPUT_ROOT}/logs/phase1_shoe_gpu${GPUS[0]}.log" 2>&1 & p0=$!
(build_pose_model_input "${CAMERA}" "${GPUS[2]}") \
  >"${OUTPUT_ROOT}/logs/phase1_camera_gpu${GPUS[2]}.log" 2>&1 & p1=$!
phase_rc=0
if ! wait "${p0}"; then phase_rc=1; fi
if ! wait "${p1}"; then phase_rc=1; fi
if ((phase_rc != 0)); then
  echo "ERROR: DINO-only phase failed; inspect ${OUTPUT_ROOT}/logs/phase1_*" >&2
  exit 92
fi

clone_true_pose_input() {
  local output=$1 root=${output}/${SLUG}
  if [[ -s "${root}/04_model_input_true_pose/model_input_manifest.json" ]]; then
    echo "[reuse paired true-pose input] ${output}"
    return 0
  fi
  if [[ -e "${root}/04_model_input_true_pose" ]]; then
    echo "ERROR: partial true-pose model input exists: ${root}/04_model_input_true_pose" >&2
    return 93
  fi
  "${PYTHON}" -u -m "${MODULE}" clone-model-input \
    --source-model-input-manifest \
      "${root}/03_model_input_pose_mask/model_input_manifest.json" \
    --target-runtime-manifest \
      "${root}/02_runtime_true_object_pose/runtime_input_manifest.json" \
    --output "${root}/04_model_input_true_pose"
}

clone_true_pose_input "${SHOE}"
clone_true_pose_input "${CAMERA}"

run_current() {
  local output=$1 object_key=$2 o_mode=$3 gpu=$4
  local root=${output}/${SLUG} runtime model inference contour
  if [[ "${o_mode}" == pose_mask ]]; then
    runtime=${root}/01_runtime_pose_mask/runtime_input_manifest.json
    model=${root}/03_model_input_pose_mask/model_input_manifest.json
    inference=${root}/05_current_pose_mask
    contour=${root}/08_contours_pose_mask
  elif [[ "${o_mode}" == true_pose ]]; then
    runtime=${root}/02_runtime_true_object_pose/runtime_input_manifest.json
    model=${root}/04_model_input_true_pose/model_input_manifest.json
    inference=${root}/06_current_true_pose
    contour=${root}/09_contours_true_pose
  else
    echo "ERROR: unsupported O mode: ${o_mode}" >&2
    return 94
  fi

  if [[ ! -s "${inference}/inference_manifest.json" ]]; then
    if [[ -e "${inference}" ]]; then
      echo "ERROR: partial current inference exists: ${inference}" >&2
      return 95
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
      --model_input_manifest "${model}" \
      --native_ss_report "${SS_REPORT}" \
      --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
      --expected_slat_step 30000 \
      --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --output_dir "${inference}" \
      --seeds 42 --weights ema --device cuda --amp_dtype bf16
  else
    echo "[reuse current inference] ${inference}"
  fi

  if [[ ! -s "${contour}/report.json" ]]; then
    if [[ -e "${contour}" ]]; then
      echo "ERROR: partial contour output exists: ${contour}" >&2
      return 96
    fi
    local mesh result
    mesh=$("${PYTHON}" -c \
      'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["passed"] and len(r["objects"])==1; print(r["objects"][0]["mesh"])' \
      "${inference}/inference_manifest.json")
    result=$("${PYTHON}" -c \
      'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["passed"] and len(r["objects"])==1; print(r["objects"][0]["result"])' \
      "${inference}/inference_manifest.json")
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
      --runtime_input_manifest "${runtime}" \
      --mesh_o "${mesh}" --mesh_frame_report "${result}" \
      --output_dir "${contour}" --object "${object_key}" \
      --contour_width 3 --method_label "SS30K+SLat30K" \
      --overview_name "SS30K_SLat30K_GT中心FPS8_原始8帧轮廓总览.png"
  else
    echo "[reuse contours] ${contour}"
  fi
}

run_reconviagen() {
  local output=$1 gpu=$2 root=${output}/${SLUG}
  local inference=${root}/07_reconviagen_once
  if [[ -s "${inference}/inference_manifest.json" ]]; then
    echo "[reuse strict ReconViaGen] ${inference}"
    return 0
  fi
  if [[ -e "${inference}" ]]; then
    echo "ERROR: partial ReconViaGen output exists: ${inference}" >&2
    return 97
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_reconviagen \
    --runtime_input_manifest \
      "${root}/01_runtime_pose_mask/runtime_input_manifest.json" \
    --output_dir "${inference}" --seeds 42 --device cuda --low_vram
}

echo "===== PHASE 2: two clips x two O branches + matched ReconViaGen ====="
(
  run_current "${SHOE}" objectron_shoe:shoe_batch14_30_obj0 pose_mask "${GPUS[0]}"
  run_reconviagen "${SHOE}" "${GPUS[0]}"
) >"${OUTPUT_ROOT}/logs/phase2_shoe_pose_recon_gpu${GPUS[0]}.log" 2>&1 & q0=$!
(run_current "${SHOE}" objectron_shoe:shoe_batch14_30_obj0 true_pose "${GPUS[1]}") \
  >"${OUTPUT_ROOT}/logs/phase2_shoe_true_gpu${GPUS[1]}.log" 2>&1 & q1=$!
(
  run_current "${CAMERA}" objectron_camera:camera_batch7_24_obj0 pose_mask "${GPUS[2]}"
  run_reconviagen "${CAMERA}" "${GPUS[2]}"
) >"${OUTPUT_ROOT}/logs/phase2_camera_pose_recon_gpu${GPUS[2]}.log" 2>&1 & q2=$!
(run_current "${CAMERA}" objectron_camera:camera_batch7_24_obj0 true_pose "${GPUS[3]}") \
  >"${OUTPUT_ROOT}/logs/phase2_camera_true_gpu${GPUS[3]}.log" 2>&1 & q3=$!

phase2_rc=0
for pid in "${q0}" "${q1}" "${q2}" "${q3}"; do
  if ! wait "${pid}"; then phase2_rc=1; fi
done
if ((phase2_rc != 0)); then
  echo "ERROR: inference phase failed; inspect ${OUTPUT_ROOT}/logs/phase2_*" >&2
  exit 98
fi

for output in "${SHOE}" "${CAMERA}"; do
  if [[ ! -s "${output}/report.json" ]]; then
    "${PYTHON}" -u -m "${MODULE}" finalize --output "${output}"
  fi
done
if [[ ! -s "${OUTPUT_ROOT}/report.json" ]]; then
  "${PYTHON}" -u -m "${MODULE}" finalize-gt-center-pair \
    --output-root "${OUTPUT_ROOT}" \
    --shoe-output "${SHOE}" --camera-output "${CAMERA}"
fi

echo "OBJECTRON GT-CENTRE SHOE+CAMERA COMPLETE: ${OUTPUT_ROOT}/report.json"
