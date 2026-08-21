#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATASET=${DATASET:-${PROJECT_ROOT}/yxc/datasets/Objectron_real_pose_2clips_20260819_v1}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/ObjectronCamera_三选帧_ReconViaGen_vs_SS30K_SLat30K_双RuntimeO轮廓_20260819_v2}
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
PLAN=${OUTPUT}/experiment_plan.json

IFS=, read -r -a GPUS <<<"${MATRIX_GPUS}"
if [[ ${#GPUS[@]} -ne 5 ]] || [[ $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) -ne 5 ]]; then
  echo "ERROR: MATRIX_GPUS must contain exactly five distinct physical GPU ids" >&2
  exit 90
fi

for path in "${PYTHON}" "${SS_REPORT}" "${SS_CHECKPOINT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done

if [[ -s "${PLAN}" ]]; then
  test ! -e "${OUTPUT}/report.json"
  echo "===== REUSE FROZEN CPU PLAN: ${PLAN} ====="
else
  test ! -e "${OUTPUT}"
  echo "===== CPU PREPARE: Objectron cameras / selections / paired runtime-O ====="
  "${PYTHON}" -u -m pose_point_depth_mv.objectron_ss30k_slat30k_frame_selection_matrix \
    prepare --dataset "${DATASET}" --output "${OUTPUT}"
fi
mkdir -p "${OUTPUT}/logs"

is_active() {
  local slug=$1
  "${PYTHON}" - "${PLAN}" "${slug}" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1],encoding="utf-8"))
rows=[row for row in plan["strategies"] if row["slug"]==sys.argv[2]]
assert len(rows)==1
raise SystemExit(0 if rows[0]["active"] else 1)
PY
}

build_pose_model_input() {
  local slug=$1 gpu=$2
  if ! is_active "${slug}"; then return 0; fi
  local root=${OUTPUT}/${slug}
  echo "[DINO-only model-input] strategy=${slug} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${root}/01_runtime_pose_mask/runtime_input_manifest.json" \
    --output_dir "${root}/03_model_input_pose_mask" \
    --device cuda --resume
}

echo "===== GPU PHASE 1: DINO-only encode; VGGT forbidden ====="
pids=()
(build_pose_model_input 01_time_uniform8 "${GPUS[0]}") >"${OUTPUT}/logs/phase1_uniform_gpu${GPUS[0]}.log" 2>&1 & pids+=("$!")
(build_pose_model_input 02_phone_spherical8 "${GPUS[1]}") >"${OUTPUT}/logs/phase1_phone_gpu${GPUS[1]}.log" 2>&1 & pids+=("$!")
(build_pose_model_input 03_random8_seed20260819 "${GPUS[2]}") >"${OUTPUT}/logs/phase1_random_gpu${GPUS[2]}.log" 2>&1 & pids+=("$!")
phase1_rc=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then phase1_rc=1; fi; done
if ((phase1_rc != 0)); then
  echo "ERROR: DINO-only model-input phase failed; inspect phase1 logs" >&2
  exit 91
fi

echo "===== CPU PAIRING: exact DINO contexts + true-object-pose geometry ====="
for slug in 01_time_uniform8 02_phone_spherical8 03_random8_seed20260819; do
  if ! is_active "${slug}"; then continue; fi
  root=${OUTPUT}/${slug}
  if [[ ! -s "${root}/04_model_input_true_pose/model_input_manifest.json" ]]; then
    "${PYTHON}" -u -m pose_point_depth_mv.objectron_ss30k_slat30k_frame_selection_matrix \
      clone-model-input \
      --source-model-input-manifest "${root}/03_model_input_pose_mask/model_input_manifest.json" \
      --target-runtime-manifest "${root}/02_runtime_true_object_pose/runtime_input_manifest.json" \
      --output "${root}/04_model_input_true_pose"
  fi
done

run_current() {
  local slug=$1 o_mode=$2 gpu=$3
  if ! is_active "${slug}"; then return 0; fi
  local root=${OUTPUT}/${slug} runtime model ours contour
  if [[ "${o_mode}" == pose_mask ]]; then
    runtime=${root}/01_runtime_pose_mask/runtime_input_manifest.json
    model=${root}/03_model_input_pose_mask/model_input_manifest.json
    ours=${root}/05_current_pose_mask
    contour=${root}/08_contours_pose_mask
  elif [[ "${o_mode}" == true_pose ]]; then
    runtime=${root}/02_runtime_true_object_pose/runtime_input_manifest.json
    model=${root}/04_model_input_true_pose/model_input_manifest.json
    ours=${root}/06_current_true_pose
    contour=${root}/09_contours_true_pose
  else
    echo "unsupported o_mode=${o_mode}" >&2; return 99
  fi
  echo "[SS30K+SLat30K route-C] strategy=${slug} o=${o_mode} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
    --model_input_manifest "${model}" \
    --native_ss_report "${SS_REPORT}" \
    --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
    --expected_slat_step 30000 \
    --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${ours}" \
    --seeds 42 --weights ema --device cuda --amp_dtype bf16
  local mesh result
  mesh=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["mesh"])' "${ours}/inference_manifest.json")
  result=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["result"])' "${ours}/inference_manifest.json")
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
    --runtime_input_manifest "${runtime}" \
    --mesh_o "${mesh}" --mesh_frame_report "${result}" \
    --output_dir "${contour}" --object objectron_camera:camera_batch7_24 \
    --contour_width 3 --method_label "SS30K+SLat30K" \
    --overview_name "SS30K_SLat30K_原始8帧相机位姿轮廓总览.png"
}

run_recon() {
  local slug=$1 gpu=$2
  if ! is_active "${slug}"; then return 0; fi
  local root=${OUTPUT}/${slug}
  echo "[strict ReconViaGen] strategy=${slug} gpu=${gpu} (once; independent of runtime-O)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_reconviagen \
    --runtime_input_manifest "${root}/01_runtime_pose_mask/runtime_input_manifest.json" \
    --output_dir "${root}/07_reconviagen_once" \
    --seeds 42 --device cuda --low_vram
}

echo "===== GPU PHASE 2: two route-C O branches + one ReconViaGen per selection ====="
(run_current 01_time_uniform8 pose_mask "${GPUS[0]}") >"${OUTPUT}/logs/phase2_gpu${GPUS[0]}.log" 2>&1 & q0=$!
(run_current 01_time_uniform8 true_pose "${GPUS[1]}") >"${OUTPUT}/logs/phase2_gpu${GPUS[1]}.log" 2>&1 & q1=$!
(run_recon 01_time_uniform8 "${GPUS[2]}") >"${OUTPUT}/logs/phase2_gpu${GPUS[2]}.log" 2>&1 & q2=$!
(
  run_current 03_random8_seed20260819 pose_mask "${GPUS[3]}"
  run_recon 03_random8_seed20260819 "${GPUS[3]}"
) >"${OUTPUT}/logs/phase2_gpu${GPUS[3]}.log" 2>&1 & q3=$!
(run_current 03_random8_seed20260819 true_pose "${GPUS[4]}") >"${OUTPUT}/logs/phase2_gpu${GPUS[4]}.log" 2>&1 & q4=$!

phase2_rc=0
for pid in "${q0}" "${q1}" "${q2}" "${q3}" "${q4}"; do
  if ! wait "${pid}"; then phase2_rc=1; fi
done
if ((phase2_rc != 0)); then
  echo "ERROR: inference/contour phase failed; inspect phase2 logs" >&2
  exit 92
fi

echo "===== FINAL CROSS-ARTIFACT AUDIT ====="
"${PYTHON}" -u -m pose_point_depth_mv.objectron_ss30k_slat30k_frame_selection_matrix \
  finalize --output "${OUTPUT}"

echo "OBJECTRON SS30K+SLAT30K FRAME-SELECTION/O MATRIX COMPLETE: ${OUTPUT}"
