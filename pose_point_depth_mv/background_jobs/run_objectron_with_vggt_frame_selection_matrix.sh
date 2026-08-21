#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATASET=${DATASET:-${PROJECT_ROOT}/yxc/datasets/Objectron_real_pose_2clips_20260819_v1}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/ObjectronCamera_三选帧_ReconViaGen_vs_VSS2k_VSLat15k_双RuntimeO轮廓_20260819_v1}

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

SS_REPORT=/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1/dev48_VSS_step2000_seed424344_2gpu03_manual_v3/aggregate_v1/report.json
V_CHECKPOINT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1/checkpoints/step_015000.pt
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PLAN=${OUTPUT}/experiment_plan.json

for path in "${PYTHON}" "${SS_REPORT}" "${V_CHECKPOINT}" "${STOCK_FREEZE}"; do
  test -s "${path}"
done
if [[ -s "${PLAN}" ]]; then
  test ! -e "${OUTPUT}/report.json"
  echo "===== REUSE FROZEN CPU PLAN: ${PLAN} ====="
else
  test ! -e "${OUTPUT}"
  echo "===== CPU PREPARE: Objectron cameras / selections / paired runtime-O ====="
  "${PYTHON}" -u -m pose_point_depth_mv.objectron_with_vggt_frame_selection_matrix \
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
  local slug=$1
  local gpu=$2
  if ! is_active "${slug}"; then return 0; fi
  local root=${OUTPUT}/${slug}
  local runtime=${root}/01_runtime_pose_mask/runtime_input_manifest.json
  local model=${root}/03_model_input_pose_mask
  echo "[model-input] strategy=${slug} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs \
    --runtime_input_manifest "${runtime}" \
    --output_dir "${model}" \
    --device cuda
}

echo "===== GPU PHASE 1: encode each accepted 8-view set exactly once ====="
pids=()
(build_pose_model_input 01_time_uniform8 5) \
  >"${OUTPUT}/logs/phase1_uniform_gpu5.log" 2>&1 & pids+=("$!")
(build_pose_model_input 02_phone_spherical8 4) \
  >"${OUTPUT}/logs/phase1_phone_gpu4.log" 2>&1 & pids+=("$!")
(build_pose_model_input 03_random8_seed20260819 3) \
  >"${OUTPUT}/logs/phase1_random_gpu3.log" 2>&1 & pids+=("$!")
phase1_rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then phase1_rc=1; fi
done
if ((phase1_rc != 0)); then
  echo "ERROR: model-input phase failed; inspect ${OUTPUT}/logs/phase1_*.log" >&2
  exit 91
fi

echo "===== CPU PAIRING: exact contexts + true-object-pose runtime geometry ====="
for slug in 01_time_uniform8 02_phone_spherical8 03_random8_seed20260819; do
  if ! is_active "${slug}"; then continue; fi
  root=${OUTPUT}/${slug}
  "${PYTHON}" -u -m pose_point_depth_mv.objectron_with_vggt_frame_selection_matrix \
    clone-model-input \
    --source-model-input-manifest "${root}/03_model_input_pose_mask/model_input_manifest.json" \
    --target-runtime-manifest "${root}/02_runtime_true_object_pose/runtime_input_manifest.json" \
    --output "${root}/04_model_input_true_pose"
done

run_current() {
  local slug=$1
  local o_mode=$2
  local gpu=$3
  if ! is_active "${slug}"; then return 0; fi
  local root=${OUTPUT}/${slug}
  local runtime model ours contour
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
    echo "unsupported o_mode=${o_mode}" >&2
    return 99
  fi
  echo "[current] strategy=${slug} o=${o_mode} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_official_with_vggt \
    --model_input_manifest "${model}" \
    --native_ss_report "${SS_REPORT}" \
    --native_slat_checkpoint "${V_CHECKPOINT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${ours}" \
    --seeds 42 --device cuda --amp_dtype bf16
  local mesh result
  mesh=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["mesh"])' "${ours}/inference_manifest.json")
  result=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["result"])' "${ours}/inference_manifest.json")
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
    --runtime_input_manifest "${runtime}" \
    --mesh_o "${mesh}" \
    --mesh_frame_report "${result}" \
    --output_dir "${contour}" \
    --object objectron_camera:camera_batch7_24 \
    --contour_width 3
}

run_recon() {
  local slug=$1
  local gpu=$2
  if ! is_active "${slug}"; then return 0; fi
  local root=${OUTPUT}/${slug}
  echo "[reconviagen] strategy=${slug} gpu=${gpu} (once, no O duplication)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_reconviagen \
    --runtime_input_manifest "${root}/01_runtime_pose_mask/runtime_input_manifest.json" \
    --output_dir "${root}/07_reconviagen_once" \
    --seeds 42 --device cuda --low_vram
}

echo "===== GPU PHASE 2: current paired-O branches + one ReconViaGen per strategy ====="
# One queue per physical GPU; jobs in a queue are serial, queues run in parallel.
(
  run_current 01_time_uniform8 pose_mask 5
  run_current 02_phone_spherical8 true_pose 5
) >"${OUTPUT}/logs/phase2_gpu5.log" 2>&1 & q5=$!
(
  run_current 01_time_uniform8 true_pose 3
  run_recon 03_random8_seed20260819 3
) >"${OUTPUT}/logs/phase2_gpu3.log" 2>&1 & q3=$!
(
  run_recon 01_time_uniform8 4
  run_current 02_phone_spherical8 pose_mask 4
) >"${OUTPUT}/logs/phase2_gpu4.log" 2>&1 & q4=$!
(
  run_current 03_random8_seed20260819 pose_mask 6
  run_recon 02_phone_spherical8 6
) >"${OUTPUT}/logs/phase2_gpu6.log" 2>&1 & q6=$!
(
  run_current 03_random8_seed20260819 true_pose 7
) >"${OUTPUT}/logs/phase2_gpu7.log" 2>&1 & q7=$!

phase2_rc=0
for pid in "${q5}" "${q3}" "${q4}" "${q6}" "${q7}"; do
  if ! wait "${pid}"; then phase2_rc=1; fi
done
if ((phase2_rc != 0)); then
  echo "ERROR: inference/contour phase failed; inspect ${OUTPUT}/logs/phase2_gpu*.log" >&2
  exit 92
fi

echo "===== FINAL CROSS-ARTIFACT AUDIT ====="
"${PYTHON}" -u -m pose_point_depth_mv.objectron_with_vggt_frame_selection_matrix \
  finalize --output "${OUTPUT}"

echo "OBJECTRON WITH-VGGT FRAME-SELECTION/O MATRIX COMPLETE: ${OUTPUT}"
