#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/真实采集5组_AR_COLMAP三选帧_固定official兼容O_SS30K_SLat30K轮廓_20260819_v3}
MATRIX_GPUS=${MATRIX_GPUS:-0,3,4,5,6}
MATRIX_MAX_PARALLEL=${MATRIX_MAX_PARALLEL:-2}

cd "${PROJECT_ROOT}"
export CUDA_HOME=${CUDA_HOME:-/home/zjr/cuda-12.1}
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "ERROR: CUDA toolkit is unavailable at CUDA_HOME=${CUDA_HOME}" >&2
  exit 89
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
SOURCE_V1=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/真实采集5组_AR与COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_三选帧轮廓_20260819_v1

for path in "${PYTHON}" "${SS_REPORT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"; do
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

"${PYTHON}" -u -m pose_point_depth_mv.real_capture_fixed_official_o_view_selection_v3 \
  prepare --output "${OUTPUT}"
mkdir -p "${OUTPUT}/logs"

DATASETS=(
  20260816_035545_862_axisuv_v5
  20260812_171117_303
  20260816_040547_970_axisuv_v5
  20260811_064454_154
  20260811_090511_346
)
SLUGS=(
  01_ar_training_spherical_fps8
  02_ar_time_uniform8
  03_ar_quality_spherical_fps8
  04_colmap_training_spherical_fps8
  05_colmap_time_uniform8
  06_colmap_quality_spherical_fps8
)
POSE_SOURCES=(ar ar ar colmap colmap colmap)
POLICIES=(
  training_spherical_farthest_valid_mask
  lexical_even_valid_mask_fallback
  object_spherical_farthest_valid_mask
  training_spherical_farthest_valid_mask
  lexical_even_valid_mask_fallback
  object_spherical_farthest_valid_mask
)

json_value() {
  local path=$1 expression=$2
  "${PYTHON}" -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); ${expression}" "${path}"
}

raw_report() {
  local dataset_name=$1 pose_source=$2 old_slug
  if [[ "${pose_source}" == ar ]]; then
    old_slug=01_ar_phone_spherical8
  else
    old_slug=04_colmap_spherical8
  fi
  printf '%s\n' "${SOURCE_V1}/objects/${dataset_name}/branches/${old_slug}/01_raw_cache/raw_cache_report.json"
}

prepare_runtime() {
  local dataset_name=$1 slot=$2
  local slug=${SLUGS[$slot]} pose_source=${POSE_SOURCES[$slot]} policy=${POLICIES[$slot]}
  local root=${OUTPUT}/objects/${dataset_name}/branches/${slug}
  local runtime=${root}/01_runtime_o
  local raw
  raw=$(raw_report "${dataset_name}" "${pose_source}")
  test -s "${raw}"
  mkdir -p "${root}"
  if [[ ! -s "${runtime}/runtime_input_manifest.json" ]]; then
    gravity=()
    if [[ "${pose_source}" == ar ]]; then gravity=(--gravity_up_w 0 1 0); fi
    "${PYTHON}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
      --raw_cache_report "${raw}" --output_dir "${runtime}" \
      --selected_view_count 8 --geometry_mode pose_mask \
      --view_selection_policy "${policy}" \
      --object_frame_view_scope all_foreground_valid \
      --model_o_axis_convention official_z_up \
      --min_completed_objects 1 --resume "${gravity[@]}"
  fi
}

audit_fixed_o() {
  local dataset_name=$1 start=$2
  local manifests=()
  for ((slot=start; slot<start+3; slot++)); do
    manifests+=("${OUTPUT}/objects/${dataset_name}/branches/${SLUGS[$slot]}/01_runtime_o/runtime_input_manifest.json")
  done
  "${PYTHON}" -c '
import json,sys,numpy as np
rows=[]
for path in sys.argv[1:]:
    report=json.load(open(path,encoding="utf-8")); row=report["objects"][0]
    with np.load(row["cache_npz"],allow_pickle=False) as cache:
        rows.append((path,row,np.asarray(cache["T_O2W"])))
assert all(row[1]["object_frame_view_scope"]=="all_foreground_valid" for row in rows)
assert all(row[1]["model_o_axis_convention"]=="official_z_up" for row in rows)
assert all(np.array_equal(rows[0][2],row[2]) for row in rows[1:])
assert len({row[1]["T_O2W_sha256"] for row in rows})==1
print({"same_T_O2W":True,"sha256":rows[0][1]["T_O2W_sha256"]})
' "${manifests[@]}"
}

run_branch_model() {
  local dataset_name=$1 slot=$2 gpu=$3
  local slug=${SLUGS[$slot]}
  local root=${OUTPUT}/objects/${dataset_name}/branches/${slug}
  local runtime=${root}/01_runtime_o
  local model=${root}/02_dino_only_input
  local current=${root}/03_current_ss30k_slat30k
  local contour=${root}/04_current_camera_contours
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
  if [[ ! -s "${contour}/report.json" ]]; then
    local mesh result object_key partial_contour
    if [[ -e "${contour}" ]]; then
      partial_contour="${contour}.partial_$(date -u +%Y%m%dT%H%M%SZ)_$$"
      mv -- "${contour}" "${partial_contour}"
      echo "PRESERVED PARTIAL CONTOUR OUTPUT: ${partial_contour}"
    fi
    mesh=$(json_value "${current}/inference_manifest.json" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["mesh"])')
    result=$(json_value "${current}/inference_manifest.json" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["result"])')
    object_key=$(json_value "${runtime}/runtime_input_manifest.json" 'assert p["passed"] and len(p["objects"])==1; print(p["objects"][0]["object_key"])')
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
      --runtime_input_manifest "${runtime}/runtime_input_manifest.json" \
      --mesh_o "${mesh}" --mesh_frame_report "${result}" \
      --output_dir "${contour}" --object "${object_key}" --contour_width 3 \
      --method_label "SS30K+SLat30K (fixed official-compatible O)" \
      --overview_name "SS30K_SLat30K_${slug}_固定O轮廓总览.png"
  fi
  echo "BRANCH COMPLETE dataset=${dataset_name} branch=${slug} gpu=${gpu}"
}

run_one() {
  local index=$1 gpu=$2 dataset_name=${DATASETS[$index]}
  echo "===== fixed-O runtime preparation: ${dataset_name} ====="
  for slot in 0 1 2 3 4 5; do prepare_runtime "${dataset_name}" "${slot}"; done
  audit_fixed_o "${dataset_name}" 0
  audit_fixed_o "${dataset_name}" 3
  echo "===== SS30K/SLat30K inference: ${dataset_name} ====="
  for slot in 0 1 2 3 4 5; do run_branch_model "${dataset_name}" "${slot}" "${gpu}"; done
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
  echo "ERROR: at least one fixed-O selector worker failed; outputs are preserved" >&2
  exit 92
fi

"${PYTHON}" -u -m pose_point_depth_mv.real_capture_fixed_official_o_view_selection_v3 \
  finalize --output "${OUTPUT}"
echo "REAL-CAPTURE FIXED-O THREE-SELECTOR V3 COMPLETE: ${OUTPUT}"
