#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
CURRENT_GPU=${CURRENT_GPU:-0}
RECON_GPU=${RECON_GPU:-7}
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-22000}
SHAPE_WORKERS=${SHAPE_WORKERS:-4}
OUTPUT=${OUTPUT:-/data/zjr/omni_holdout64_posemask_ss30k_slat30k_vs_recon_shape_20260819_v1}

OMNI=/data/zjr/omni_real_video500_download_20260804_v2
POSE_RUNTIME=${OMNI}/M11N_holdout64_pose_mask_runtime_o_blind_v1/runtime_input_manifest.json
POSE_MODEL_INPUT=${OMNI}/M11O_holdout64_pose_mask_dino_only_blind_v1/model_input_manifest.json
LABEL=${OMNI}/M11E_holdout64_mesh_o_labels_v1/runtime_o_label_manifest.json

SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT_CHECKPOINT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
ABC_R_EVIDENCE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

CURRENT_OUT=${OUTPUT}/01_ss30k_slat30k_posemask_seed42
RECON_OUT=${OUTPUT}/02_reconviagen_posemask_seed42
SHAPE_OUT=${OUTPUT}/03_end_to_end_normalized_proper_sim3_shape
LOG_ROOT=${OUTPUT}/logs

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ReconViaGen:${PROJECT_ROOT}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

if [[ "${CURRENT_GPU}" == "${RECON_GPU}" ]]; then
  echo "ERROR: CURRENT_GPU and RECON_GPU must differ" >&2
  exit 90
fi
if ! [[ "${SHAPE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SHAPE_WORKERS must be positive" >&2
  exit 91
fi
for path in \
  "${PYTHON}" "${POSE_RUNTIME}" "${POSE_MODEL_INPUT}" "${LABEL}" \
  "${SS_REPORT}" "${SLAT_CHECKPOINT}" "${ABC_R_EVIDENCE}" "${STOCK_FREEZE}"
do
  test -s "${path}"
done
mkdir -p "${LOG_ROOT}"

gpu_preflight() {
  local gpu=$1
  local free_mib
  free_mib=$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
  if ! [[ "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_GPU_MIB )); then
    echo "ERROR: physical GPU ${gpu} free=${free_mib:-?} MiB; required=${MIN_FREE_GPU_MIB}" >&2
    return 95
  fi
  echo "GPU ${gpu} preflight passed: free=${free_mib} MiB"
}

manifest_complete() {
  local path=$1 expected_format=$2 expected_method=$3
  [[ -s "${path}" ]] || return 1
  "${PYTHON}" -c '
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r.get("format") == sys.argv[2]
assert r.get("method") == sys.argv[3]
assert r.get("passed") is True
assert r.get("seeds") == [42]
assert int(r.get("object_count",-1)) == 64
assert int(r.get("record_count",-1)) == 64
' "${path}" "${expected_format}" "${expected_method}" >/dev/null 2>&1
}

CURRENT_MANIFEST=${CURRENT_OUT}/inference_manifest.json
RECON_MANIFEST=${RECON_OUT}/inference_manifest.json
CURRENT_FORMAT=pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference_manifest.v1
CURRENT_METHOD=proobjaverse_official_native_ss_trained_slat
RECON_FORMAT=pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1
RECON_METHOD=reconviagen_original

if ! manifest_complete "${CURRENT_MANIFEST}" "${CURRENT_FORMAT}" "${CURRENT_METHOD}"; then
  gpu_preflight "${CURRENT_GPU}"
fi
if ! manifest_complete "${RECON_MANIFEST}" "${RECON_FORMAT}" "${RECON_METHOD}"; then
  gpu_preflight "${RECON_GPU}"
fi

current_pid=""
recon_pid=""
if manifest_complete "${CURRENT_MANIFEST}" "${CURRENT_FORMAT}" "${CURRENT_METHOD}"; then
  echo "reuse complete SS30K+SLat30K inference: ${CURRENT_MANIFEST}"
else
  (
    CUDA_VISIBLE_DEVICES="${CURRENT_GPU}" "${PYTHON}" -u -m \
      pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
      --model_input_manifest "${POSE_MODEL_INPUT}" \
      --native_ss_report "${SS_REPORT}" \
      --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
      --expected_slat_step 30000 \
      --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --output_dir "${CURRENT_OUT}" \
      --seeds 42 --weights ema --device cuda --amp_dtype bf16
  ) >"${LOG_ROOT}/01_ss30k_slat30k_inference.log" 2>&1 &
  current_pid=$!
  echo "started SS30K+SLat30K inference pid=${current_pid} physical_gpu=${CURRENT_GPU}"
fi

if manifest_complete "${RECON_MANIFEST}" "${RECON_FORMAT}" "${RECON_METHOD}"; then
  echo "reuse complete ReconViaGen inference: ${RECON_MANIFEST}"
else
  (
    CUDA_VISIBLE_DEVICES="${RECON_GPU}" "${PYTHON}" -u -m \
      pose_point_depth_mv.infer_omni_real_reconviagen \
      --runtime_input_manifest "${POSE_RUNTIME}" \
      --output_dir "${RECON_OUT}" \
      --seeds 42 --device cuda --low_vram
  ) >"${LOG_ROOT}/02_reconviagen_inference.log" 2>&1 &
  recon_pid=$!
  echo "started ReconViaGen inference pid=${recon_pid} physical_gpu=${RECON_GPU}"
fi

inference_rc=0
if [[ -n "${current_pid}" ]]; then
  wait "${current_pid}" || inference_rc=$?
  if (( inference_rc != 0 )); then
    echo "ERROR: SS30K+SLat30K inference failed rc=${inference_rc}" >&2
  fi
fi
if [[ -n "${recon_pid}" ]]; then
  recon_rc=0
  wait "${recon_pid}" || recon_rc=$?
  if (( recon_rc != 0 )); then
    echo "ERROR: ReconViaGen inference failed rc=${recon_rc}" >&2
    if (( inference_rc == 0 )); then
      inference_rc=${recon_rc}
    fi
  fi
fi
if (( inference_rc != 0 )); then
  exit "${inference_rc}"
fi

manifest_complete "${CURRENT_MANIFEST}" "${CURRENT_FORMAT}" "${CURRENT_METHOD}"
manifest_complete "${RECON_MANIFEST}" "${RECON_FORMAT}" "${RECON_METHOD}"

"${PYTHON}" -u -m \
  pose_point_depth_mv.evaluate_omni_pose_mask_ss30k_slat30k_vs_reconviagen_shape \
  prepare \
  --pose_mask_runtime_manifest "${POSE_RUNTIME}" \
  --pose_mask_model_input_manifest "${POSE_MODEL_INPUT}" \
  --label_manifest "${LABEL}" \
  --current_manifest "${CURRENT_MANIFEST}" \
  --reconviagen_manifest "${RECON_MANIFEST}" \
  --output_dir "${SHAPE_OUT}" \
  --expected_objects 64 --seed 42 --worker_count "${SHAPE_WORKERS}" \
  --candidate_samples 2000 --alignment_samples 10000 \
  --candidate_iterations 12 --final_iterations 50 \
  --surface_samples 20000 --bootstrap_samples 10000 --metric_seed 20260819 \
  >"${LOG_ROOT}/03_shape_prepare.log" 2>&1

worker_pids=()
for ((worker_id=0; worker_id<SHAPE_WORKERS; worker_id++)); do
  (
    "${PYTHON}" -u -m \
      pose_point_depth_mv.evaluate_omni_pose_mask_ss30k_slat30k_vs_reconviagen_shape \
      worker --output_dir "${SHAPE_OUT}" --worker_id "${worker_id}" \
      --worker_count "${SHAPE_WORKERS}" --resume
  ) >"${LOG_ROOT}/04_shape_worker_${worker_id}.log" 2>&1 &
  worker_pids+=("$!")
done

worker_rc=0
for pid in "${worker_pids[@]}"; do
  rc=0
  wait "${pid}" || rc=$?
  if (( rc != 0 )); then
    echo "ERROR: shape worker pid=${pid} failed rc=${rc}" >&2
    worker_rc=${rc}
  fi
done
if (( worker_rc != 0 )); then
  exit "${worker_rc}"
fi

"${PYTHON}" -u -m \
  pose_point_depth_mv.evaluate_omni_pose_mask_ss30k_slat30k_vs_reconviagen_shape \
  merge --output_dir "${SHAPE_OUT}" \
  2>&1 | tee "${LOG_ROOT}/05_shape_merge.log"

echo "OMNI POSE-MASK SS30K+SLAT30K VS RECON SHAPE COMPLETE"
echo "report: ${SHAPE_OUT}/report.json"
echo "summary: ${SHAPE_OUT}/summary.txt"
