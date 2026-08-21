#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:?GPU is required}
OUTPUT=${OUTPUT:?OUTPUT is required}

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

REPLAY=${PROJECT_ROOT}/pose_point_depth_mv/outputs/可视AR/OmniHoldout64复杂样本真实采集流程回放_20260811_v1
RUNTIME=${REPLAY}/PointMask_64候选筛8_v1/02_runtime_o_point_mask/runtime_input_manifest.json
MODEL=${OUTPUT}/00_冻结8视图_native_VGGT_DINO输入
OURS=${OUTPUT}/01_VSS2k_VSLat15k原生输出
RECON=${OUTPUT}/02_ReconViaGen原版输出
CONTOUR=${OUTPUT}/VSS2k_VSLat15k_相机轮廓
SS_REPORT=/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1/dev48_VSS_step2000_seed424344_2gpu03_manual_v3/aggregate_v1/report.json
V_CHECKPOINT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1/checkpoints/step_015000.pt
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

for path in "${RUNTIME}" "${SS_REPORT}" "${V_CHECKPOINT}" "${STOCK_FREEZE}"; do test -s "${path}"; done
mkdir -p "${OUTPUT}"

if [ ! -s "${MODEL}/model_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${MODEL}" ]; then RESUME+=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs_frozen_v2 \
    --runtime_input_manifest "${RUNTIME}" \
    --output_dir "${MODEL}" \
    --device cuda \
    "${RESUME[@]}"
fi

if [ ! -s "${OURS}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_official_with_vggt \
    --model_input_manifest "${MODEL}/model_input_manifest.json" \
    --native_ss_report "${SS_REPORT}" \
    --native_slat_checkpoint "${V_CHECKPOINT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${OURS}" \
    --seeds 42 --device cuda --amp_dtype bf16
fi

if [ ! -s "${RECON}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_omni_real_reconviagen_frozen_v2 \
    --runtime_input_manifest "${RUNTIME}" \
    --output_dir "${RECON}" \
    --seeds 42 --device cuda --low_vram
fi

OURS_MESH=$("${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1
print(p["objects"][0]["mesh"])
' "${OURS}/inference_manifest.json")
OURS_RESULT=$("${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1
print(p["objects"][0]["result"])
' "${OURS}/inference_manifest.json")

if [ ! -s "${CONTOUR}/report.json" ]; then
  test ! -e "${CONTOUR}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
    --runtime_input_manifest "${RUNTIME}" \
    --mesh_o "${OURS_MESH}" \
    --mesh_frame_report "${OURS_RESULT}" \
    --output_dir "${CONTOUR}" \
    --object plant:plant_012 \
    --contour_width 3
fi

"${PYTHON}" -u -m pose_point_depth_mv.finalize_with_vggt_qualitative_outputs omni \
  --output_dir "${OUTPUT}" \
  --runtime_input_manifest "${RUNTIME}" \
  --ours_manifest "${OURS}/inference_manifest.json" \
  --reconviagen_manifest "${RECON}/inference_manifest.json" \
  --contour_report "${CONTOUR}/report.json"

echo "OMNI QUALITATIVE REPLAY COMPLETE: ${OUTPUT}"
