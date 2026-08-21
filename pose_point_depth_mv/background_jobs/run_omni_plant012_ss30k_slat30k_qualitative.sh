#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${GPU:?GPU is required}
OUTPUT=${OUTPUT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/OmniPlant012冻结8视图_ReconViaGen_vs_SS30K_SLat30K_相机轮廓_20260819_v1}

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

REPLAY=${PROJECT_ROOT}/pose_point_depth_mv/outputs/可视AR/OmniHoldout64复杂样本真实采集流程回放_20260811_v1
RUNTIME=${REPLAY}/PointMask_64候选筛8_v1/02_runtime_o_point_mask/runtime_input_manifest.json
DINO_SOURCE=${REPLAY}/PointMask_64候选筛8_v1/03_dino_only_input8
OLD_OUTPUT=${PROJECT_ROOT}/pose_point_depth_mv/outputs2/OmniPlant012冻结8视图_ReconViaGen_vs_VSS2k_VSLat15k_相机轮廓_20260818_v1
RECON_SOURCE=${OLD_OUTPUT}/02_ReconViaGen原版输出

SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT_CHECKPOINT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
ABC_R_EVIDENCE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

CURRENT=${OUTPUT}/01_SS30K_SLat30K原生输出
CONTOUR=${OUTPUT}/SS30K_SLat30K_相机轮廓
LOG=${OUTPUT}/run.log

for path in \
  "${RUNTIME}" \
  "${DINO_SOURCE}/model_input_manifest.json" \
  "${RECON_SOURCE}/inference_manifest.json" \
  "${SS_REPORT}" \
  "${SLAT_CHECKPOINT}" \
  "${ABC_R_EVIDENCE}" \
  "${STOCK_FREEZE}"
do
  test -s "${path}"
done

if [[ ! -s "${OUTPUT}/prepare_identity.json" ]]; then
  test ! -e "${OUTPUT}"
  "${PYTHON}" -u -m pose_point_depth_mv.finalize_omni_ss30k_slat30k_qualitative \
    prepare \
    --output_dir "${OUTPUT}" \
    --runtime_input_manifest "${RUNTIME}" \
    --dino_input_dir "${DINO_SOURCE}" \
    --reconviagen_source_dir "${RECON_SOURCE}"
fi

exec > >(tee -a "${LOG}") 2>&1

if [[ ! -s "${CURRENT}/inference_manifest.json" ]]; then
  echo "===== no-VGGT SS30K + no-VGGT SLat30K seed42 ====="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
    --model_input_manifest "${OUTPUT}/00_冻结8视图_DINO-only输入/model_input_manifest.json" \
    --native_ss_report "${SS_REPORT}" \
    --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
    --expected_slat_step 30000 \
    --cross_deployment_bridge_report "${ABC_R_EVIDENCE}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --output_dir "${CURRENT}" \
    --seeds 42 --weights ema --device cuda --amp_dtype bf16 \
    --object plant:plant_012
fi

CURRENT_MESH=$("${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1
print(p["objects"][0]["mesh"])
' "${CURRENT}/inference_manifest.json")
CURRENT_RESULT=$("${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["passed"] and len(p["objects"])==1
print(p["objects"][0]["result"])
' "${CURRENT}/inference_manifest.json")

if [[ ! -s "${CONTOUR}/report.json" ]]; then
  test ! -e "${CONTOUR}"
  echo "===== physical runtime-O camera contours ====="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
    --runtime_input_manifest "${RUNTIME}" \
    --mesh_o "${CURRENT_MESH}" \
    --mesh_frame_report "${CURRENT_RESULT}" \
    --output_dir "${CONTOUR}" \
    --object plant:plant_012 \
    --contour_width 3 \
    --method_label "no-VGGT SS30K+SLat30K" \
    --overview_name "SS30K_SLat30K_原始8帧相机位姿轮廓总览.png"
fi

echo "===== final cross-artifact audit ====="
"${PYTHON}" -u -m pose_point_depth_mv.finalize_omni_ss30k_slat30k_qualitative \
  finalize \
  --output_dir "${OUTPUT}" \
  --current_manifest "${CURRENT}/inference_manifest.json" \
  --contour_report "${CONTOUR}/report.json" \
  --native_ss_report "${SS_REPORT}" \
  --native_slat_checkpoint "${SLAT_CHECKPOINT}" \
  --bridge_report "${ABC_R_EVIDENCE}" \
  --stock_slat_freeze "${STOCK_FREEZE}"

echo "============================================================"
echo "OMNI PLANT012 no-VGGT SS30K+SLAT30K COMPLETE"
echo "OUTPUT=${OUTPUT}"
echo "============================================================"
