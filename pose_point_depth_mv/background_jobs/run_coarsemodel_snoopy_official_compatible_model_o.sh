#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
ROOT=${ROOT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/CoarseModel_snoopy_指定8帧_COLMAPPose_ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v1}
EVAL_GPU=${EVAL_GPU:-4}

cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn} SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OBJECT=${ROOT}/objects/snoopy
SOURCE_RUNTIME=${OBJECT}/02_runtime_o/runtime_input_manifest.json
SOURCE_MODEL=${OBJECT}/03_dino_only_input/model_input_manifest.json
RUNTIME=${OBJECT}/08_official_compatible_model_o_runtime
MODEL=${OBJECT}/09_official_compatible_model_o_dino_input
INFER=${OBJECT}/10_official_compatible_model_o_ss30k_slat30k
CONTOUR=${OBJECT}/11_official_compatible_model_o_camera_contours
SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
BRIDGE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

for path in "${PYTHON}" "${SOURCE_RUNTIME}" "${SOURCE_MODEL}" "${SS_REPORT}" "${SLAT}" "${BRIDGE}" "${FREEZE}"; do
  test -s "${path}"
done

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_official_compatible_model_o \
  build-runtime --source_manifest "${SOURCE_RUNTIME}" --output_dir "${RUNTIME}"
"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_official_compatible_model_o \
  build-model-input --source_manifest "${SOURCE_MODEL}" \
  --runtime_manifest "${RUNTIME}/runtime_input_manifest.json" --output_dir "${MODEL}"

if [[ ! -s "${INFER}/inference_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat \
    --model_input_manifest "${MODEL}/model_input_manifest.json" \
    --native_ss_report "${SS_REPORT}" --native_slat_checkpoint "${SLAT}" \
    --expected_slat_step 30000 --cross_deployment_bridge_report "${BRIDGE}" \
    --stock_slat_freeze "${FREEZE}" --output_dir "${INFER}" \
    --seeds 42 --weights ema --device cuda --amp_dtype bf16
fi

if [[ ! -s "${CONTOUR}/report.json" ]]; then
  readarray -t VALUES < <("${PYTHON}" -c '
import json,sys
r=json.load(open(sys.argv[1])); x=r["objects"][0]
print(x["mesh"]); print(x["result"]); print(x["object_key"])
' "${INFER}/inference_manifest.json")
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -u -m \
    pose_point_depth_mv.render_runtime_o_mesh_camera_contours \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --mesh_o "${VALUES[0]}" --mesh_frame_report "${VALUES[1]}" \
    --output_dir "${CONTOUR}" --object "${VALUES[2]}" --contour_width 3 \
    --method_label "SS30K+SLat30K (official-compatible Z-up model-O)" \
    --overview_name "snoopy_SS30K_SLat30K_official兼容modelO_轮廓总览.png"
fi

"${PYTHON}" -u -m pose_point_depth_mv.coarsemodel_snoopy_official_compatible_model_o \
  finalize --root "${ROOT}"
echo "SNOOPY OFFICIAL-COMPATIBLE MODEL-O COMPLETE: ${ROOT}"
