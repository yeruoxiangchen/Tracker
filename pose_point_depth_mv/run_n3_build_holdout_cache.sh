#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

SOURCE=/data/reconvggt_pointpose_v9_ssfixed_odsplit_20260712/cache/holdout/manifest.json
CACHE=/data/ar_ss_flow_pose_lifting_holdout48_v1_20260718
INDICES=0,2,3,5,7,9,10,12,14,16,18,20,21,23,24,26,28,30,31,32,33,34,35,36,37,39,41,42,43,44,45,47,48,49,51,53,54,56,58,59,61,62,63,65,67,68,69,70
GPU=${GPU:-1}
OVERALL=0

if [ -f "${CACHE}/manifest.json" ]; then
  echo "reuse holdout cache: ${CACHE}/manifest.json"
  BUILD_CODE=0
elif [ -e "${CACHE}" ]; then
  echo "incomplete holdout cache exists: ${CACHE}"
  BUILD_CODE=98
else
  CUDA_VISIBLE_DEVICES=${GPU} \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u \
    ar_ss_flow/build_pose_lifting_cache.py \
    --source_cache_manifest "${SOURCE}" \
    --output_dir "${CACHE}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --vggt_pretrained Stable-X/vggt-object-v0-1 \
    --indices "${INDICES}" \
    --max_samples 0 \
    --device cuda \
    --image_resolution 518 \
    --vggt_feature_index -1 \
    --min_depth_matches 8 \
    --affine_improvement_ratio 0.90 \
    --save_correct_geometry \
    --log_every 1 \
    2>&1 | tee "${CACHE}.build.log"
  BUILD_CODE=${PIPESTATUS[0]}
fi
echo "${BUILD_CODE}" > "${CACHE}.build.exit_code"
if [ "${BUILD_CODE}" -ne 0 ]; then
  OVERALL=${BUILD_CODE}
fi

if [ "${BUILD_CODE}" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=${GPU} \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u \
    ar_ss_flow/audit_pose_lifting_cache.py \
    --cache_manifest "${CACHE}/manifest.json" \
    --output_dir "${CACHE}/independent_audit" \
    --indices all \
    --max_samples 0 \
    --device cuda \
    --min_depth_enabled_ratio 0.80 \
    --max_cached_geometry_diff 2.0e-3 \
    --max_roundtrip_error 1.0e-4 \
    --fail_on_error \
    2>&1 | tee "${CACHE}/independent_audit.log"
  GEOMETRY_CODE=${PIPESTATUS[0]}
else
  GEOMETRY_CODE=99
fi
echo "${GEOMETRY_CODE}" > "${CACHE}.geometry_audit.exit_code"
if [ "${GEOMETRY_CODE}" -ne 0 ]; then
  OVERALL=${GEOMETRY_CODE}
fi

if [ "${GEOMETRY_CODE}" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=${GPU} \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u \
    ar_ss_flow/audit_cached_stock_condition.py \
    --cache_manifest "${CACHE}/manifest.json" \
    --output_dir "${CACHE}/stock_condition_audit_2_4_8" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --indices 0,5,2 \
    --device cuda \
    --max_abs_tolerance 0.0 \
    --rms_tolerance 0.0 \
    --min_cosine 1.0 \
    --require_fp16_equal \
    --fail_on_error \
    2>&1 | tee "${CACHE}/stock_condition_audit_2_4_8.log"
  STOCK_CODE=${PIPESTATUS[0]}
else
  STOCK_CODE=99
fi
echo "${STOCK_CODE}" > "${CACHE}.stock_audit.exit_code"
if [ "${STOCK_CODE}" -ne 0 ]; then
  OVERALL=${STOCK_CODE}
fi

echo "${OVERALL}" > "${CACHE}.runner.status"
echo "N3 holdout cache complete: status=${OVERALL}"
exit 0
