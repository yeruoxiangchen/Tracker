#!/usr/bin/env bash
# Plant Omni Holdout: 64 frozen RGB/mask/pose candidates -> current selector -> 8 model views.
#
# This is an offline invocation of the current real-capture reconstruction backend.
# It does not start collect_ar_object_server.py and does not rewrite source pose,
# masks, or the frozen COLMAP point cloud.

set -Eeuo pipefail

trap 'rc=$?; echo "[FAILED] line=${LINENO} rc=${rc}" >&2; exit "${rc}"' ERR

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${GPU:-6}

INPUT_ROOT=/home/zjr/Tracker/pose_point_depth_mv/outputs/可视AR/OmniHoldout64复杂样本真实采集流程回放_20260811_v1/PointMask_64候选筛8_v1
RAW64=${INPUT_ROOT}/01_raw64/raw_cache_report.json

OUT=/home/zjr/Tracker/pose_point_depth_mv/outputs/可视AR/OmniHoldout64复杂样本真实采集流程回放_20260811_v1/PointMask_64原始输入_当前真实流程_v1
RUNTIME=${OUT}/02_runtime_o
DINO=${OUT}/03_dino_only_input8
INFER=${OUT}/04_no_vggt_inference
BUNDLE=${OUT}/05_world_mesh_bundle

RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_CKPT=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT_CKPT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

for required in \
  "${RAW64}" \
  "${SS_CKPT}" \
  "${SLAT_CKPT}" \
  "${SS_CONTRACT}" \
  "${SLAT_CONTRACT}" \
  "${STOCK_FREEZE}"
do
  if [[ ! -s "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 66
  fi
done

echo "[0/5] Audit frozen 64-view input; no files are modified"
"${PY}" -c 'import json,sys,numpy as np; r=json.load(open(sys.argv[1])); x=r["objects"][0]; z=np.load(x["cache_npz"]); source=np.load(x["candidate_subset"]["source_raw_cache"]); names=[str(v) for v in z["frame_name"].tolist()]; source_names=[str(v) for v in source["frame_name"].tolist()]; indices=np.asarray([source_names.index(v) for v in names]); assert x["registered_pair_count"]==64; assert len(names)==len(z["T_W2C"])==64; assert len(z["P_W"])>0; assert np.array_equal(z["P_W"],source["P_W"]); assert np.array_equal(z["T_W2C"],source["T_W2C"][indices]); print({"candidate_views":64,"original_sparse_points":len(z["P_W"]),"pose_exact_subset":True,"point_cloud_exact":True,"mask_paths_reused":True})' "${RAW64}"

echo "[1/5] Current point-mask runtime: 64 candidates -> azimuth-balanced 8"
"${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs \
  --raw_cache_report "${RAW64}" \
  --output_dir "${RUNTIME}" \
  --object plant:plant_012 \
  --selected_view_count 8 \
  --view_selection_policy object_azimuth_balanced_valid_mask \
  --geometry_mode point_mask \
  --min_object_points 100 \
  --min_mask_observations 2 \
  --min_mask_support_ratio 0.35 \
  --gravity_up_w 0 1 0 \
  --min_completed_objects 1 \
  --resume

echo "[2/5] Audit selected views"
"${PY}" -c 'import json,sys; r=json.load(open(sys.argv[1])); x=r["objects"][0]; assert x["selected_view_count"]==8; assert len(x["selected_frame_names"])==8; assert x["view_selection"]["policy"]=="object_azimuth_balanced_valid_mask"; print({"candidate_views":64,"model_views":8,"selected_frames":x["selected_frame_names"],"point_mask_points":x["runtime_frame_stats"]["support"]["mask_supported_point_count"]})' "${RUNTIME}/runtime_input_manifest.json"

echo "[3/5] DINO-only encode of the selected 8 views on physical GPU ${GPU}"
env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${DINO}" \
    --object plant:plant_012 \
    --device cuda \
    --resume

"${PY}" -c 'import json,sys; r=json.load(open(sys.argv[1])); x=r["objects"][0]; assert x["encoder_stats"]["view_count"]==8; assert x["encoder_stats"]["visual_shape"][0]==8; print({"DINO_model_views":x["encoder_stats"]["view_count"],"visual_shape":x["encoder_stats"]["visual_shape"]})' "${DINO}/model_input_manifest.json"

echo "[4/5] Mixed no-VGGT SS + SLat inference on physical GPU ${GPU}"
if [[ -s "${INFER}/inference_manifest.json" ]]; then
  echo "reuse completed inference: ${INFER}/inference_manifest.json"
else
  env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn \
    SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed \
      --model_input_manifest "${DINO}/model_input_manifest.json" \
      --native_ss_checkpoint "${SS_CKPT}" \
      --native_slat_checkpoint "${SLAT_CKPT}" \
      --ss_migration_contract "${SS_CONTRACT}" \
      --slat_migration_contract "${SLAT_CONTRACT}" \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --output_dir "${INFER}" \
      --seeds 42 \
      --weights ema \
      --amp_dtype bf16 \
      --device cuda
fi

echo "[5/5] Package runtime-O and sparse-world Mesh"
if [[ -s "${BUNDLE}/report.json" ]]; then
  echo "reuse completed bundle: ${BUNDLE}/report.json"
else
  "${PY}" -u -m pose_point_depth_mv.package_coarsemodel_real_no_vggt_results \
    --inference_manifest "${INFER}/inference_manifest.json" \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --raw_cache_report "${RAW64}" \
    --output_dir "${BUNDLE}"
fi

"${PY}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] is True; c=r["cases"][0]; print({"passed":True,"runtime_O_mesh":c["predicted_runtime_o"]["path"],"world_mesh":c["predicted_sparse_world"]["path"],"report":sys.argv[1]})' "${BUNDLE}/report.json"

echo "complete: ${BUNDLE}/report.json"
