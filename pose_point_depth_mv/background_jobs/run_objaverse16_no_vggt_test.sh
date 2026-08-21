#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${OBJAVERSE16_GPU:-4}
ROOT=${OBJAVERSE16_ROOT:-/data/zjr/objaverse16_no_vggt_mixed_20260810_v1}
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SOURCE=/data/zjr/reviewed_mixed1k_semantic_object_ss_repaired_v1_20260730/test.json
SELECTION=${ROOT}/O0_frozen_objaverse_test16_v1.json
POINT_PRIOR=${ROOT}/O1_point_prior_seed20260810_v1
POINTPOSE=${ROOT}/O2_pointpose_cache_v1
LIFT_FULL=${ROOT}/O3_lifting_full_v1
LIFT_DINO=${ROOT}/O4_lifting_dino_only_v1
MODEL_INPUT=${ROOT}/O5_model_inputs_target_free_v1
INFERENCE=${ROOT}/O6_native_no_vggt_mixed_seed42_v1
EVALUATION=${ROOT}/O7b_canonical_mesh_eval_axisfixed_20k_v2
SS=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
LOG_DIR=${ROOT}/logs
STATE=${LOG_DIR}/Objaverse16_no_vggt_test.state
EXIT_CODE=${LOG_DIR}/Objaverse16_no_vggt_test.exit_code
LOCK=${LOG_DIR}/Objaverse16_no_vggt_test.lock

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse16 refused: another job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "Objaverse16 no-VGGT test finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in \
  "${PY}" "${SOURCE}" "${SS}" "${SLAT}" "${SS_CONTRACT}" \
  "${SLAT_CONTRACT}" "${STOCK_FREEZE}"; do
  test -s "${REQUIRED}"
done

echo "[O0] 冻结16个未见Objaverse test对象，并审计训练对象不重叠"
"${PY}" -u -m pose_point_depth_mv.freeze_objaverse16_test \
  --test_manifest "${SOURCE}" \
  --output_manifest "${SELECTION}" \
  --seed 20260810 \
  --point_prior_seed 20260810 \
  --group_quotas legacy897=10,gap_objaverse288=3,pilot_objaverse217=3 \
  --prior_view_choices 2,4,8 \
  --resume

echo "[O1] CPU构建冻结的稀疏点先验；只用于重放训练式视图/O框架准备"
if [ ! -s "${POINT_PRIOR}/manifest.json" ]; then
  "${PY}" -u trellis_point_prior_mv/build_point_prior_dataset.py \
    --source_manifest "${SELECTION}" \
    --output_dir "${POINT_PRIOR}" \
    --indices all \
    --max_frames 8 \
    --seed 20260810 \
    --grid_transform pixal3d_rotation \
    --num_prior_views_choices 2,4,8 \
    --point_count_choices 50,100,300,800,1500 \
    --min_support 1 \
    --min_support_ratio 0.45 \
    --dropout_min 0.0 \
    --dropout_max 0.65 \
    --coord_jitter 1 \
    --outlier_ratio 0.03 \
    --front_depth_epsilon 0.02 \
    --log_every 1
fi

echo "[O2] CPU构建16例PointPose缓存"
if [ ! -s "${POINTPOSE}/manifest.json" ]; then
  "${PY}" -u reconvggt_ar_adapter_a/build_pointpose_ss_cache.py \
    --source_manifest "${SELECTION}" \
    --prior_manifest "${POINT_PRIOR}/manifest.json" \
    --output_dir "${POINTPOSE}" \
    --indices all \
    --log_every 1
fi

echo "[O3] GPU构建共享几何Full lifting缓存；VGGT只在该离线预处理阶段运行"
if [ ! -s "${LIFT_FULL}/manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u ar_ss_flow/build_pose_lifting_cache.py \
    --source_cache_manifest "${POINTPOSE}/manifest.json" \
    --output_dir "${LIFT_FULL}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --vggt_pretrained Stable-X/vggt-object-v0-1 \
    --indices all \
    --device cuda \
    --save_correct_geometry \
    --overwrite \
    --log_every 1
fi

echo "[O4] 派生DINO-only缓存；不复制VGGT特征、VGGT context或VGGT深度"
if [ ! -s "${LIFT_DINO}/lifting_manifest.json" ]; then
  DINO_RESUME=()
  if [ -e "${LIFT_DINO}" ]; then DINO_RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.derive_dino_only_lifting_cache \
    --source_manifest "${LIFT_FULL}/manifest.json" \
    --output_dir "${LIFT_DINO}" \
    --indices all \
    --ss_context_tokens 4096 \
    "${DINO_RESUME[@]}"
fi

echo "[O5] 封装目标隔离的DINO+Pose模型输入；点坐标不进入推理payload"
"${PY}" -u -m pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs \
  --selection_manifest "${SELECTION}" \
  --lifting_manifest "${LIFT_DINO}/lifting_manifest.json" \
  --output_dir "${MODEL_INPUT}" \
  --resume

echo "[O6] GPU运行当前mixed real+synth no-VGGT SS/SLat，seed=42"
CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.infer_objaverse16_no_vggt_mixed \
  --model_input_manifest "${MODEL_INPUT}/model_input_manifest.json" \
  --native_ss_checkpoint "${SS}" \
  --native_slat_checkpoint "${SLAT}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --ss_migration_contract "${SS_CONTRACT}" \
  --slat_migration_contract "${SLAT_CONTRACT}" \
  --output_dir "${INFERENCE}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --seeds 42 \
  --weights ema \
  --device cuda \
  --amp_dtype bf16

echo "[O7b] 固定decoder轴变换后，CPU统一20k表面采样评测canonical Objaverse GT"
"${PY}" -u -m pose_point_depth_mv.evaluate_objaverse16_no_vggt \
  --selection_manifest "${SELECTION}" \
  --lifting_manifest "${LIFT_DINO}/lifting_manifest.json" \
  --inference_manifest "${INFERENCE}/inference_manifest.json" \
  --output_dir "${EVALUATION}" \
  --surface_samples 20000 \
  --fscore_thresholds 0.01,0.02,0.05 \
  --resume

"${PY}" -c 'import json,sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["passed"] is True and r["formal"] is False
assert r["protocol_scope"] == "frozen_objaverse_test16"
assert r["object_count"] == 16 and r["record_count"] == 16
assert r["target_or_metric_consumed_during_inference"] is False
assert r["point_cloud_tensor_consumed_during_inference"] is False
print({"passed": True, "objects": 16, "report": sys.argv[1]})' \
  "${EVALUATION}/report.json"
