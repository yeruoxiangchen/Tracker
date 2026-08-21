#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${COARSEMODEL_POSE_MASK_GPU:-4}
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

OLD3=/data/zjr/coarsemodel_real_no_vggt_test_20260810_v1
EXP3=/data/zjr/coarsemodel_real_no_vggt_pose_mask_20260810_v1
RAW3=${OLD3}/raw_cache_v1/raw_cache_report.json
REF3=${OLD3}/runtime_o_min32_v1/runtime_input_manifest.json
RUNTIME3=${EXP3}/runtime_pose_mask_8view_v1
MODEL3=${EXP3}/dino_pose_mask_8view_v1
INFER3=${EXP3}/native_no_vggt_pose_mask_seed42_v1
REBASE3=${EXP3}/rebased_to_point_mask_reference_o_v1
VIS3=/home/zjr/Tracker/pose_point_depth_mv/outputs/可视化Mesh/CoarseModel真实采集_NoVGGT_PoseMask无点云测试_20260810_v1

OLD2=/data/zjr/coarsemodel_real_allviews_threeway_20260810_v1
EXP2=/data/zjr/coarsemodel_real_allviews_pose_mask_20260810_v1
RAW2=${OLD2}/raw_cache_allviews_v1/raw_cache_report.json
REFA=${OLD2}/runtime_0512_all18_v1/runtime_input_manifest.json
REFB=${OLD2}/runtime_0513_all16_v1/runtime_input_manifest.json
RUNTIMEA=${EXP2}/runtime_pose_mask_0512_all18_v1
RUNTIMEB=${EXP2}/runtime_pose_mask_0513_all16_v1
MODELA=${EXP2}/dino_pose_mask_0512_all18_v1
MODELB=${EXP2}/dino_pose_mask_0513_all16_v1
INFERA=${EXP2}/current_pose_mask_0512_all18_seed42_v1
INFERB=${EXP2}/current_pose_mask_0513_all16_seed42_v1
REBASEA=${EXP2}/rebased_0512_to_point_mask_reference_o_v1
REBASEB=${EXP2}/rebased_0513_to_point_mask_reference_o_v1
PIXALA=${OLD2}/pixal_0512_seed42_v1/inference_manifest.json
PIXALB=${OLD2}/pixal_0513_seed42_v1/inference_manifest.json
VIS2=/home/zjr/Tracker/pose_point_depth_mv/outputs/可视化Mesh/CoarseModel两组真实采集_PoseMask无点云_全视图三模型视频对比_显示归一化_20260810_v1

LOG_ROOT=${EXP2}/logs
STATE=${LOG_ROOT}/coarsemodel_pose_mask_no_point.state
EXIT_CODE=${LOG_ROOT}/coarsemodel_pose_mask_no_point.exit_code
LOCK=${LOG_ROOT}/coarsemodel_pose_mask_no_point.lock
mkdir -p "${LOG_ROOT}" "${EXP3}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "CoarseModel PoseMask no-point job refused: lock is held: ${LOCK}" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "CoarseModel PoseMask no-point background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${SS}" "${SLAT}" "${SS_CONTRACT}" "${SLAT_CONTRACT}" \
                "${FREEZE}" "${RAW3}" "${REF3}" "${RAW2}" "${REFA}" \
                "${REFB}" "${PIXALA}" "${PIXALB}"; do
  test -s "${REQUIRED}"
done

build_runtime() {
  local RAW=$1 REFERENCE=$2 OUTPUT=$3 COUNT=$4 VIEWS=$5
  if [ ! -s "${OUTPUT}/runtime_input_manifest.json" ]; then
    RESUME=()
    if [ -e "${OUTPUT}" ]; then RESUME=(--resume); fi
    "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_pose_mask_runtime_inputs \
      --raw_cache_report "${RAW}" \
      --reference_runtime_manifest "${REFERENCE}" \
      --output_dir "${OUTPUT}" \
      --protocol_scope coarsemodel_real_qualitative \
      --subset_count "${COUNT}" --subset_offset 0 --subset_seed 20260810 \
      --selected_view_count "${VIEWS}" \
      "${RESUME[@]}"
  fi
}

build_dino() {
  local RUNTIME=$1 OUTPUT=$2
  if [ ! -s "${OUTPUT}/model_input_manifest.json" ]; then
    RESUME=()
    if [ -e "${OUTPUT}" ]; then RESUME=(--resume); fi
    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
      --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
      --output_dir "${OUTPUT}" --device cuda \
      "${RESUME[@]}"
  fi
}

infer_pose_mask() {
  local MODEL=$1 OUTPUT=$2
  if [ ! -s "${OUTPUT}/inference_manifest.json" ]; then
    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed \
      --model_input_manifest "${MODEL}/model_input_manifest.json" \
      --native_ss_checkpoint "${SS}" \
      --native_slat_checkpoint "${SLAT}" \
      --ss_migration_contract "${SS_CONTRACT}" \
      --slat_migration_contract "${SLAT_CONTRACT}" \
      --stock_slat_freeze "${FREEZE}" \
      --output_dir "${OUTPUT}" \
      --seeds 42 --weights ema --amp_dtype bf16 --device cuda
  fi
}

rebase_to_reference() {
  local INFERENCE=$1 RUNTIME=$2 REFERENCE=$3 OUTPUT=$4
  if [ ! -s "${OUTPUT}/inference_manifest.json" ]; then
    "${PY}" -u -m pose_point_depth_mv.rebase_pose_mask_inference_to_reference_o \
      --pose_mask_inference_manifest "${INFERENCE}/inference_manifest.json" \
      --pose_mask_runtime_manifest "${RUNTIME}/runtime_input_manifest.json" \
      --reference_runtime_manifest "${REFERENCE}" \
      --output_dir "${OUTPUT}"
  fi
}

echo "[N1] 三例8视图：构建Pose+Mask runtime-O（CPU，明确不读取P_W）"
build_runtime "${RAW3}" "${REF3}" "${RUNTIME3}" 3 8

echo "[N2] 三例8视图：DINO-only编码与No-VGGT推理（GPU）"
build_dino "${RUNTIME3}" "${MODEL3}"
infer_pose_mask "${MODEL3}" "${INFER3}"

echo "[N3] 三例8视图：导出Pose+Mask世界Mesh和旧reference-O对照Mesh（CPU）"
rebase_to_reference "${INFER3}" "${RUNTIME3}" "${REF3}" "${REBASE3}"
if [ ! -s "${VIS3}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.package_coarsemodel_real_no_vggt_results \
    --inference_manifest "${INFER3}/inference_manifest.json" \
    --runtime_input_manifest "${RUNTIME3}/runtime_input_manifest.json" \
    --raw_cache_report "${RAW3}" \
    --output_dir "${VIS3}"
fi

echo "[N3V] 三例8视图：生成各自Pose+Mask normal环绕视频（GPU）"
CUDA_VISIBLE_DEVICES="${GPU}" MPLCONFIGDIR=/tmp/matplotlib \
"${PY}" -u -m pose_point_depth_mv.render_coarsemodel_pose_mask_turntables \
  --bundle_report "${VIS3}/report.json" \
  --device cuda --render_frames 48 --render_resolution 512 --fps 12 \
  --display_margin 1.0 --resume

echo "[N4] 两例全视图：分别构建18-view/16-view Pose+Mask runtime-O（CPU）"
build_runtime "${RAW2}" "${REFA}" "${RUNTIMEA}" 1 18
build_runtime "${RAW2}" "${REFB}" "${RUNTIMEB}" 1 16

echo "[N5] 两例全视图：DINO-only编码与No-VGGT推理（GPU）"
build_dino "${RUNTIMEA}" "${MODELA}"
build_dino "${RUNTIMEB}" "${MODELB}"
infer_pose_mask "${MODELA}" "${INFERA}"
infer_pose_mask "${MODELB}" "${INFERB}"

echo "[N6] 两例全视图：映射回旧Point+Mask reference-O（CPU，无GT拟合）"
rebase_to_reference "${INFERA}" "${RUNTIMEA}" "${REFA}" "${REBASEA}"
rebase_to_reference "${INFERB}" "${RUNTIMEB}" "${REFB}" "${REBASEB}"

echo "[N7] 两例全视图：生成Pose+Mask/ReconViaGen/Pixal3D显示归一化视频（GPU）"
if [ ! -s "${VIS2}/report.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" MPLCONFIGDIR=/tmp/matplotlib \
  "${PY}" -u -m pose_point_depth_mv.render_coarsemodel_current_recon_pixal_videos \
    --current_manifest "${INFERA}/inference_manifest.json" \
    --current_manifest "${INFERB}/inference_manifest.json" \
    --pixal_manifest "${PIXALA}" \
    --pixal_manifest "${PIXALB}" \
    --raw_cache_report "${RAW2}" \
    --output_dir "${VIS2}" \
    --device cuda --render_frames 48 --render_resolution 512 --fps 12 \
    --display_margin 1.0 --comparison_autoframe \
    --comparison_target_fill 0.72 --comparison_foreground_chroma 18 \
    --current_display_name "Pose+Mask No-Point No-VGGT" \
    --resume
fi

echo "[N8] 最终只读合同核验"
"${PY}" - "${RUNTIME3}/runtime_input_manifest.json" \
  "${RUNTIMEA}/runtime_input_manifest.json" "${RUNTIMEB}/runtime_input_manifest.json" \
  "${VIS3}/report.json" "${VIS2}/report.json" \
  "${REBASE3}/inference_manifest.json" "${REBASEA}/inference_manifest.json" \
  "${REBASEB}/inference_manifest.json" <<'PY'
import json
import sys

runtime_paths = sys.argv[1:4]
for path in runtime_paths:
    report = json.load(open(path, encoding="utf-8"))
    assert report["passed"] is True
    assert report["protocol_scope"] == "coarsemodel_real_qualitative"
    assert report["point_cloud_consumed"] is False
    assert all(row["point_cloud_fields_read"] == [] for row in report["objects"])
package = json.load(open(sys.argv[4], encoding="utf-8"))
videos = json.load(open(sys.argv[5], encoding="utf-8"))
assert package["passed"] is True and package["object_count"] == 3
assert package["turntable_review"]["passed"] is True
assert package["turntable_review"]["object_count"] == 3
assert all(row["turntable"]["passed"] is True for row in package["cases"])
assert videos["passed"] is True and videos["object_count"] == 2
for path, count in zip(sys.argv[6:], (3, 1, 1)):
    report = json.load(open(path, encoding="utf-8"))
    assert report["passed"] is True and report["object_count"] == count
    assert report["point_cloud_consumed"] is False
    assert report["coordinate_policy"].startswith("observable O_posemask->W")
print({
    "passed": True,
    "point_cloud_consumed": False,
    "world_mesh_cases": 3,
    "allview_video_cases": 2,
    "pose_mask_rebased_cases": 5,
    "package_report": sys.argv[4],
    "video_report": sys.argv[5],
})
PY
