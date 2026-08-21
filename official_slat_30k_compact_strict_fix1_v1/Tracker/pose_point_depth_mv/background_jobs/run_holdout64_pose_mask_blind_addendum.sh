#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${HOLDOUT64_POSE_MASK_GPU:-4}
ROOT=/data/zjr/omni_real_video500_download_20260804_v2
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
PROTOCOL=/home/zjr/Tracker/pose_point_depth_mv/protocols/Holdout64_PoseMask盲态追加协议_20260810.json
SPLIT=${ROOT}/D6_novel500_dev64_holdout64_v3_pilotfree_eval/holdout.json
RAW=${ROOT}/M11B_holdout64_raw_cache_v1/raw_cache_report.json
REFERENCE=${ROOT}/M11C_holdout64_runtime_o_v1/runtime_input_manifest.json
LABEL=${ROOT}/M11E_holdout64_mesh_o_labels_v1/runtime_o_label_manifest.json
POINT=${ROOT}/M11H_holdout64_native_no_vggt_mixed1244_seed42_v1/inference_manifest.json
REAL_FULL=${ROOT}/M11I_holdout64_native_v2_realadapt_step1000_seed42_v1/inference_manifest.json
SYNTH_FULL=${ROOT}/M11J_holdout64_native_v2_parent_seed42_v1/inference_manifest.json
RECON=${ROOT}/M11K_holdout64_reconviagen_original_seed42_v1/inference_manifest.json
PIXAL=${ROOT}/M11L_holdout64_pixal3d_official_seed42_v1/inference_manifest.json
POSE_RUNTIME=${ROOT}/M11N_holdout64_pose_mask_runtime_o_blind_v1
POSE_MODEL=${ROOT}/M11O_holdout64_pose_mask_dino_only_blind_v1
POSE_INFER=${ROOT}/M11P_holdout64_pose_mask_no_vggt_mixed1244_seed42_blind_v1
POSE_REBASED=${ROOT}/M11Q_holdout64_pose_mask_reference_o_seed42_blind_v1
JOINT=${ROOT}/M11R_holdout64_sixway_pose_mask_blind_addendum_v1
SS=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
JOB=Holdout64_PoseMask_blind_addendum
STATE=${RUN}/logs/${JOB}.state
EXIT_CODE=${RUN}/logs/${JOB}.exit_code
LOCK=${RUN}/logs/${JOB}.lock

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "M11N-R refused: another blind-addendum job holds ${LOCK}" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "Holdout64 PoseMask blind addendum finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s metrics_blinded=true\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

"${PY}" -m pose_point_depth_mv.freeze_holdout64_pose_mask_blind_protocol \
  verify --contract "${PROTOCOL}"

# Validate completeness and exact 64xseed42 coverage without reading any metric report.
"${PY}" - "${POINT}" "${REAL_FULL}" "${SYNTH_FULL}" "${RECON}" "${PIXAL}" <<'PY'
import json
import sys
from pathlib import Path

expected_methods = (
    "native_no_vggt_mixed",
    "native_v2_full",
    "native_v2_full",
    "reconviagen_original",
    "pixal3d_official_single_reference_view",
)
for value, expected_method in zip(sys.argv[1:], expected_methods):
    path = Path(value).resolve()
    manifest = json.load(open(path, encoding="utf-8"))
    rows = list(manifest.get("objects", []))
    pairs = {(str(row.get("object_key")), int(row.get("seed", -1))) for row in rows}
    assert manifest.get("passed") is True
    assert manifest.get("target_or_metric_consumed") is False
    assert manifest.get("seeds") == [42]
    assert manifest.get("object_count") == 64
    assert manifest.get("record_count") == 64
    assert len(rows) == 64 and len(pairs) == 64
    assert {seed for _, seed in pairs} == {42}
    assert all(row.get("passed") is True for row in rows)
    assert all(row.get("method") == expected_method for row in rows)
print({"passed": True, "existing_inference_manifests": 5, "metrics_read": False})
PY

echo "[M11N] 构建正式64例Pose+Mask runtime-O（CPU，不读取P_W/GT/旧Mesh/指标）"
if [ ! -s "${POSE_RUNTIME}/runtime_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${POSE_RUNTIME}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_pose_mask_runtime_inputs \
    --raw_cache_report "${RAW}" \
    --reference_runtime_manifest "${REFERENCE}" \
    --frozen_split_manifest "${SPLIT}" \
    --output_dir "${POSE_RUNTIME}" \
    --protocol_scope formal_holdout64_blind_addendum \
    --subset_count 64 --subset_offset 0 --selected_view_count 8 \
    "${RESUME[@]}"
fi

echo "[M11O] 编码同一64例DINO-only条件（GPU，无VGGT）"
if [ ! -s "${POSE_MODEL}/model_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${POSE_MODEL}" ]; then RESUME=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${POSE_RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${POSE_MODEL}" --device cuda \
    "${RESUME[@]}"
fi

echo "[M11P] 用冻结mixed no-VGGT SS/SLat EMA执行seed42推理（GPU）"
if [ ! -s "${POSE_INFER}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed \
    --model_input_manifest "${POSE_MODEL}/model_input_manifest.json" \
    --native_ss_checkpoint "${SS}" --native_slat_checkpoint "${SLAT}" \
    --ss_migration_contract "${SS_CONTRACT}" \
    --slat_migration_contract "${SLAT_CONTRACT}" \
    --stock_slat_freeze "${FREEZE}" --output_dir "${POSE_INFER}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi

echo "[M11Q] O_posemask -> W -> O_reference（CPU，无GT拟合）"
if [ ! -s "${POSE_REBASED}/inference_manifest.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.rebase_pose_mask_inference_to_reference_o \
    --pose_mask_inference_manifest "${POSE_INFER}/inference_manifest.json" \
    --pose_mask_runtime_manifest "${POSE_RUNTIME}/runtime_input_manifest.json" \
    --reference_runtime_manifest "${REFERENCE}" \
    --protocol_scope formal_holdout64_blind_addendum \
    --expected_objects 64 --output_dir "${POSE_REBASED}"
fi

echo "[M11R] 一次性六路统一20k采样正式评测（CPU，日志不打印指标）"
if [ ! -s "${JOINT}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_holdout64_pose_mask_blind_addendum \
    --blind_protocol_contract "${PROTOCOL}" \
    --frozen_split_manifest "${SPLIT}" \
    --label_manifest "${LABEL}" \
    --reference_runtime_manifest "${REFERENCE}" \
    --point_mask_manifest "${POINT}" \
    --pose_mask_rebased_manifest "${POSE_REBASED}/inference_manifest.json" \
    --real_full_manifest "${REAL_FULL}" \
    --synthetic_full_manifest "${SYNTH_FULL}" \
    --reconviagen_manifest "${RECON}" \
    --pixal3d_manifest "${PIXAL}" \
    --output_dir "${JOINT}" --surface_samples 20000
fi

"${PY}" - "${JOINT}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True and report["formal"] is True
assert report["object_count"] == 64 and report["record_count"] == 384
assert report["passed_semantics"] == "protocol and mesh completeness only; not method victory"
print({
    "formal_protocol_passed": True,
    "objects": 64,
    "records": 384,
    "metrics_printed": False,
    "next": "run the separate U1 unblind command",
})
PY
