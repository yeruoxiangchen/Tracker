#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${POSE_MASK_ABLATION_GPU:-5}
ROOT=/data/zjr/omni_real_video500_download_20260804_v2
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
EXP=${POSE_MASK_ABLATION_EXP:-/data/zjr/native_no_vggt_pose_mask_benchmark16_20260810_v1}
STAGE=${POSE_MASK_STAGE_PREFIX:-P}
SUBSET_OFFSET=${POSE_MASK_SUBSET_OFFSET:-0}
SUBSET_COUNT=${POSE_MASK_SUBSET_COUNT:-16}
EXPECTED_TOTAL=${POSE_MASK_EXPECTED_TOTAL_OBJECTS:-${SUBSET_COUNT}}
PRIOR_REPORT=${POSE_MASK_PRIOR_ABLATION_REPORT:-}
JOB_BASENAME=${POSE_MASK_JOB_BASENAME:-P1_P5_pose_mask_ablation}
RAW=${ROOT}/D8_benchmark32_raw_cache_v1/raw_cache_report.json
REFERENCE=${ROOT}/D9_benchmark32_runtime_o_v2/runtime_input_manifest.json
LABEL=${ROOT}/D11_benchmark32_mesh_o_labels_v2/runtime_o_label_manifest.json
BASELINE_SOURCE_MODEL=${ROOT}/M9_benchmark32_dino_only_model_inputs_v1/model_input_manifest.json
FIVEWAY=${ROOT}/M9R_benchmark32_fiveway_with_records_20260809_v1/report.json
RUNTIME=${EXP}/${STAGE}1_pose_mask_runtime16_v1
BASELINE_MODEL=${EXP}/${STAGE}2A_point_mask_model_inputs16_v1/model_input_manifest.json
MODEL=${EXP}/${STAGE}2B_pose_mask_dino_only_model_inputs16_v1
BASELINE_INFER=${EXP}/${STAGE}3A_point_mask_no_vggt_seed42_v1
INFER=${EXP}/${STAGE}3B_pose_mask_no_vggt_seed42_v1
REBASED=${EXP}/${STAGE}4_pose_mask_rebased_reference_o_seed42_v1
EVAL=${EXP}/${STAGE}5_point_mask_vs_pose_mask_paired16_v1
EXTERNAL=${EXP}/${STAGE}6_pose_mask_external_bases_shared_surface_v1
SS=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
STATE=${EXP}/logs/${JOB_BASENAME}.state
EXIT_CODE=${EXP}/logs/${JOB_BASENAME}.exit_code
LOCK=${EXP}/logs/${JOB_BASENAME}.lock

mkdir -p "${EXP}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "PoseMask Benchmark16 refused: another job holds ${LOCK}" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "PoseMask Benchmark16 background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${RAW}" "${REFERENCE}" "${LABEL}" "${BASELINE_SOURCE_MODEL}" \
                "${SS}" "${SLAT}" "${SS_CONTRACT}" "${SLAT_CONTRACT}" \
                "${FREEZE}" "${FIVEWAY}"; do
  test -s "${REQUIRED}"
done
case "${SUBSET_OFFSET}:${SUBSET_COUNT}:${EXPECTED_TOTAL}" in
  *[!0-9:]*|:*|*::*) echo "subset settings must be nonnegative integers" >&2; exit 96 ;;
esac
if [ "${SUBSET_COUNT}" -le 0 ] || [ $((SUBSET_OFFSET + SUBSET_COUNT)) -gt 32 ]; then
  echo "subset rank slice must fit within Benchmark32" >&2
  exit 96
fi
if [ -n "${PRIOR_REPORT}" ]; then test -s "${PRIOR_REPORT}"; fi

echo "[${STAGE}1] 构建哈希排名 ${SUBSET_OFFSET}..$((SUBSET_OFFSET + SUBSET_COUNT - 1)) 的 pose+mask runtime-O（CPU，不读取点云）"
if [ ! -s "${RUNTIME}/runtime_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${RUNTIME}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_pose_mask_runtime_inputs \
    --raw_cache_report "${RAW}" \
    --reference_runtime_manifest "${REFERENCE}" \
    --output_dir "${RUNTIME}" \
    --subset_count "${SUBSET_COUNT}" --subset_offset "${SUBSET_OFFSET}" \
    --subset_seed 20260810 \
    --selected_view_count 8 \
    "${RESUME[@]}"
fi

echo "[${STAGE}2A] 冻结 point+mask 基线的同对象 model-input 子清单（CPU）"
if [ ! -s "${BASELINE_MODEL}" ]; then
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.subset_dino_only_model_inputs \
    --source_model_input_manifest "${BASELINE_SOURCE_MODEL}" \
    --ablation_split "${RUNTIME}/ablation_split.json" \
    --output "${BASELINE_MODEL}"
fi

echo "[${STAGE}2B] 编码同一RGB/mask的 pose+mask DINO-only 条件（GPU）"
if [ ! -s "${MODEL}/model_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${MODEL}" ]; then RESUME=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --output_dir "${MODEL}" --device cuda \
    "${RESUME[@]}"
fi

echo "[${STAGE}3A] 重放同对象 point+mask 基线，锁定相同位置噪声（GPU）"
if [ ! -s "${BASELINE_INFER}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed \
    --model_input_manifest "${BASELINE_MODEL}" \
    --native_ss_checkpoint "${SS}" \
    --native_slat_checkpoint "${SLAT}" \
    --ss_migration_contract "${SS_CONTRACT}" \
    --slat_migration_contract "${SLAT_CONTRACT}" \
    --stock_slat_freeze "${FREEZE}" \
    --output_dir "${BASELINE_INFER}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi

echo "[${STAGE}3B] 冻结同一模型推理 pose+mask，EMA、CFG5、seed42（GPU）"
if [ ! -s "${INFER}/inference_manifest.json" ]; then
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
    --output_dir "${INFER}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi

echo "[${STAGE}4] O_posemask -> W -> O_reference（CPU，无GT拟合）"
if [ ! -s "${REBASED}/inference_manifest.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.rebase_pose_mask_inference_to_reference_o \
    --pose_mask_inference_manifest "${INFER}/inference_manifest.json" \
    --pose_mask_runtime_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --reference_runtime_manifest "${REFERENCE}" \
    --output_dir "${REBASED}"
fi

echo "[${STAGE}5] 与point+mask重放做同对象、同噪声、20k点成对评测（CPU）"
if [ ! -s "${EVAL}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_pose_mask_pointcloud_ablation \
    --label_manifest "${LABEL}" \
    --reference_runtime_manifest "${REFERENCE}" \
    --pose_mask_runtime_manifest "${RUNTIME}/runtime_input_manifest.json" \
    --baseline_manifest "${BASELINE_INFER}/inference_manifest.json" \
    --pose_mask_rebased_manifest "${REBASED}/inference_manifest.json" \
    --output_dir "${EVAL}" \
    --surface_samples 20000 --expected_objects "${SUBSET_COUNT}"
fi

"${PY}" - "${EVAL}/report.json" "${SUBSET_COUNT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True and report["formal"] is False
expected = int(sys.argv[2])
assert report["object_count"] == expected and report["record_count"] == 2 * expected
print({
    "passed": True,
    "formal": False,
    "objects": report["object_count"],
    "summary": report["summary"],
    "paired_comparison": report["paired_comparison"],
    "report": sys.argv[1],
})
PY

echo "[${STAGE}6] 用同一surface seed重量化Pose+Mask与冻结外部base（CPU）"
if [ ! -s "${EXTERNAL}/report.json" ]; then
  REPORT_ARGS=(--ablation_report "${EVAL}/report.json")
  if [ -n "${PRIOR_REPORT}" ]; then
    REPORT_ARGS=(--ablation_report "${PRIOR_REPORT}" "${REPORT_ARGS[@]}")
  fi
  MPLCONFIGDIR=/tmp/matplotlib \
  "${PY}" -u -m pose_point_depth_mv.evaluate_pose_mask_external_bases \
    "${REPORT_ARGS[@]}" \
    --fiveway_report "${FIVEWAY}" \
    --output_dir "${EXTERNAL}" \
    --surface_samples 20000 \
    --expected_objects "${EXPECTED_TOTAL}"
fi

"${PY}" - "${EXTERNAL}/report.json" "${EXPECTED_TOTAL}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True and report["formal"] is False
assert report["object_count"] == int(sys.argv[2])
print({
    "passed": True,
    "formal": False,
    "objects": report["object_count"],
    "pose_mask_summary": report["summary"]["pose_mask"],
    "pose_mask_paired_comparisons": report["pose_mask_paired_comparisons"],
    "report": sys.argv[1],
})
PY
