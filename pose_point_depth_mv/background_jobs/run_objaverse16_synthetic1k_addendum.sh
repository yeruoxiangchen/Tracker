#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${OBJAVERSE16_SYNTHETIC1K_GPU:-4}
ROOT=${OBJAVERSE16_ROOT:-/data/zjr/objaverse16_no_vggt_mixed_20260810_v1}
RUN=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1
SELECTION=${ROOT}/O0_frozen_objaverse_test16_v1.json
MODEL_INPUT=${ROOT}/O5_model_inputs_target_free_v1/model_input_manifest.json
RECON=${ROOT}/O8_reconviagen_original_seed42_v1/inference_manifest.json
O9=${ROOT}/O9_current_vs_reconviagen_axisfixed_20k_v1/report.json
SS=${RUN}/ss868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SLAT=${RUN}/slat868_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SS_REPORT=${RUN}/ss_eval_final32_step2000_ema_sourcebalanced_v2/report.json
SLAT_REPORT=${RUN}/slat868_step2000_seed42_1gpu_v1/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
INFERENCE=${ROOT}/O10_synthetic1k_no_vggt_seed42_v1
EVALUATION=${ROOT}/O11_synthetic1k_current_reconviagen_axisfixed_20k_v1
LOG_DIR=${ROOT}/logs
STATE=${LOG_DIR}/Objaverse16_synthetic1k_addendum.state
EXIT_CODE=${LOG_DIR}/Objaverse16_synthetic1k_addendum.exit_code
LOCK=${LOG_DIR}/Objaverse16_synthetic1k_addendum.lock

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse16 synthetic1k refused: another job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "Objaverse16 synthetic1k addendum finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "OBJAVERSE16_SYNTHETIC1K_GPU must be one non-negative GPU index" >&2
  exit 96
fi
for REQUIRED in \
  "${PY}" "${SELECTION}" "${MODEL_INPUT}" "${RECON}" "${O9}" \
  "${SS}" "${SLAT}" "${SS_REPORT}" "${SLAT_REPORT}" "${STOCK_FREEZE}"; do
  test -s "${REQUIRED}"
done

echo "[O10] 同一冻结Objaverse16输入运行旧reviewed1k synthetic No-VGGT SS/SLat"
CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.infer_objaverse16_no_vggt_synthetic1k \
  --model_input_manifest "${MODEL_INPUT}" \
  --native_ss_checkpoint "${SS}" \
  --native_slat_checkpoint "${SLAT}" \
  --native_ss_report "${SS_REPORT}" \
  --native_slat_report "${SLAT_REPORT}" \
  --stock_slat_freeze "${STOCK_FREEZE}" \
  --output_dir "${INFERENCE}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --seeds 42 \
  --weights ema \
  --device cuda \
  --amp_dtype bf16

echo "[O11] 复用O9基线，统一固定轴变换/canonical GT/20k采样做三路评测"
"${PY}" -u -m pose_point_depth_mv.evaluate_objaverse16_synthetic1k_addendum \
  --selection_manifest "${SELECTION}" \
  --synthetic1k_inference_manifest "${INFERENCE}/inference_manifest.json" \
  --existing_o9_report "${O9}" \
  --output_dir "${EVALUATION}" \
  --surface_samples 20000 \
  --fscore_thresholds 0.01,0.02,0.05 \
  --resume

"${PY}" -c 'import json,sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["passed"] is True and r["formal"] is False
assert r["methods"] == ["synthetic1k_no_vggt", "current_no_vggt", "reconviagen_original"]
assert r["object_count"] == 16 and r["record_count"] == 48
assert r["pair_count_per_comparison"] == 16
assert r["coordinate_evaluation"]["applied_identically_to_all_methods"] is True
print({"passed": True, "objects": 16, "records": 48, "report": sys.argv[1]})' \
  "${EVALUATION}/report.json"
