#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${MIXED_NO_VGGT_BENCH_GPU:-6}
ROOT=/data/zjr/omni_real_video500_download_20260804_v2
ADAPT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
RUNTIME=${ROOT}/D9_benchmark32_runtime_o_v2/runtime_input_manifest.json
LABEL=${ROOT}/D11_benchmark32_mesh_o_labels_v2/runtime_o_label_manifest.json
FULL_MODEL=${ROOT}/D12_benchmark32_native_v2_model_inputs_v2/model_input_manifest.json
DINO_MODEL=${ROOT}/M9_benchmark32_dino_only_model_inputs_v1
SS_FINAL=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SLAT_FINAL=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
SLAT_CONTRACT=${RUN}/contracts/slat_real_full_ema_v1.json
SS_EVIDENCE=${RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SLAT_REPORT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/report.json
SS_REAL=${ADAPT}/ss_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SLAT_REAL=${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SLAT_REAL_REPORT=${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
NO_VGGT=${ROOT}/M9_benchmark32_native_no_vggt_mixed1244_seed42_v1
REAL_FULL=${ROOT}/S7_benchmark32_native_v2_realadapt_step1000_seed42_v1
SYNTH_FULL=${ROOT}/D13_benchmark32_native_v2_full_seed42_v2/inference_manifest.json
RECON=${ROOT}/D14_benchmark32_reconviagen_original_seed42_v1/inference_manifest.json
PIXAL=${ROOT}/D15_benchmark32_pixal3d_official_seed42_v1/inference_manifest.json
EVAL=${ROOT}/M9_benchmark32_fiveway_no_vggt_mixed1244_seed42_v1
STATE=${RUN}/logs/M9_benchmark32.state
EXIT_CODE=${RUN}/logs/M9_benchmark32.exit_code
LOCK=${RUN}/logs/M9_benchmark32.lock

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "M9 refused: another final no-VGGT benchmark holds ${LOCK}" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${RUNTIME}" "${LABEL}" "${FULL_MODEL}" \
                "${SS_FINAL}" "${SLAT_FINAL}" "${SS_CONTRACT}" \
                "${SLAT_CONTRACT}" "${SS_EVIDENCE}" "${SLAT_REPORT}" \
                "${SS_REAL}" "${SLAT_REAL}" "${SLAT_REAL_REPORT}" \
                "${FREEZE}" "${SYNTH_FULL}" "${RECON}" "${PIXAL}"; do
  test -s "${REQUIRED}"
done

if [ ! -s "${DINO_MODEL}/model_input_manifest.json" ]; then
  RESUME=()
  if [ -e "${DINO_MODEL}" ]; then RESUME=(--resume); fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs \
    --runtime_input_manifest "${RUNTIME}" \
    --output_dir "${DINO_MODEL}" \
    --device cuda \
    "${RESUME[@]}"
fi

if [ ! -s "${REAL_FULL}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_v2 \
    --model_input_manifest "${FULL_MODEL}" \
    --native_ss_checkpoint "${SS_REAL}" \
    --native_slat_checkpoint "${SLAT_REAL}" \
    --stock_slat_freeze "${FREEZE}" \
    --output_dir "${REAL_FULL}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi

if [ ! -s "${NO_VGGT}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed \
    --model_input_manifest "${DINO_MODEL}/model_input_manifest.json" \
    --native_ss_checkpoint "${SS_FINAL}" \
    --native_slat_checkpoint "${SLAT_FINAL}" \
    --ss_migration_contract "${SS_CONTRACT}" \
    --slat_migration_contract "${SLAT_CONTRACT}" \
    --stock_slat_freeze "${FREEZE}" \
    --output_dir "${NO_VGGT}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi

if [ ! -s "${EVAL}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_omni_real_no_vggt_final \
    --label_manifest "${LABEL}" \
    --no_vggt_manifest "${NO_VGGT}/inference_manifest.json" \
    --real_full_manifest "${REAL_FULL}/inference_manifest.json" \
    --synthetic_full_manifest "${SYNTH_FULL}" \
    --reconviagen_manifest "${RECON}" \
    --pixal3d_manifest "${PIXAL}" \
    --output_dir "${EVAL}" \
    --protocol_scope development_benchmark32 \
    --expected_objects 32 \
    --surface_samples 20000
fi

"${PY}" - "${EVAL}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True
assert report["formal"] is False and report["holdout64_consumed"] is False
decision = report["no_vggt_decision"]
print({
    "protocol_passed": True,
    "superiority_passed": decision["superiority_passed"],
    "primary_non_regression_passed": decision["primary_non_regression_passed"],
    "secondary_retention_passed": decision["secondary_retention_passed"],
    "holdout_unlock_passed": decision["holdout_unlock_passed"],
})
assert decision["holdout_unlock_passed"] is True
PY
