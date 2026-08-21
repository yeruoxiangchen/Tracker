#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${REAL_S7_GPU:-5}
ROOT=/data/zjr/omni_real_video500_download_20260804_v2
ADAPT=/data/zjr/native_v2_real500_domain_adapt_20260806_v2
MODEL=${ROOT}/D12_benchmark32_native_v2_model_inputs_v2/model_input_manifest.json
LABEL=${ROOT}/D11_benchmark32_mesh_o_labels_v2/runtime_o_label_manifest.json
SS=${ADAPT}/ss_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
SLAT=${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2/checkpoints/step_001000.pt
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
PARENT=${ROOT}/D13_benchmark32_native_v2_full_seed42_v2/inference_manifest.json
RECON=${ROOT}/D14_benchmark32_reconviagen_original_seed42_v1/inference_manifest.json
PIXAL=${ROOT}/D15_benchmark32_pixal3d_official_seed42_v1/inference_manifest.json
INFER=${ROOT}/S7_benchmark32_native_v2_realadapt_step1000_seed42_v1
EVAL=${ROOT}/S7_benchmark32_fourway_realadapt_step1000_seed42_v1
STATE=${ADAPT}/logs/S7_benchmark32_background.state
EXIT_CODE=${ADAPT}/logs/S7_benchmark32_background.exit_code
LOCK=${ADAPT}/logs/S7_benchmark32_background.lock

mkdir -p "${ADAPT}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "S7 start refused: another S7 job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "S7 background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${MODEL}" "${LABEL}" "${SS}" "${SLAT}" "${FREEZE}" \
                "${PARENT}" "${RECON}" "${PIXAL}" \
                "${ADAPT}/slat_v2_real_step1000_seed42_2gpu_v2/report.json"; do
  test -s "${REQUIRED}"
done

if [ ! -s "${INFER}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_omni_real_native_v2 \
    --model_input_manifest "${MODEL}" \
    --native_ss_checkpoint "${SS}" \
    --native_slat_checkpoint "${SLAT}" \
    --stock_slat_freeze "${FREEZE}" \
    --output_dir "${INFER}" \
    --seeds 42 --weights ema --amp_dtype bf16 --device cuda
fi

if [ ! -s "${EVAL}/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_omni_real_native_adaptation \
    --label_manifest "${LABEL}" \
    --adapted_native_manifest "${INFER}/inference_manifest.json" \
    --parent_native_manifest "${PARENT}" \
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
print({
    "protocol_passed": True,
    "adaptation_primary_passed": report["adaptation_decision"]["primary_passed"],
    "secondary_all_nonnegative": report["adaptation_decision"]["secondary_all_nonnegative"],
})
PY
