#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${OBJAVERSE16_OBJAVERSE2K_GPU:-0}
STEP=${OBJAVERSE2K_SLAT_SELECTED_STEP:?set OBJAVERSE2K_SLAT_SELECTED_STEP after dev64 checkpoint selection}
STEP_PAD=$(printf '%06d' "${STEP}")
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
OBJ16=/data/zjr/objaverse16_no_vggt_mixed_20260810_v1
SELECTION=${OBJ16}/O0_frozen_objaverse_test16_v1.json
LIFTING=${OBJ16}/O4_lifting_dino_only_v1/lifting_manifest.json
MODEL_INPUT=${OBJ16}/O5_model_inputs_target_free_v1/model_input_manifest.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS=${SS_RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
SLAT=${RUN}/slat_objaverse2135_step2000_seed42_4gpu_v1/checkpoints/step_${STEP_PAD}.pt
DEV_REPORT=${RUN}/eval_dev64_step${STEP_PAD}_stock_m8_objaverse2k_v1/comparison/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
INFERENCE=${OBJ16}/O8_objaverse2k_step${STEP_PAD}_seed42_v1
EVALUATION=${OBJ16}/O9_objaverse2k_step${STEP_PAD}_mesh_eval20k_v1
LOG_DIR=${RUN}/logs
STATE=${LOG_DIR}/objaverse16_objaverse2k_step${STEP_PAD}.state
EXIT_CODE=${LOG_DIR}/objaverse16_objaverse2k_step${STEP_PAD}.exit_code
LOCK=${LOG_DIR}/objaverse16_objaverse2k.lock

if [[ ! "${GPU}" =~ ^[0-9]+$ ]] || [[ ! "${STEP}" =~ ^[0-9]+$ ]]; then
  echo "GPU and selected step must be non-negative integers" >&2
  exit 96
fi
for REQUIRED in \
  "${SELECTION}" "${LIFTING}" "${MODEL_INPUT}" "${SS}" "${SS_REPORT}" \
  "${SLAT}" "${DEV_REPORT}" "${STOCK_FREEZE}"; do
  test -s "${REQUIRED}"
done

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then echo "Objaverse16 Objaverse2K diagnostic is already running" >&2; exit 99; fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s selected_step=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" "${STEP}" >"${STATE}"
rm -f "${EXIT_CODE}"

if [ ! -s "${INFERENCE}/inference_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_objaverse16_no_vggt_objaverse2k \
    --model_input_manifest "${MODEL_INPUT}" \
    --native_ss_checkpoint "${SS}" \
    --native_slat_checkpoint "${SLAT}" \
    --stock_slat_freeze "${STOCK_FREEZE}" \
    --native_ss_report "${SS_REPORT}" \
    --slat_dev_report "${DEV_REPORT}" \
    --output_dir "${INFERENCE}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --seeds 42 --weights ema --device cuda --amp_dtype bf16 \
    >"${LOG_DIR}/objaverse16_objaverse2k_step${STEP_PAD}_inference.log" 2>&1
fi

"${PY}" -u -m pose_point_depth_mv.evaluate_objaverse16_no_vggt \
  --selection_manifest "${SELECTION}" \
  --lifting_manifest "${LIFTING}" \
  --inference_manifest "${INFERENCE}/inference_manifest.json" \
  --output_dir "${EVALUATION}" \
  --surface_samples 20000 \
  --fscore_thresholds 0.01,0.02,0.05 \
  --resume

"${PY}" - "${EVALUATION}/report.json" "${SLAT}" "${DEV_REPORT}" <<'PY'
import json, sys
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
report = json.load(open(sys.argv[1], encoding="utf-8"))
dev = json.load(open(sys.argv[3], encoding="utf-8"))
assert report["passed"] is True and report["formal"] is False
assert report["protocol_scope"] == "frozen_objaverse_test16"
assert report["object_count"] == 16 and report["record_count"] == 16
assert report["target_or_metric_consumed_during_inference"] is False
assert report["point_cloud_tensor_consumed_during_inference"] is False
assert dev["checkpoints"]["objaverse2k"]["sha256"] == sha256_file(sys.argv[2])
print({"passed": True, "objects": 16, "scope": "final diagnostic, formal=false"})
PY
