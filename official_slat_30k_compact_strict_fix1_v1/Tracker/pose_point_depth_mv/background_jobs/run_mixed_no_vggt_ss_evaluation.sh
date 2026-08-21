#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${MIXED_NO_VGGT_SS_EVAL_GPU:-6}
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
VAL=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1/lifting_val64_dino_only_v1/lifting_manifest.json
CKPT=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt
TRAIN_REPORT=${RUN}/ss_mixed_step2000_seed42_1gpu_v1/report.json
CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
CAL=${RUN}/ss_calibration_synthetic_fixedcfg3_count125_v3
EVAL=${RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3
STATE=${RUN}/logs/M5_ss_evaluation.state
EXIT_CODE=${RUN}/logs/M5_ss_evaluation.exit_code
LOCK=${RUN}/logs/M5_ss_evaluation.lock

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "M5 refused: another mixed SS evaluation holds ${LOCK}" >&2
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

for REQUIRED in "${VAL}" "${CKPT}" "${TRAIN_REPORT}" "${CONTRACT}"; do
  test -s "${REQUIRED}"
done

"${PY}" - "${CKPT}" "${TRAIN_REPORT}" "${CONTRACT}" <<'PY'
import json
import sys
import torch

from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.real_full_no_vggt_migration import (
    load_migration_contract,
    validate_destination_migration,
)

checkpoint_path, report_path, contract_path = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu")
report = json.load(open(report_path, encoding="utf-8"))
contract = load_migration_contract(contract_path, stage="ss")
assert checkpoint["format"] == NATIVE_SS_NO_VGGT_VERSION
assert checkpoint["step"] == 2000
assert report["passed"] is True and report["completed"] is True
validate_native_ss_no_vggt_checkpoint(
    checkpoint, pretrained="Stable-X/trellis-vggt-v0-2", allow_v2_parent=False
)
validate_destination_migration(checkpoint, contract)
print({"passed": True, "checkpoint_step": checkpoint["step"], "weights": "ema"})
PY

# The evaluator creates its output directory before contract validation.  A
# failed preflight may therefore leave an empty directory; only empty dirs are
# safe to remove automatically on retry.
rmdir "${CAL}" 2>/dev/null || true
if [ ! -s "${CAL}/calibration.json" ]; then
  if [ -e "${CAL}" ]; then
    echo "M5 calibration output is partial; preserve and inspect: ${CAL}" >&2
    exit 98
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.evaluate_native_ss_genrecon_no_vggt \
    --mode calibrate \
    --cache_manifest "${VAL}" \
    --checkpoint "${CKPT}" \
    --output_dir "${CAL}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --object_start 16 --object_end 32 \
    --joint_seeds 42,43,44 \
    --candidate_cfg_strengths 3 \
    --weights ema --steps 25 --cfg_interval 0.5,1.0 \
    --guidance_rescale 0.0 --rescale_t 3.0 --amp_dtype bf16 \
    --bootstrap_samples 5000 \
    --min_iou_gain_mean 0.0 --min_iou_win_rate 0.5 \
    --min_recall_gain_mean 0.0 --min_latent_mse_gain_mean 0.0 \
    --min_count_ratio 0.85 --max_count_ratio 1.25 \
    --min_pose_control_iou_advantage 0.0
fi

"${PY}" - "${CAL}/calibration.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["passed"] is True and payload["selected"] is not None
print({"calibration_passed": True, "selected": payload["selected"]})
PY

rmdir "${EVAL}" 2>/dev/null || true
if [ ! -s "${EVAL}/report.json" ]; then
  if [ -e "${EVAL}" ]; then
    echo "M5 evaluation output is partial; preserve and inspect: ${EVAL}" >&2
    exit 97
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.evaluate_native_ss_genrecon_no_vggt \
    --mode evaluate \
    --cache_manifest "${VAL}" \
    --checkpoint "${CKPT}" \
    --output_dir "${EVAL}" \
    --calibration "${CAL}/calibration.json" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --object_start 32 --object_end 64 \
    --joint_seeds 42,43,44 \
    --weights ema --steps 25 --cfg_interval 0.5,1.0 \
    --guidance_rescale 0.0 --rescale_t 3.0 --amp_dtype bf16 \
    --bootstrap_samples 5000 \
    --min_iou_gain_mean 0.0 --min_iou_win_rate 0.5 \
    --min_recall_gain_mean 0.0 --min_latent_mse_gain_mean 0.0 \
    --min_count_ratio 0.85 --max_count_ratio 1.25 \
    --min_pose_control_iou_advantage 0.0
fi

"${PY}" - "${EVAL}/report.json" <<'PY'
import json
import sys

from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence

payload, binding = load_no_vggt_ss_evidence(sys.argv[1])
assert payload["passed"] is True
assert binding["checkpoint_step"] == 2000 and binding["weights"] == "ema"
print({
    "passed": True,
    "objects": payload["correct"]["object_count"],
    "cfg_strength": binding["cfg_strength"],
    "checks": payload["checks"],
})
PY
