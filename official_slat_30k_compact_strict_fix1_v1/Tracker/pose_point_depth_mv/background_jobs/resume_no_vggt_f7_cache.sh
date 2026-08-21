#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${NO_VGGT_SLAT_CACHE_GPU:-0}
RUN=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1
LIFT=${RUN}/lifting_train868_dino_only_v1/lifting_manifest.json
SS_CKPT=${RUN}/ss868_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt
SS_REPORT=${RUN}/ss_eval_final32_step2000_ema_sourcebalanced_v2/report.json
SLAT_ROOT=/data/zjr/native3d_condition_reviewed1k_inputs_20260730_v3/lh_slats_train_val_v2
OUT=${RUN}/slat_cache_train868_seed42_v1
STATE=${RUN}/logs/F7_slat_cache_background.state
EXIT_CODE=${RUN}/logs/F7_slat_cache_background.exit_code
LOCK=${RUN}/logs/F7_slat_cache_background.lock

mkdir -p "${RUN}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "F7 resume refused: another F7 background job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "F7 background job finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT

printf 'started_at=%s state=running gpu=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

if pgrep -u "$(id -u)" -af -- '-m pose_point_depth_mv\.build_direct_slat_cache_no_vggt( |$)' >/dev/null; then
  echo "F7 resume refused: a legacy no-VGGT cache process is still running" >&2
  pgrep -u "$(id -u)" -af -- '-m pose_point_depth_mv\.build_direct_slat_cache_no_vggt( |$)' >&2
  exit 99
fi

for REQUIRED in "${LIFT}" "${SS_CKPT}" "${SS_REPORT}"; do
  test -s "${REQUIRED}"
done
test -d "${SLAT_ROOT}"

"${PY}" - "${SS_REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["format"] == "pose_point_depth_mv.native_ss_no_vggt_eval.v1"
assert report["passed"] is True
print({"formal_final32_passed": True, "checkpoint_step": report["protocol"]["checkpoint_step"]})
PY

if [ -s "${OUT}/manifest.json" ]; then
  echo "F7 already complete: ${OUT}/manifest.json"
  exit 0
fi

RESUME=()
if [ -e "${OUT}" ]; then
  RESUME=(--resume)
  echo "F7 resuming preserved partial cache: ${OUT}"
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.build_direct_slat_cache_no_vggt \
  --lifting_manifest "${LIFT}" \
  --slat_root "${SLAT_ROOT}" \
  --flow_checkpoint "${SS_CKPT}" \
  --native_ss_report "${SS_REPORT}" \
  --condition_arch native_ss_genrecon_v2 \
  --output_dir "${OUT}" \
  --ss_seeds 42 \
  --expected_ss_step 2000 \
  --amp_dtype bf16 \
  --require_all_objects \
  "${RESUME[@]}"

test -s "${OUT}/manifest.json"
