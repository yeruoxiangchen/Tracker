#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPU=${OBJAVERSE16_RECON_GPU:-4}
ROOT=${OBJAVERSE16_ROOT:-/data/zjr/objaverse16_no_vggt_mixed_20260810_v1}
SELECTION=${ROOT}/O0_frozen_objaverse_test16_v1.json
LIFT_FULL=${ROOT}/O3_lifting_full_v1/manifest.json
CURRENT=${ROOT}/O6_native_no_vggt_mixed_seed42_v1/inference_manifest.json
LEGACY_O7=${ROOT}/O7_canonical_mesh_eval_20k_v1/report.json
RECON=${ROOT}/O8_reconviagen_original_seed42_v1
JOINT=${ROOT}/O9_current_vs_reconviagen_axisfixed_20k_v1
LOG_DIR=${ROOT}/logs
STATE=${LOG_DIR}/Objaverse16_reconviagen_addendum.state
EXIT_CODE=${LOG_DIR}/Objaverse16_reconviagen_addendum.exit_code
LOCK=${LOG_DIR}/Objaverse16_reconviagen_addendum.lock

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse16 ReconViaGen refused: another job holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  echo "Objaverse16 ReconViaGen addendum finished: rc=${RC}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpu=%s\n' "$(date --iso-8601=seconds)" "${GPU}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${PY}" "${SELECTION}" "${LIFT_FULL}" "${CURRENT}"; do
  test -s "${REQUIRED}"
done

LEGACY_O7_ARG=()
if [ -s "${LEGACY_O7}" ]; then
  LEGACY_O7_ARG=(--legacy_o7_report "${LEGACY_O7}")
fi

echo "[O8] 同一冻结RGB/mask视图运行原版ReconViaGen stock；不传相机或点云"
CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.infer_objaverse16_reconviagen \
  --selection_manifest "${SELECTION}" \
  --source_lifting_manifest "${LIFT_FULL}" \
  --output_dir "${RECON}" \
  --pretrained Stable-X/trellis-vggt-v0-2 \
  --seeds 42 \
  --device cuda \
  --low_vram \
  --multiimage_algo multidiffusion \
  --resume

echo "[O9] 两方法统一固定decoder轴变换、canonical GT和20k表面采样联合评测"
"${PY}" -u -m pose_point_depth_mv.evaluate_objaverse16_reconviagen \
  --selection_manifest "${SELECTION}" \
  --current_inference_manifest "${CURRENT}" \
  --reconviagen_inference_manifest "${RECON}/inference_manifest.json" \
  "${LEGACY_O7_ARG[@]}" \
  --output_dir "${JOINT}" \
  --surface_samples 20000 \
  --fscore_thresholds 0.01,0.02,0.05 \
  --resume

"${PY}" -c 'import json,sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["passed"] is True and r["formal"] is False
assert r["methods"] == ["current_no_vggt", "reconviagen_original"]
assert r["object_count"] == 16 and r["record_count"] == 32 and r["pair_count"] == 16
assert r["coordinate_evaluation"]["applied_identically_to_both_methods"] is True
print({"passed": True, "objects": 16, "pairs": 16, "report": sys.argv[1]})' \
  "${JOINT}/report.json"
