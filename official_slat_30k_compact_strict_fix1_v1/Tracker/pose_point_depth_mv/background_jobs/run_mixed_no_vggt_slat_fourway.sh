#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
ROOT=/data/zjr/mixed_no_vggt_slat_fourway_20260811_v1
GPU=${MIXED_SLAT_FOURWAY_GPU:-4}
OBJECTS_PER_DOMAIN=${MIXED_SLAT_FOURWAY_OBJECTS_PER_DOMAIN:-16}
TRAIN_STEP=${MIXED_SLAT_FOURWAY_TRAIN_STEP:-2000}
if ! [[ "${TRAIN_STEP}" =~ ^[0-9]+$ ]] || \
   [ "${TRAIN_STEP}" -lt 1 ] || [ "${TRAIN_STEP}" -gt 2000 ]; then
  echo "MIXED_SLAT_FOURWAY_TRAIN_STEP must be an integer in [1,2000]" >&2
  exit 96
fi
STEP_PAD=$(printf '%06d' "${TRAIN_STEP}")
CHECKPOINT=${RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_${STEP_PAD}.pt
if [ "${TRAIN_STEP}" -eq 2000 ]; then
  OUT=${ROOT}/synth${OBJECTS_PER_DOMAIN}_real${OBJECTS_PER_DOMAIN}_seed42
  JOB=fourway_synth${OBJECTS_PER_DOMAIN}_real${OBJECTS_PER_DOMAIN}
else
  OUT=${ROOT}/step${STEP_PAD}_synth${OBJECTS_PER_DOMAIN}_real${OBJECTS_PER_DOMAIN}_seed42
  JOB=fourway_step${STEP_PAD}_synth${OBJECTS_PER_DOMAIN}_real${OBJECTS_PER_DOMAIN}
fi
STATE=${ROOT}/logs/${JOB}.state
EXIT_CODE=${ROOT}/logs/${JOB}.exit_code
LOCK=${ROOT}/logs/${JOB}.lock

mkdir -p "${ROOT}/logs"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "duplicate mixed SLat four-way job: ${JOB}" >&2
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
printf 'started_at=%s state=running gpu=%s objects_per_domain=%s train_step=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" "${OBJECTS_PER_DOMAIN}" \
  "${TRAIN_STEP}" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in \
  "${RUN}/manifests/mixed_slat_synth868_real376_v1.json" \
  "${RUN}/manifests/mixed_ss_lifting_synth868_real376_v1.json" \
  "${CHECKPOINT}" \
  "${RUN}/contracts/slat_real_full_ema_v1.json" \
  /data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json; do
  test -s "${REQUIRED}"
done

if [ -s "${OUT}/report.json" ] && [ -s "${OUT}/summary.txt" ]; then
  "${PY}" - "${OUT}/report.json" "${TRAIN_STEP}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
assert p["passed"] is True and p["formal"] is False
assert p["training_overlap"] is True
assert p["stock_context_views"] == "all"
assert p["run_config"]["checkpoint_step"] == int(sys.argv[2])
PY
  echo "reuse complete mixed SLat four-way report: ${OUT}"
  exit 0
fi

RESUME=()
if [ -e "${OUT}" ]; then
  RESUME=(--resume)
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
ATTN_BACKEND=flash_attn \
SPCONV_ALGO=native \
MPLCONFIGDIR=/tmp/matplotlib \
NUMBA_CACHE_DIR=/tmp/numba_cache \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -u -m pose_point_depth_mv.evaluate_mixed_no_vggt_slat_fourway \
  --cache_manifest "${RUN}/manifests/mixed_slat_synth868_real376_v1.json" \
  --lifting_cache_manifest "${RUN}/manifests/mixed_ss_lifting_synth868_real376_v1.json" \
  --checkpoint "${CHECKPOINT}" \
  --migration_contract "${RUN}/contracts/slat_real_full_ema_v1.json" \
  --stock_slat_freeze \
    /data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json \
  --output_dir "${OUT}" \
  --weights ema \
  --objects_per_domain "${OBJECTS_PER_DOMAIN}" \
  --selection_seed 20260811 \
  --noise_seed 42 \
  --surface_samples 20000 \
  --bootstrap_samples 10000 \
  --amp_dtype bf16 \
  "${RESUME[@]}"

test -s "${OUT}/report.json"
test -s "${OUT}/summary.txt"
