#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPUS=${OBJ2K_DEV64_RECON_GPUS:-0,1,2,3,4,5,6,7}
ROOT=${OBJ2K_DEV64_RECON_ROOT:-/data/zjr/objaverse2k_dev64_vs_reconviagen_20260813_v1}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
DEV_LIFT=${RUN}/split_dev64_v1/dev/lifting_manifest.json
DEV_CACHE=${RUN}/slat_cache_dev64_seed424344_merged_v1/manifest.json
NATIVE_RUN=${RUN}/eval_dev64_step002000_stock_m8_objaverse2k_8gpu_v1
SELECTION=${ROOT}/selection.json
LIFTING_SUBSET=${ROOT}/lifting_subset.json
RECON_ROOT=${ROOT}/reconviagen_original
EVALUATION=${ROOT}/evaluation_strict_20k
LOG_DIR=${ROOT}/logs
STATE=${LOG_DIR}/run.state
EXIT_CODE=${LOG_DIR}/run.exit_code
LOCK=${LOG_DIR}/run.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]; then
  echo "OBJ2K_DEV64_RECON_GPUS requires eight distinct GPUs" >&2
  exit 96
fi
for REQUIRED in "${PY}" "${DEV_LIFT}" "${DEV_CACHE}"; do
  test -s "${REQUIRED}"
done
NATIVE_REPORT_ARGS=()
for WORKER in 0 1 2 3; do
  REPORT=${NATIVE_RUN}/objaverse2k_worker_${WORKER}/report.json
  test -s "${REPORT}"
  NATIVE_REPORT_ARGS+=(--native_report "${REPORT}")
done

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K dev64 vs ReconViaGen is already running" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" >"${STATE}"
rm -f "${EXIT_CODE}"

"${PY}" -u -m pose_point_depth_mv.prepare_objaverse2k_dev64_reconviagen_selection \
  --lifting_manifest "${DEV_LIFT}" \
  --slat_manifest "${DEV_CACHE}" \
  "${NATIVE_REPORT_ARGS[@]}" \
  --output_selection "${SELECTION}" \
  --output_lifting_subset "${LIFTING_SUBSET}" \
  --resume

PIDS=()
for WORKER in 0 1 2 3 4 5 6 7; do
  WORKER_OUT=${RECON_ROOT}/worker_${WORKER}
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${WORKER}]}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -u -m pose_point_depth_mv.infer_objaverse16_reconviagen \
    --selection_manifest "${SELECTION}" \
    --source_lifting_manifest "${LIFTING_SUBSET}" \
    --output_dir "${WORKER_OUT}" \
    --pretrained Stable-X/trellis-vggt-v0-2 \
    --seeds 42,43,44 --device cuda --low_vram \
    --multiimage_algo multidiffusion \
    --worker_index "${WORKER}" --num_workers 8 --resume \
    >"${LOG_DIR}/recon_worker_${WORKER}.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then FAILED=1; fi
done
if [ "${FAILED}" -ne 0 ]; then
  echo "At least one ReconViaGen worker failed; inspect ${LOG_DIR}" >&2
  exit 2
fi

EVAL_ARGS=()
for WORKER in 0 1 2 3 4 5 6 7; do
  MANIFEST=${RECON_ROOT}/worker_${WORKER}/inference_manifest.json
  test -s "${MANIFEST}"
  EVAL_ARGS+=(--reconviagen_manifest "${MANIFEST}")
done
"${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_dev64_vs_reconviagen \
  --selection_manifest "${SELECTION}" \
  "${EVAL_ARGS[@]}" \
  --output_dir "${EVALUATION}" \
  --surface_samples 20000 \
  --fscore_thresholds 0.01,0.02,0.05 \
  --bootstrap_samples 10000 \
  --resume

test -s "${EVALUATION}/report.json"
echo "Objaverse2K dev64 strict report: ${EVALUATION}/report.json"
