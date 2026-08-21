#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker
source official_slat_with_vggt_perf_v1/source_server_env.sh

FORMAL_GPUS=${FORMAL_GPUS:-0,1,2,3,4,5,6,7}
FORMAL=${ROOT}/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1
FLOG=${ROOT}/logs/V_with_vggt_train2000_step15000_seed42_8gpu_strict_perf_v1_v1.log
EXIT_CODE=${FLOG}.exit_code

mkdir -p "${ROOT}/logs"

test -s "${FULL}/with_vggt_slat_manifest.json"
test -s "${FULL}/with_vggt_lifting_manifest.json"
test ! -e "${FORMAL}"
test ! -e "${FLOG}"
test ! -e "${EXIT_CODE}"

IFS=, read -r -a GPU_ARRAY <<<"${FORMAL_GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ]; then
  echo "ERROR: P7 requires exactly eight physical GPUs; got ${FORMAL_GPUS}" >&2
  exit 90
fi

exec > >(tee "${FLOG}") 2>&1

echo "============================================================"
echo "Official Train2000 with-VGGT fresh training: step0 -> step15000"
echo "started: $(date -Is)"
echo "host: $(hostname)"
echo "physical GPUs: ${FORMAL_GPUS}"
echo "output: ${FORMAL}"
echo "log: ${FLOG}"
echo "============================================================"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

set +e
CUDA_VISIBLE_DEVICES="${FORMAL_GPUS}" \
  "${TORCHRUN}" --standalone --nproc_per_node=8 \
  -m official_slat_with_vggt_perf_v1.train_proobjaverse_official \
  --cache_manifest "${FULL}/with_vggt_slat_manifest.json" \
  --lifting_cache_manifest "${FULL}/with_vggt_lifting_manifest.json" \
  --target_decoder_audit "${DECODER}" \
  --native_ss_report "${TRAINING_SS_REPORT}" \
  --stock_slat_freeze "${FREEZE}" \
  --output_dir "${FORMAL}" \
  --max_steps 25000 \
  --run_until_step 15000 \
  --save_every 1000 \
  --log_every 10 \
  --grad_accum 1 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --persistent_workers \
  --pin_memory \
  --torch_num_threads 2 \
  --torch_num_interop_threads 1
TRAIN_RC=$?
set -e

printf '%s\n' "${TRAIN_RC}" > "${EXIT_CODE}"
echo "finished: $(date -Is)"
echo "P7_TRAIN_RC=${TRAIN_RC}"
exit "${TRAIN_RC}"
