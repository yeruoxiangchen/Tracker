#!/usr/bin/env bash
set -euo pipefail

source /home/zjr/anaconda3/etc/profile.d/conda.sh
conda activate reconviagen

PROJECT=${PROJECT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-3,4,5,6,7}

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/data/zjr/slat_train2000_trajectory_archives/slat_train2000_trajectory_step10000_25000_strict_fix1_v1/checkpoints}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1}
MASTER_LOG=${MASTER_LOG:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/logs/slat_trajectory_15k20k25k_5gpu_strict_fix1_v1.log}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT}:${PROJECT}/ReconViaGen:${PROJECT}/ReconViaGen/wheels/vggt${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT}"
test -x "${PY}"
test -d "${CHECKPOINT_ROOT}"
mkdir -p "$(dirname "${MASTER_LOG}")"

echo "============================================================"
echo "ProObjaverse SLat trajectory evaluation: 15k / 20k / 25k"
echo "five GPUs: ${EVAL_GPUS}"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "output root: ${OUTPUT_ROOT}"
echo "started: $(date -Is)"
echo "============================================================"

set +e
"${PY}" -u -m pose_point_depth_mv.run_proobjaverse_slat_trajectory_eval_5gpu \
  --project_root "${PROJECT}" \
  --python "${PY}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpus "${EVAL_GPUS}" \
  "$@" \
  2>&1 | tee -a "${MASTER_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "finished: $(date -Is)" | tee -a "${MASTER_LOG}"
echo "exit status: ${STATUS}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
exit "${STATUS}"
