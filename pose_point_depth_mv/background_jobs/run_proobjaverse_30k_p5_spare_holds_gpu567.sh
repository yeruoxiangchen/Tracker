#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

export EVAL_GPUS=4,5,6,7
source pose_point_depth_mv/background_jobs/source_proobjaverse_30k_dev64_abc_r_env.sh

HOLD_ROOT=${EVAL30K_ROOT}/logs/p5_spare_gpu_holds_v1
HOLD_MIB=${EVAL_GPU_HOLD_MIB:-64}
LABEL=ss30k_p5_spare_until_p6
mkdir -p "${HOLD_ROOT}"

if ! [[ ${HOLD_MIB} =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_GPU_HOLD_MIB must be a positive integer" >&2
  exit 90
fi

for gpu in 5 6 7; do
  session=ss30k_p5_spare_hold_gpu${gpu}
  ready=${HOLD_ROOT}/gpu${gpu}.ready.json
  log=${HOLD_ROOT}/gpu${gpu}.log
  ! tmux has-session -t "${session}" 2>/dev/null
  test ! -e "${ready}"
  tmux new-session -d -s "${session}" \
    "bash -lc 'cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${gpu} ${PY} -u pose_point_depth_mv/background_jobs/hold_eval_gpu.py --physical-gpu ${gpu} --memory-mib ${HOLD_MIB} --label ${LABEL} --ready-file ${ready} >> ${log} 2>&1'"
  printf 'holder gpu=%s session=%s ready=%s log=%s\n' \
    "${gpu}" "${session}" "${ready}" "${log}"
done

deadline=$((SECONDS + 120))
for gpu in 5 6 7; do
  ready=${HOLD_ROOT}/gpu${gpu}.ready.json
  session=ss30k_p5_spare_hold_gpu${gpu}
  while [[ ! -s ${ready} ]]; do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "ERROR: holder exited before ready: GPU${gpu}" >&2
      for cleanup_gpu in 5 6 7; do
        tmux kill-session -t "ss30k_p5_spare_hold_gpu${cleanup_gpu}" 2>/dev/null || true
      done
      exit 91
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: timeout waiting for holder: GPU${gpu}" >&2
      for cleanup_gpu in 5 6 7; do
        tmux kill-session -t "ss30k_p5_spare_hold_gpu${cleanup_gpu}" 2>/dev/null || true
      done
      exit 92
    fi
    sleep 0.2
  done
done

echo "P5 SPARE GPU HOLDERS READY: GPUs 5,6,7 (${HOLD_MIB} MiB each)"
echo "P6 launcher will release these exact holders before starting workers."
