#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

export EVAL_GPUS=4,5,6,7
source pose_point_depth_mv/background_jobs/source_proobjaverse_30k_dev64_abc_r_env.sh

if [[ ${EVAL_GPU_COUNT} -ne 4 ]]; then
  echo "ERROR: P6 requires exactly four GPUs; got ${EVAL_GPUS}" >&2
  exit 90
fi

test -s "${SS_CALIBRATION}/calibration.json"
test -s "${DEV_SS_TARGETS}/lifting_manifest.json"
test -s "${SS30K_CHECKPOINT}"
test ! -e "${SS_DEV_SHARDS}"

# Preserve the three spare-GPU reservations throughout P5, then release only
# these workflow-owned tmux sessions immediately before P6 claims 4/5/6/7.
for gpu in 5 6 7; do
  tmux kill-session -t "ss30k_p5_spare_hold_gpu${gpu}" 2>/dev/null || true
done
sleep 2
if pgrep -af '[h]old_eval_gpu.py.*ss30k_p5_spare_until_p6' >/dev/null; then
  echo "ERROR: a P5 spare-GPU holder survived its tmux release" >&2
  pgrep -af '[h]old_eval_gpu.py.*ss30k_p5_spare_until_p6' >&2 || true
  exit 91
fi

mkdir -p "${SS_DEV_SHARDS}/logs"
IFS=, read -r -a GPUS <<<"${EVAL_GPUS}"
STARTS=(0 16 32 48)
ENDS=(16 32 48 64)

for i in 0 1 2 3; do
  output=${SS_DEV_SHARDS}/shard${i}_${STARTS[$i]}_${ENDS[$i]}
  log=${SS_DEV_SHARDS}/logs/shard${i}_gpu${GPUS[$i]}.log
  session=ss30k_dev64_${i}
  ! tmux has-session -t "${session}" 2>/dev/null
  tmux new-session -d -s "${session}" \
    "bash -lc 'cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${GPUS[$i]} ${PY} -u -m pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_no_vggt --mode evaluate --allow_cross_manifest_calibration --cache_manifest ${DEV_SS_TARGETS}/lifting_manifest.json --checkpoint ${SS30K_CHECKPOINT} --calibration ${SS_CALIBRATION}/calibration.json --output_dir ${output} --object_start ${STARTS[$i]} --object_end ${ENDS[$i]} --joint_seeds 42,43,44 --weights ema --steps 25 --cfg_interval 0.5,1.0 --guidance_rescale 0.0 --rescale_t 3.0 --amp_dtype bf16 --bootstrap_samples 5000 >> ${log} 2>&1'"
  printf 'worker=%d gpu=%s range=[%s,%s) session=%s log=%s\n' \
    "${i}" "${GPUS[$i]}" "${STARTS[$i]}" "${ENDS[$i]}" \
    "${session}" "${log}"
done

echo "P6 FOUR-GPU WORKERS LAUNCHED; aggregate was not started."
echo "monitor: tail -F ${SS_DEV_SHARDS}/logs/*.log"
