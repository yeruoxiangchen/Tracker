#!/usr/bin/env bash
set -u

OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1}
PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
PROTOCOL=${OUT}/protocol.json
MASTER=${OUT}/logs/master.log

date -u -Is
echo "============================================================"
if [[ ! -s "${PROTOCOL}" ]]; then
  frozen=$(grep -c '^\[freeze\]' "${MASTER}" 2>/dev/null || true)
  echo "stage=freeze_source_identity"
  echo "objects_hashed=${frozen}/200"
  tail -n 5 "${MASTER}" 2>/dev/null || true
else
  "${PY}" -m pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark \
    status --protocol "${PROTOCOL}" 2>&1 || true
  echo "------------------------------------------------------------"
  for log in "${OUT}"/logs/worker_*_gpu*.log; do
    [[ -f "${log}" ]] || continue
    printf '%s: ' "$(basename "${log}")"
    tail -n 1 "${log}"
  done
fi
echo "------------------------------------------------------------"
if tmux has-session -t omni200render 2>/dev/null; then
  echo "tmux=omni200render RUNNING"
else
  echo "tmux=omni200render EXITED_OR_COMPLETE"
fi
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null || true
