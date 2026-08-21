#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

SEED=${1:-42}
GPU=${2:-1}
CACHE=/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714
RUN=pose_point_depth_mv/outputs/c0_3_gaussian3_train16_s200_seed${SEED}_bf16_20260718
OVERALL=0

json_pass_code() {
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(98)
r = json.loads(p.read_text(encoding="utf-8"))
schema_ok = (
    bool(r.get("checkpoint_sha256"))
    and r.get("hard_admitted_soft_weight_protocol", {}).get("formal_n3_gate") is True
    and r.get("continuous_soft_weight_protocol", {}).get("c1_ablation_only") is True
)
raise SystemExit(0 if r.get("passed") is True and schema_ok else 2)
' "$1"
}

if [ -f "${RUN}/checkpoints/last.pt" ] && [ -f "${RUN}/train_report.json" ]; then
  echo "reuse completed training files: ${RUN}"
  TRAIN_CODE=0
elif [ -e "${RUN}" ]; then
  echo "incomplete training output exists: ${RUN}"
  TRAIN_CODE=98
else
  CUDA_VISIBLE_DEVICES=${GPU} \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ATTN_BACKEND=flash_attn \
  SPCONV_ALGO=native \
  MPLCONFIGDIR=/tmp/matplotlib \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
    pose_point_depth_mv.train_correspondence_head \
    --cache_manifest "${CACHE}/manifest.json" \
    --output_dir "${RUN}" \
    --indices 0-15 \
    --max_steps 200 \
    --save_every 50 \
    --log_every 10 \
    --seed "${SEED}" \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --grad_accum 1 \
    --grad_clip 1.0 \
    --amp_dtype bf16 \
    --nonfinite_policy error \
    --max_nonfinite_attempts 0 \
    --hidden_dim 64 \
    --pair_hidden_dim 96 \
    --min_views 2 \
    --max_train_views 0 \
    --spatial_tolerance gaussian3 \
    --train_controls pose_cyclic1,depth_view_cyclic1,visual_view_cyclic1 \
    --sample_bce_weight 1.0 \
    --sample_rank_weight 1.0 \
    --voxel_bce_weight 0.25 \
    --hard_negative_weight 0.5 \
    --rank_margin 0.25 \
    --voxel_rank_weight 1.0 \
    --voxel_rank_margin 0.25 \
    --voxel_rank_temperature 0.10 \
    --voxel_rank_hard_weight 0.5 \
    --voxel_reliability_weighting uniform \
    2>&1 | tee "${RUN}.log"
  TRAIN_CODE=${PIPESTATUS[0]}
fi
echo "${TRAIN_CODE}" > "${RUN}.train.exit_code"
if [ "${TRAIN_CODE}" -ne 0 ]; then
  OVERALL=${TRAIN_CODE}
fi

if [ "${TRAIN_CODE}" -eq 0 ]; then
  /home/zjr/anaconda3/envs/reconviagen/bin/python -u \
    reconvggt_ar_adapter_a/audit_pointpose_training_run.py \
    --train_report "${RUN}/train_report.json" \
    --checkpoint "${RUN}/checkpoints/last.pt" \
    --expected_updates 200 \
    --max_nonfinite_attempts 0 \
    --output "${RUN}/finite_run_audit.json" \
    2>&1 | tee "${RUN}/finite_run_audit.log"
  AUDIT_CODE=${PIPESTATUS[0]}
else
  AUDIT_CODE=99
fi
echo "${AUDIT_CODE}" > "${RUN}.audit.exit_code"
if [ "${AUDIT_CODE}" -ne 0 ]; then
  OVERALL=${AUDIT_CODE}
fi

run_split() {
  local SPLIT=$1
  local INDICES=$2
  local OUT=$3
  local CODE
  if [ -f "${OUT}/report.json" ] && [ -d "${OUT}/voxel_maps" ]; then
    json_pass_code "${OUT}/report.json"
    CODE=$?
    echo "reuse ${SPLIT} report: code=${CODE}"
  elif [ -e "${OUT}" ]; then
    echo "incomplete ${SPLIT} output exists: ${OUT}"
    CODE=98
  else
    CUDA_VISIBLE_DEVICES=${GPU} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn \
    SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
      pose_point_depth_mv.eval_voxel_selfcal_correspondence \
      --cache_manifest "${CACHE}/manifest.json" \
      --checkpoint "${RUN}/checkpoints/last.pt" \
      --output_dir "${OUT}" \
      --indices "${INDICES}" \
      --split_name "${SPLIT}" \
      --max_samples 0 \
      --device cuda \
      --threshold 0.0 \
      --bootstrap_samples 10000 \
      --min_voxel_positive_ratio 0.60 \
      --min_per_object_positive_ratio 0.50 \
      --min_object_local_pass_rate 0.65 \
      --min_heldout_gate_positive_ratio 0.65 \
      --min_spatial_control_object_win_rate 0.65 \
      --min_spatial_control_gate_positive_ratio 0.65 \
      --min_spatial_std 1e-4 \
      --max_permutation_diff 1e-5 \
      --spatial_tolerance checkpoint \
      --soft_gate_temperature 0.25 \
      --soft_gate_reliability_power 1.0 \
      --continuous_gate_max_scale 0.10 \
      --save_maps \
      --fail_on_decision \
      2>&1 | tee "${OUT}.log"
    CODE=${PIPESTATUS[0]}
  fi
  echo "${CODE}" > "${OUT}.exit_code"
  if [ "${CODE}" -ne 0 ]; then
    OVERALL=${CODE}
  fi
}

if [ "${AUDIT_CODE}" -eq 0 ]; then
  run_split train16 0-15 "${RUN}/c0_3_train16"
  run_split fresh48 16-63 "${RUN}/c0_3_fresh48"
else
  echo "skip C0.3 evaluation because finite audit did not pass"
fi

echo "${OVERALL}" > "${RUN}/runner.status"
echo "C0.3 seed=${SEED} complete: status=${OVERALL}"

# This script is launched with /bin/bash under nohup. It never owns the caller shell.
exit 0
