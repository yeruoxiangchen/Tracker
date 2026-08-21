#!/usr/bin/env bash
set -uo pipefail

cd /home/zjr/Tracker || exit 97

GPU=${GPU:-1}
CACHE=/data/ar_ss_flow_pose_lifting_overfit64_v1_20260714
RUN=pose_point_depth_mv/outputs/c0_3_gaussian3_smoke4_s5_seed42_bf16_20260718
OVERALL=0

if [ -f "${RUN}/checkpoints/last.pt" ] && [ -f "${RUN}/train_report.json" ]; then
  echo "reuse C0.3 smoke: ${RUN}"
  TRAIN_CODE=0
elif [ -e "${RUN}" ]; then
  echo "incomplete smoke output exists: ${RUN}"
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
    --indices 0-3 \
    --max_steps 5 \
    --save_every 5 \
    --log_every 1 \
    --seed 42 \
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
    --expected_updates 5 \
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

if [ "${AUDIT_CODE}" -eq 0 ]; then
  /home/zjr/anaconda3/envs/reconviagen/bin/python -c '
import torch, sys
checkpoint = torch.load(sys.argv[1], map_location="cpu")
protocol = checkpoint["model_summary"]["protocol"]
assert protocol["training_spatial_tolerance"] == "gaussian3"
assert protocol["spatial_tolerance_symmetric_across_branches"] is True
assert protocol["spatial_tolerance_fixed_correct_support"] is True
assert checkpoint["args"]["voxel_reliability_weighting"] == "uniform"
print("C0.3 checkpoint protocol PASS")
' "${RUN}/checkpoints/last.pt" 2>&1 | tee "${RUN}/protocol_audit.log"
  PROTOCOL_CODE=${PIPESTATUS[0]}
else
  PROTOCOL_CODE=99
fi
echo "${PROTOCOL_CODE}" > "${RUN}.protocol.exit_code"
if [ "${PROTOCOL_CODE}" -ne 0 ]; then
  OVERALL=${PROTOCOL_CODE}
fi

echo "${OVERALL}" > "${RUN}/runner.status"
echo "C0.3 smoke complete: status=${OVERALL}"
exit 0
