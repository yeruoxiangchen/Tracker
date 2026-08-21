#!/usr/bin/env bash
# Common paths only.  This file launches no process and changes no artifact.

export PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
export PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
export PROTOCOL_SHA=${PROTOCOL_SHA:-86235b8039130c5c9e50f463e022142b967cf9d6d51089f049f434476e1218c6}
export RAW30K=${RAW30K:-/data/zjr/ProObjaverse-300K-ReconViaGen-30K}
export SELECTION30K=${SELECTION30K:-/data/zjr/ProObjaverse-300K-ReconViaGen-30K-state/combined_audit/combined_selection_30k.json}
export F39_PROTOCOL_DIR=${F39_PROTOCOL_DIR:-/data/zjr/proobjaverse_official_slat_protocol30k_seed20260813_f39_frozen_v1}
export RELOCATED_PROTOCOL_DIR=${RELOCATED_PROTOCOL_DIR:-/data/zjr/proobjaverse_official_slat_protocol30k_seed20260813_source_relocated_v1}

export TRAINING_SS_EVIDENCE=${TRAINING_SS_EVIDENCE:-/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json}
export STOCK_SLAT_FREEZE=${STOCK_SLAT_FREEZE:-/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json}

CHECKPOINT_BUNDLE=${CHECKPOINT_BUNDLE:-/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1}
export SS30K_CHECKPOINT=${SS30K_CHECKPOINT:-${CHECKPOINT_BUNDLE}/ss/checkpoints/step_030000.pt}
if [[ -z ${SLAT30K_CHECKPOINT:-} ]]; then
  BUNDLED_SLAT=${CHECKPOINT_BUNDLE}/slat/checkpoints/step_030000.pt
  EXISTING_SLAT=/data/zjr/proobjaverse_official_slat_train29861_20260817_v1/F39_fresh_step150000_seed42_8gpu_strict_fix1_warmup3000_v1/checkpoints/step_030000.pt
  if [[ -s ${BUNDLED_SLAT} ]]; then
    export SLAT30K_CHECKPOINT=${BUNDLED_SLAT}
  else
    export SLAT30K_CHECKPOINT=${EXISTING_SLAT}
  fi
fi

export EVAL30K_ROOT=${EVAL30K_ROOT:-/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1}
export BRIDGE_COMPACT=${BRIDGE_COMPACT:-${EVAL30K_ROOT}/cache_bridge32_compact_v2}
export DEV_COMPACT=${DEV_COMPACT:-${EVAL30K_ROOT}/cache_dev64_compact_v2}
export BRIDGE_SS_TARGETS=${BRIDGE_SS_TARGETS:-${EVAL30K_ROOT}/ss_targets_bridge32}
export DEV_SS_TARGETS=${DEV_SS_TARGETS:-${EVAL30K_ROOT}/ss_targets_dev64}
export SS_CALIBRATION=${SS_CALIBRATION:-${EVAL30K_ROOT}/ss30k_cfg5_bridge32_compact_image_size_fix1_v2}
export SS_DEV_SHARDS=${SS_DEV_SHARDS:-${EVAL30K_ROOT}/ss30k_dev64_shards}
export SS_DEV_AGGREGATE=${SS_DEV_AGGREGATE:-${EVAL30K_ROOT}/ss30k_dev64_aggregate}
export ABC_ROOT=${ABC_ROOT:-${EVAL30K_ROOT}/abc_dev64}
export R_ROOT=${R_ROOT:-${EVAL30K_ROOT}/strict_reconviagen_dev64}
export FINAL_ROOT=${FINAL_ROOT:-${EVAL30K_ROOT}/abc_r_dev64_aggregate}

export EVAL_GPUS=${EVAL_GPUS:-4,5,7}
IFS=, read -r -a _OFFICIAL30K_EVAL_GPU_ARRAY <<<"${EVAL_GPUS}"
if [[ ${#_OFFICIAL30K_EVAL_GPU_ARRAY[@]} -lt 3 || ${#_OFFICIAL30K_EVAL_GPU_ARRAY[@]} -gt 4 ]]; then
  echo "ERROR: EVAL_GPUS must contain three or four comma-separated GPU ids" >&2
  return 90 2>/dev/null || exit 90
fi
if [[ $(printf '%s\n' "${_OFFICIAL30K_EVAL_GPU_ARRAY[@]}" | sort -u | wc -l) -ne ${#_OFFICIAL30K_EVAL_GPU_ARRAY[@]} ]]; then
  echo "ERROR: EVAL_GPUS must contain distinct GPU ids" >&2
  return 91 2>/dev/null || exit 91
fi
export EVAL_GPU_COUNT=${#_OFFICIAL30K_EVAL_GPU_ARRAY[@]}
export EVAL_GPU0=${_OFFICIAL30K_EVAL_GPU_ARRAY[0]}
export EVAL_GPU1=${_OFFICIAL30K_EVAL_GPU_ARRAY[1]}
export EVAL_GPU2=${_OFFICIAL30K_EVAL_GPU_ARRAY[2]}
if [[ ${#_OFFICIAL30K_EVAL_GPU_ARRAY[@]} -eq 4 ]]; then
  export EVAL_GPU3=${_OFFICIAL30K_EVAL_GPU_ARRAY[3]}
else
  unset EVAL_GPU3
fi
unset _OFFICIAL30K_EVAL_GPU_ARRAY
export ENCODER_RUNTIME=${ENCODER_RUNTIME:-/home/zjr/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_enc_conv3d_16l8_fp16}
export DECODER_SNAPSHOT=${DECODER_SNAPSHOT:-/home/zjr/.cache/huggingface/hub/models--Stable-X--trellis-vggt-v0-2/snapshots/647659a5ad5fbf67e22793e7b5e2cee4b30c5d13}

export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/Pixal3D:${PROJECT_ROOT}/ReconViaGen:${PROJECT_ROOT}/ReconViaGen/wheels/vggt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SPCONV_ALGO=native
export ATTN_BACKEND=flash_attn
