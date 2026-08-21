#!/usr/bin/env bash
# Source this file before any numbered source-server command.  It deliberately
# does not change the caller's `set -e/-u/pipefail` state.

WITH_VGGT_PROJECT_ROOT=${WITH_VGGT_PROJECT_ROOT:-/home/zjr/Tracker}
cd "${WITH_VGGT_PROJECT_ROOT}"

source /home/zjr/anaconda3/etc/profile.d/conda.sh
conda activate reconviagen

export PYTHONPATH="$PWD:$PWD/ReconViaGen:$PWD/ReconViaGen/wheels/vggt"
export WITH_VGGT_STRICT_FIX1_ROOT="$PWD/a72_perf_v1_fix1_testcompat1/Tracker"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SPCONV_ALGO=native
export ATTN_BACKEND=flash_attn

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
TORCHRUN=/home/zjr/anaconda3/envs/reconviagen/bin/torchrun

ROOT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1
SPLIT=${ROOT}/protocol2128_train2000_v1/train.json
BASE=${ROOT}/cache_train2000_protocol2128_views8_v1

DECODER=${ROOT}/decoder_audit32_protocol2128_v1/report.json
TRAINING_SS_REPORT=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

OUT8=${ROOT}/cache_train8_protocol2128_views8_with_vggt_sidecar_v1
OUT64=${ROOT}/cache_train64_protocol2128_views8_with_vggt_sidecar_v1
SMOKE=${ROOT}/V_with_vggt_train64_step2_seed42_1gpu_strict_perf_v1_smoke_v1
FULL=${ROOT}/cache_train2000_protocol2128_views8_with_vggt_sidecar_v1

