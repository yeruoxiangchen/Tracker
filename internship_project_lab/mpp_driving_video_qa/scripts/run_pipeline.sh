#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
ASSET_ROOT="${MPP_ASSET_ROOT:-/home/zjr/Tracker/internship_project_lab/mpp_llava_video_workspace}"
UPSTREAM_ROOT="${MPP_UPSTREAM_ROOT:-${ASSET_ROOT}/third_party/MPP-LLaVA}"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-${ASSET_ROOT}/checkpoints}"
DATA_DIR="${LINGO_PROCESSED_DIR:-${PROJECT_ROOT}/datasets/processed/lingoqa_v1}"
CACHE_DIR="${LINGO_CACHE_DIR:-${PROJECT_ROOT}/datasets/cache/lingoqa_5f_mpp}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mpp_lingoqa_lora_v1}"
FEATURE_STAGE="${MPP_FEATURE_STAGE:-projected}"
TRAIN_PROJECTOR="${MPP_TRAIN_PROJECTOR:-0}"
ACTION="${1:-help}"

common=(--upstream-root "$UPSTREAM_ROOT" --checkpoint-root "$CHECKPOINT_ROOT")

case "$ACTION" in
  cache)
    cache_extra=()
    if [[ -n "${MPP_VISUAL_CHECKPOINT_ROOT:-}" ]]; then
      cache_extra+=(--visual-checkpoint-root "$MPP_VISUAL_CHECKPOINT_ROOT")
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      PYTHONPATH="$PROJECT_ROOT:$UPSTREAM_ROOT" "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node=4 --master_port="${MASTER_PORT:-29631}" \
      "$PROJECT_ROOT/scripts/cache_visual_features.py" \
      --data "$DATA_DIR/train.jsonl" --data "$DATA_DIR/dev.jsonl" \
      --data "$DATA_DIR/eval.jsonl" --cache-dir "$CACHE_DIR" \
      --feature-stage "$FEATURE_STAGE" "${cache_extra[@]}" "${common[@]}"
    ;;
  train)
    train_extra=()
    if [[ -n "${SFT_MAX_STEPS:-}" ]]; then
      train_extra+=(--max-steps "$SFT_MAX_STEPS")
    fi
    if [[ "${SFT_GRADIENT_CHECKPOINTING:-1}" == "0" ]]; then
      train_extra+=(--no-gradient-checkpointing)
    else
      train_extra+=(--gradient-checkpointing)
    fi
    if [[ -n "${SFT_RESUME_FROM:-}" ]]; then
      train_extra+=(--resume-from "$SFT_RESUME_FROM")
    fi
    if [[ "$TRAIN_PROJECTOR" == "1" ]]; then
      train_extra+=(--train-projector)
    fi
    if [[ -n "${MPP_PROJECTOR_LR:-}" ]]; then
      train_extra+=(--projector-lr "$MPP_PROJECTOR_LR")
    fi
    if [[ -n "${MPP_PROJECTOR_INIT:-}" ]]; then
      train_extra+=(--projector "$MPP_PROJECTOR_INIT")
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
      PYTHONPATH="$PROJECT_ROOT:$UPSTREAM_ROOT" "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node=4 --master_port="${MASTER_PORT:-29632}" \
      "$PROJECT_ROOT/scripts/train_sft.py" --data "$DATA_DIR/train.jsonl" \
      --dev-data "$DATA_DIR/dev.jsonl" --cache-dir "$CACHE_DIR" \
      --output-dir "$OUTPUT_DIR" --epochs "${SFT_EPOCHS:-3}" \
      --grad-accumulation "${SFT_GRAD_ACCUMULATION:-4}" \
      --max-dev-records "${SFT_MAX_DEV_RECORDS:-256}" \
      --save-every "${SFT_SAVE_EVERY:-0}" \
      --feature-stage "$FEATURE_STAGE" \
      --task-instruction-version "${TASK_INSTRUCTION_VERSION:-legacy_v1}" \
      "${train_extra[@]}" "${common[@]}"
    ;;
  eval_base|eval_sft)
    name="base"; adapter=()
    if [[ "$ACTION" == "eval_sft" ]]; then
      name="sft"; adapter=(--adapter "$OUTPUT_DIR/final_adapter")
    fi
    projector=()
    if [[ "$ACTION" == "eval_sft" && "$FEATURE_STAGE" == "qformer" ]]; then
      projector=(--projector "$OUTPUT_DIR/final_projector.pt")
    fi
    eval_data="${EVAL_DATA:-$DATA_DIR/eval.jsonl}"
    eval_output="${EVAL_OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_${name}_${EVAL_RUN_TAG:-v1}}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$PROJECT_ROOT:$UPSTREAM_ROOT" \
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/generate_answers.py" \
      --data "$eval_data" --cache-dir "$CACHE_DIR" \
      --output-dir "$eval_output" \
      --feature-stage "$FEATURE_STAGE" \
      --task-instruction-version "${TASK_INSTRUCTION_VERSION:-legacy_v1}" \
      "${adapter[@]}" "${projector[@]}" "${common[@]}"
    ;;
  score_base|score_sft)
    name="${ACTION#score_}"
    score_input="${EVAL_OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_${name}_${EVAL_RUN_TAG:-v1}}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$PROJECT_ROOT" \
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_answers.py" \
      --predictions "$score_input/predictions.jsonl" \
      --output-dir "$score_input/scored" --lingo-judge
    ;;
  *)
    echo "usage: $0 {cache|train|eval_base|eval_sft|score_base|score_sft}" >&2
    exit 2
    ;;
esac
