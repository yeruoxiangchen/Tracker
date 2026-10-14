#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
CACHE_DIR="${LINGO_QFORMER_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1}"
DATA_DIR="${TARGETED_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_action_targeted_v1}"
OUTPUT_DIR="${TARGETED_OUTPUT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_action_targeted_v1}"
JOINT_DIR="$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1"
DIAG_DIR="$PROJECT_ROOT/outputs/action_targeted_diagnosis_v1"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1

model_args() {
  weights=()
  case "$1" in
    base) ;;
    old_lora) weights=(--adapter "$PROJECT_ROOT/outputs/mpp_lingoqa_full_v2_lora_1epoch/final_adapter") ;;
    joint) weights=(--adapter "$JOINT_DIR/final_adapter" --projector "$JOINT_DIR/final_projector.pt") ;;
    targeted) weights=(--adapter "$OUTPUT_DIR/final_adapter" --projector "$OUTPUT_DIR/final_projector.pt") ;;
    *) echo "unknown model: $1" >&2; exit 2 ;;
  esac
}

generate_and_score() {
  "$PYTHON_BIN" -u scripts/generate_answers.py \
    --data "$1" --output-dir "$2" --cache-dir "$CACHE_DIR" \
    --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --feature-stage qformer --task-instruction-version legacy_v1 \
    --batch-size 1 --max-new-tokens 96 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py \
    --predictions "$2/predictions.jsonl" --output-dir "$2/scored" --lingo-judge
}

case "${1:-help}" in
  prepare)
    "$PYTHON_BIN" scripts/prepare_action_targeted_sft.py --output-dir "$DATA_DIR"
    "$PYTHON_BIN" scripts/validate_full_v2.py --data-dir "$DATA_DIR" \
      --cache-dir "$CACHE_DIR" --feature-stage qformer
    "$PYTHON_BIN" scripts/diagnose_action_targeted.py prepare
    ;;
  diagnose)
    model="${2:?specify base, old_lora, joint, or targeted}"
    model_args "$model"
    generate_and_score "$DIAG_DIR/diagnostic.jsonl" "$DIAG_DIR/$model"
    ;;
  diagnostic_report)
    shift
    models=(base old_lora joint)
    if [ "$#" -gt 0 ]; then models=("$@"); fi
    "$PYTHON_BIN" scripts/diagnose_action_targeted.py report --models "${models[@]}"
    "$PYTHON_BIN" scripts/diagnose_action_targeted.py review
    ;;
  train)
    "$PYTHON_BIN" scripts/validate_full_v2.py --data-dir "$DATA_DIR" \
      --cache-dir "$CACHE_DIR" --feature-stage qformer
    init=(--init-adapter "$JOINT_DIR/final_adapter" --projector "$JOINT_DIR/final_projector.pt")
    if [ -n "${SFT_RESUME_FROM:-}" ]; then init=(--resume-from "$SFT_RESUME_FROM"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE:-4}" --master_port="${MASTER_PORT:-29661}" \
      scripts/train_sft.py --data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" \
      --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
      --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
      --feature-stage qformer --train-projector --gradient-checkpointing \
      --epochs "${SFT_EPOCHS:-1}" --batch-size 1 --grad-accumulation 4 \
      --lr "${SFT_LR:-5e-6}" --projector-lr "${MPP_PROJECTOR_LR:-5e-6}" \
      --save-every 200 --max-steps "${SFT_MAX_STEPS:-0}" "${init[@]}"
    ;;
  eval)
    split="${2:?specify action, scenery, or eval}"
    case "$split" in
      action|scenery) data="$DATA_DIR/dev_${split}.jsonl" ;;
      eval) data="$DATA_DIR/eval.jsonl" ;;
      *) exit 2 ;;
    esac
    model_args targeted
    generate_and_score "$data" "$PROJECT_ROOT/outputs/projector_sft_eval/targeted_${split}"
    "$PYTHON_BIN" scripts/compare_runs.py \
      --baseline "outputs/projector_sft_eval/joint_${split}/scored/scored_predictions.jsonl" \
      --candidate "outputs/projector_sft_eval/targeted_${split}/scored/scored_predictions.jsonl" \
      --output "outputs/projector_sft_eval/joint_vs_targeted_${split}.json"
    ;;
  status)
    "$PYTHON_BIN" scripts/report_progress.py --output-dir "$OUTPUT_DIR"
    ;;
  *) echo "usage: $0 prepare|diagnose MODEL|diagnostic_report [MODELS...]|train|eval SPLIT|status" >&2; exit 2 ;;
esac
