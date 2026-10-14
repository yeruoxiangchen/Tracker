#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
SOURCE_DIR="${FEEDBACK_SOURCE_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_qformer_all_v1}"
DATA_DIR="${FEEDBACK_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_v1}"
INITIAL_DIR="${FEEDBACK_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
PROBE_DIR="${FEEDBACK_PROBE_DIR:-$PROJECT_ROOT/outputs/feedback_opd_probe_v1}"
TRAIN_DIR="${FEEDBACK_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_opd_10k_v1}"
CONTROL_DIR="${FEEDBACK_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_10k_v1}"
EVAL_DIR="${FEEDBACK_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
common=(--cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT"
        --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT"
        --initial-dir "$INITIAL_DIR" --task-instruction-version legacy_v1
        --seed "${FEEDBACK_SEED:-44}" --max-new-tokens "${FEEDBACK_MAX_NEW_TOKENS:-96}"
        --teacher-mode "${FEEDBACK_TEACHER_MODE:-reference_draft}"
        --max-txt-len "${FEEDBACK_MAX_TXT_LEN:-1536}" --judge-device "${FEEDBACK_JUDGE_DEVICE:-cpu}")
report_args=(--data-dir "$DATA_DIR" --initial-dir "$INITIAL_DIR" --probe-dir "$PROBE_DIR"
             --train-dir "$TRAIN_DIR" --control-dir "$CONTROL_DIR" --eval-dir "$EVAL_DIR")
report() { "$PYTHON_BIN" scripts/report_feedback_opd.py "${report_args[@]}" --action "$1"; }
evaluate() {
  local model="$1" split="$2" data_path destination
  case "$split" in
    action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
    eval) data_path="$DATA_DIR/eval.jsonl" ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  case "$model" in
    baseline) weights=(--adapter "$INITIAL_DIR/final_adapter" --projector "$INITIAL_DIR/final_projector.pt"
                       --qformer-adapter "$INITIAL_DIR/final_qformer_adapter.pt") ;;
    sft_control) weights=(--adapter "$CONTROL_DIR/final/adapter" --projector "$CONTROL_DIR/final/projector.pt"
                          --qformer-adapter "$CONTROL_DIR/final/qformer_adapter.pt") ;;
    feedback_opd) weights=(--adapter "$TRAIN_DIR/final/adapter" --projector "$TRAIN_DIR/final/projector.pt"
                           --qformer-adapter "$TRAIN_DIR/final/qformer_adapter.pt") ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
  esac
  destination="$EVAL_DIR/${model}_${split}"
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --output-dir "$destination" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva \
    --precision-protocol fp32_trainables_v1 --qformer-mode auto --task-instruction-version legacy_v1 \
    --max-new-tokens "${FEEDBACK_MAX_NEW_TOKENS:-96}" --batch-size 1 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
}
case "${1:-help}" in
  prepare)
    "$PYTHON_BIN" -u scripts/prepare_feedback_opd.py --source-dir "$SOURCE_DIR" --output-dir "$DATA_DIR" \
      --cache-dir "$CACHE_DIR" --train-size "${FEEDBACK_TRAIN_SIZE:-10000}" \
      --probe-size "${FEEDBACK_PROBE_SIZE:-2000}" --seed "${FEEDBACK_SEED:-44}" \
      --max-qa-per-segment "${FEEDBACK_MAX_QA_PER_SEGMENT:-2}"
    ;;
  probe)
    "$PYTHON_BIN" -u scripts/probe_feedback_opd.py --data "$DATA_DIR/probe.jsonl" --output-dir "$PROBE_DIR" \
      "${common[@]}" --num-samples "${FEEDBACK_PROBE_SAMPLES:-2}" --token-cases "${FEEDBACK_TOKEN_CASES:-40}"
    ;;
  smoke)
    "$PYTHON_BIN" -u scripts/smoke_feedback_opd.py --data "$DATA_DIR/probe.jsonl" \
      --output-dir "${FEEDBACK_SMOKE_DIR:-$PROJECT_ROOT/outputs/feedback_opd_real_smoke_v1}" "${common[@]}"
    ;;
  train_control|train_opd)
    mode=feedback_opd; destination="$TRAIN_DIR"
    if [[ "$1" == train_control ]]; then mode=sft_control; destination="$CONTROL_DIR"; fi
    extra=()
    if [[ -n "${FEEDBACK_CHECKPOINT_STEPS:-}" ]]; then
      read -r -a selected_steps <<< "$FEEDBACK_CHECKPOINT_STEPS"
      extra+=(--checkpoint-steps "${selected_steps[@]}")
    fi
    if [[ -n "${FEEDBACK_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$FEEDBACK_RESUME_FROM"); fi
    if [[ "${FEEDBACK_GLOBAL_BATCH_SIZE:-0}" != 0 ]]; then
      extra+=(--global-batch-size "$FEEDBACK_GLOBAL_BATCH_SIZE")
    fi
    if [[ "${FEEDBACK_FAST:-0}" == 1 ]]; then
      extra+=(--answer-only-logits --gradient-bucket-mb "${FEEDBACK_BUCKET_MB:-25}"
              --profile-every "${FEEDBACK_PROFILE_EVERY:-50}"
              --runtime-log-flush-every 4)
      if [[ "$mode" == feedback_opd ]]; then
        extra+=(--teacher-gate-cache-dir "${FEEDBACK_GATE_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/feedback_teacher_gate_v2}"
                --runtime-judge-device "${FEEDBACK_RUNTIME_JUDGE_DEVICE:-cpu}"
                --judge-equivalence-report "${FEEDBACK_JUDGE_EQUIVALENCE_REPORT:-$PROJECT_ROOT/outputs/feedback_opd_speed_validation_v1/judge.json}")
      fi
    fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-4}" \
      --master_port="${MASTER_PORT:-29696}" scripts/train_feedback_opd.py \
      --data "$DATA_DIR/train.jsonl" --output-dir "$destination" "${common[@]}" \
      --training-mode "$mode" --probe-report "${FEEDBACK_PROBE_REPORT:-$PROBE_DIR/probe_report.json}" \
      --epochs "${FEEDBACK_EPOCHS:-1}" --max-steps "${FEEDBACK_MAX_STEPS:-0}" \
      --grad-accumulation "${GRAD_ACCUMULATION:-4}" --save-every "${FEEDBACK_SAVE_EVERY:-100}" \
      --lr "${FEEDBACK_LR:-2.5e-6}" --projector-lr "${FEEDBACK_PROJECTOR_LR:-2.5e-6}" \
      --qformer-lr "${FEEDBACK_QFORMER_LR:-5e-6}" --ce-weight "${FEEDBACK_CE_WEIGHT:-1}" \
      --opd-weight "${FEEDBACK_OPD_WEIGHT:-1}" "${extra[@]}"
    ;;
  eval) evaluate "${2:?specify baseline|sft_control|feedback_opd}" "${3:?specify action|scenery|eval}" ;;
  eval_dev)
    for model in baseline sft_control feedback_opd; do
      for split in action scenery; do evaluate "$model" "$split"; done
    done
    report report
    ;;
  eval_selected)
    report report
    selected="$(report selected)"
    evaluate baseline eval
    if [[ "$selected" != baseline ]]; then evaluate "$selected" eval; fi
    report report_eval
    ;;
  report) report report ;;
  status) report status ;;
  *) echo "usage: bash scripts/run_feedback_opd.sh prepare|smoke|probe|train_control|train_opd|eval MODEL SPLIT|eval_dev|eval_selected|report|status" ;;
esac
