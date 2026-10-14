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
DATA_DIR="${FULL_QFORMER_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_qformer_full100k_v1}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
JOINT_DIR="${JOINT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1}"
TRAIN_DIR="${FULL_QFORMER_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_full100k_2epoch_v1}"
EVAL_DIR="${FULL_QFORMER_EVAL_DIR:-$PROJECT_ROOT/outputs/full_qformer_eval_v1}"
EPOCHS="${SFT_EPOCHS:-2}"
NPROC="${NPROC_PER_NODE:-4}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
common=(--upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT"
        --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva)
model_common=("${common[@]}" --precision-protocol fp32_trainables_v1 --task-instruction-version legacy_v1)
weights_for() {
  local model="$1"
  case "$model" in
    joint) weights=(--adapter "$JOINT_DIR/final_adapter" --projector "$JOINT_DIR/final_projector.pt" --qformer-mode frozen) ;;
    epoch-*|checkpoint-*) weights=(--adapter "$TRAIN_DIR/$model/adapter" --projector "$TRAIN_DIR/$model/projector.pt"
                     --qformer-adapter "$TRAIN_DIR/$model/qformer_adapter.pt" --qformer-mode auto) ;;
    final) weights=(--adapter "$TRAIN_DIR/final_adapter" --projector "$TRAIN_DIR/final_projector.pt"
                   --qformer-adapter "$TRAIN_DIR/final_qformer_adapter.pt" --qformer-mode auto) ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
  esac
}
evaluate() {
  local model="$1" split="$2" data_path destination
  case "$split" in
    action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
    eval) data_path="$DATA_DIR/eval.jsonl" ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  destination="$EVAL_DIR/${model}_${split}"
  weights_for "$model"
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --cache-dir "$CACHE_DIR" \
    --output-dir "$destination" "${model_common[@]}" "${weights[@]}" --batch-size 1 --max-new-tokens 96
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
}
case "${1:-help}" in
  prepare)
    "$PYTHON_BIN" -u scripts/prepare_full_qformer_sft.py --output-dir "$DATA_DIR" --train-size "${SFT_TRAIN_SIZE:-100000}"
    "$PYTHON_BIN" -u -m mpp_video_qa.visual_bridge --source "$VISUAL_ROOT/ckpt/blip2/blip2_pretrained_flant5xxl.pth" --output "$BRIDGE_INIT"
    ;;
  cache)
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29684}" \
      scripts/cache_visual_features.py "${common[@]}" --view full --cache-dir "$CACHE_DIR" \
      --data "$DATA_DIR/train.jsonl" --data "$DATA_DIR/dev.jsonl" --data "$DATA_DIR/eval.jsonl"
    ;;
  cache_status) "$PYTHON_BIN" scripts/report_cache_progress.py --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" ;;
  smoke)
    "$PYTHON_BIN" scripts/prepare_full_qformer_sft.py --output-dir "$DATA_DIR" --train-size "${SFT_TRAIN_SIZE:-100000}" \
      --smoke-cache "$CACHE_DIR" --smoke-output "$PROJECT_ROOT/datasets/processed/lingoqa_qformer_full_smoke_v1"
    FULL_QFORMER_DATA_DIR="$PROJECT_ROOT/datasets/processed/lingoqa_qformer_full_smoke_v1" \
    FULL_QFORMER_TRAIN_DIR="${FULL_QFORMER_SMOKE_DIR:-$PROJECT_ROOT/outputs/full_qformer_smoke_4gpu_v1}" \
    SFT_MAX_STEPS=2 SAVE_EVERY=1 MAX_DEV_RECORDS=0 bash "$0" train
    ;;
  train)
    init=(--init-adapter "${SFT_INIT_ADAPTER:-$JOINT_DIR/final_adapter}" --projector "${SFT_INIT_PROJECTOR:-$JOINT_DIR/final_projector.pt}")
    if [[ -n "${SFT_INIT_QFORMER:-}" ]]; then init+=(--qformer-adapter "$SFT_INIT_QFORMER"); fi
    if [[ -n "${SFT_RESUME_FROM:-}" ]]; then init=(--resume-from "$SFT_RESUME_FROM"); fi
    provenance=()
    if [[ -n "${SFT_PARENT_RUN:-}" ]]; then provenance=(--parent-run "$SFT_PARENT_RUN"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29685}" \
      scripts/train_sft.py "${model_common[@]}" "${init[@]}" "${provenance[@]}" \
      --data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" --cache-dir "$CACHE_DIR" --output-dir "$TRAIN_DIR" \
      --qformer-mode "${QFORMER_MODE:-full}" --train-qformer --qformer-gradient-checkpointing --train-projector \
      --qformer-lr "${QFORMER_LR:-1e-5}" --projector-lr "${PROJECTOR_LR:-5e-6}" --lr "${SFT_LR:-5e-6}" \
      --epochs "$EPOCHS" --batch-size 1 --grad-accumulation "${GRAD_ACCUMULATION:-4}" \
      --seed "${SFT_SEED:-42}" --warmup-ratio "${SFT_WARMUP_RATIO:-0.03}" \
      --gradient-checkpointing --max-dev-records "${MAX_DEV_RECORDS:-64}" --save-every "${SAVE_EVERY:-2000}" --max-steps "${SFT_MAX_STEPS:-0}"
    ;;
  status) "$PYTHON_BIN" scripts/report_progress.py --output-dir "$TRAIN_DIR" ;;
  eval) evaluate "${2:?model: joint|epoch-01|epoch-02|final}" "${3:?split: action|scenery|eval}" ;;
  eval_dev)
    for split in action scenery; do evaluate joint "$split"; done
    for ((epoch=1; epoch<=EPOCHS; epoch++)); do
      model="$(printf 'epoch-%02d' "$epoch")"
      for split in action scenery; do evaluate "$model" "$split"; done
    done
    bash "$0" report
    ;;
  report)
    models=(joint)
    for ((epoch=1; epoch<=EPOCHS; epoch++)); do models+=("$(printf 'epoch-%02d' "$epoch")"); done
    "$PYTHON_BIN" scripts/report_full_qformer.py --data-dir "$DATA_DIR" --eval-dir "$EVAL_DIR" \
      --train-dir "$TRAIN_DIR" --joint-dir "$JOINT_DIR" --models "${models[@]}"
    ;;
  eval_selected)
    selected="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_model"])' "$EVAL_DIR/report.json")"
    evaluate joint eval
    if [[ "$selected" != joint ]]; then
      evaluate "$selected" eval
      "$PYTHON_BIN" scripts/compare_runs.py --baseline "$EVAL_DIR/joint_eval/scored/scored_predictions.jsonl" \
        --candidate "$EVAL_DIR/${selected}_eval/scored/scored_predictions.jsonl" \
        --output "$EVAL_DIR/joint_vs_${selected}_eval.json"
    fi
    ;;
  ask)
    weights_for "${2:?model: joint|epoch-01|epoch-02|final}"
    "$PYTHON_BIN" scripts/ask_video.py --video "${3:?video path}" --question "${4:?question}" \
      --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" --visual-checkpoint-root "$VISUAL_ROOT" \
      --bridge-init "$BRIDGE_INIT" --precision-protocol fp32_trainables_v1 --task-instruction-version legacy_v1 "${weights[@]}"
    ;;
  eval_status)
    "$PYTHON_BIN" -c 'import json,sys; from pathlib import Path
root=Path(sys.argv[1])
for folder in sorted(root.glob("*_*")):
    if not folder.is_dir(): continue
    pred=folder/"predictions.jsonl"
    count=sum(1 for _ in pred.open()) if pred.exists() else 0
    print(folder.name, count, "scored" if (folder/"scored/summary.json").exists() else "pending")' "$EVAL_DIR"
    ;;
  *) echo "usage: $0 prepare|cache|cache_status|smoke|train|status|eval MODEL SPLIT|eval_dev|report|eval_selected|eval_status|ask MODEL VIDEO QUESTION" >&2; exit 2 ;;
esac
