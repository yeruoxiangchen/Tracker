#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
DETECTOR_PYTHON="${DETECTOR_PYTHON:-/home/zjr/anaconda3/envs/mpp_regions/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
SOURCE_DIR="${REGION_SOURCE_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
PROPOSAL_DIR="${REGION_PROPOSAL_DIR:-$PROJECT_ROOT/outputs/auto_region_proposals_v1}"
DATA_DIR="${REGION_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_auto_regions_40k_v1}"
FULL_CACHE="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
REGION_CACHE="${REGION_CACHE_DIR:-$PROJECT_ROOT/datasets/cache/lingoqa_selected_regions_eva_v1}"
INIT_DIR="${REGION_INIT_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2/final}"
TRAIN_ROOT="${REGION_TRAIN_ROOT:-$PROJECT_ROOT/outputs/mpp_auto_region_sft_40k_2epoch_v1}"
EVAL_DIR="${REGION_EVAL_DIR:-$PROJECT_ROOT/outputs/auto_region_eval_v1}"
DETECTOR_MODEL="${DETECTOR_MODEL:-IDEA-Research/grounding-dino-tiny}"
EPOCHS="${REGION_EPOCHS:-2}"
NPROC="${NPROC_PER_NODE:-4}"
read -r -a BRANCHES <<< "${REGION_BRANCHES:-roi}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Only the detector uses its dedicated cache. Preserve any user-supplied hub
# cache for MPP/Judge instead of replacing the old offline model cache.
DETECTOR_HUB_CACHE="${REGION_DETECTOR_HUB_CACHE:-$PROJECT_ROOT/checkpoints/detector_cache}"
common=(--upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT"
        --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT")
model_common=("${common[@]}" --feature-stage eva --precision-protocol fp32_trainables_v1
              --task-instruction-version legacy_v1 --max-txt-len 1536)
report_args=(--data-dir "$DATA_DIR" --proposal-dir "$PROPOSAL_DIR" --region-cache-dir "$REGION_CACHE"
             --cache-dir "$FULL_CACHE" --train-root "$TRAIN_ROOT" --eval-dir "$EVAL_DIR" --init-dir "$INIT_DIR"
             --branches "${BRANCHES[@]}" --epochs "$EPOCHS" --tokenizer "$CHECKPOINT_ROOT/ckpt/Qwen7B-chat"
             --bridge-init "$BRIDGE_INIT")
report() { "$PYTHON_BIN" -u scripts/report_auto_regions.py "${report_args[@]}" --action "$1"; }
weights_for() {
  local branch="$1" checkpoint="$2"
  case "$branch" in frozen|full|duplicate|roi) ;; *) echo "Unknown branch: $branch" >&2; exit 2 ;; esac
  if [[ "$branch" == frozen ]]; then
    weights=(--adapter "$INIT_DIR/adapter" --projector "$INIT_DIR/projector.pt" --qformer-adapter "$INIT_DIR/qformer_adapter.pt")
  elif [[ "$checkpoint" == final ]]; then
    weights=(--adapter "$TRAIN_ROOT/$branch/final_adapter" --projector "$TRAIN_ROOT/$branch/final_projector.pt"
             --qformer-adapter "$TRAIN_ROOT/$branch/final_qformer_adapter.pt")
  else
    weights=(--adapter "$TRAIN_ROOT/$branch/$checkpoint/adapter" --projector "$TRAIN_ROOT/$branch/$checkpoint/projector.pt"
             --qformer-adapter "$TRAIN_ROOT/$branch/$checkpoint/qformer_adapter.pt")
  fi
}
evaluate() {
  local name="$1" branch="$2" condition="$3" checkpoint="$4" split="$5"
  weights_for "$branch" "$checkpoint"
  "$PYTHON_BIN" -u scripts/generate_answers.py "${model_common[@]}" "${weights[@]}" --qformer-mode auto \
    --data "$DATA_DIR/$split.jsonl" --cache-dir "$FULL_CACHE" --region-views "$DATA_DIR/$split.views.jsonl" \
    --region-cache-dir "$REGION_CACHE" --region-condition "$condition" --output-dir "$EVAL_DIR/${name}_$split" \
    --batch-size 1 --max-new-tokens 96
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$EVAL_DIR/${name}_$split/predictions.jsonl" \
    --output-dir "$EVAL_DIR/${name}_$split/scored" --lingo-judge
}
case "${1:-help}" in
  install_detector) bash scripts/install_region_detector.sh ;;
  propose)
    extra=()
    if [[ -n "${DETECTOR_REVISION:-}" ]]; then extra+=(--revision "$DETECTOR_REVISION"); fi
    env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_CACHE="$DETECTOR_HUB_CACHE" \
      "$DETECTOR_PYTHON" -u -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29701}" \
      scripts/propose_regions.py --data "$SOURCE_DIR/train.jsonl" --data "$SOURCE_DIR/dev.jsonl" \
      --data "$SOURCE_DIR/eval.jsonl" --output-dir "$PROPOSAL_DIR" --model "$DETECTOR_MODEL" "${extra[@]}"
    ;;
  prepare)
    "$PYTHON_BIN" -u scripts/prepare_region_views.py --source-dir "$SOURCE_DIR" --proposal-dir "$PROPOSAL_DIR" \
      --output-dir "$DATA_DIR" --max-regions-per-frame "${REGION_MAX_PER_FRAME:-2}" \
      --min-score "${REGION_MIN_SCORE:-0.35}" --context-padding "${REGION_PADDING:-0.2}" \
      --full-replay-fraction 0.2 --seed 44
    ;;
  preview)
    "$PYTHON_BIN" scripts/preview_region_views.py --data "$DATA_DIR/${2:-dev}.jsonl" \
      --views "$DATA_DIR/${2:-dev}.views.jsonl" --output "$EVAL_DIR/region_preview_${2:-dev}.html" --count 20
    ;;
  cache)
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29702}" \
      scripts/cache_region_features.py "${common[@]}" --catalog "$DATA_DIR/regions.jsonl" \
      --cache-dir "$REGION_CACHE" --batch-size "${REGION_CACHE_BATCH_SIZE:-4}" --existing-policy validate
    ;;
  preflight) report preflight ;;
  status|cache_status|eval_status) report status ;;
  eval_frozen)
    for condition in full duplicate roi; do evaluate "frozen_$condition" frozen "$condition" init dev; done
    report report
    ;;
  train)
    branch="${2:-roi}"
    case "$branch" in full|duplicate|roi) ;; *) echo "train branch must be full/duplicate/roi" >&2; exit 2 ;; esac
    report preflight
    init=(--init-adapter "$INIT_DIR/adapter" --projector "$INIT_DIR/projector.pt" --qformer-adapter "$INIT_DIR/qformer_adapter.pt")
    if [[ -n "${REGION_RESUME_FROM:-}" ]]; then init=(--resume-from "$REGION_RESUME_FROM"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29703}" \
      scripts/train_sft.py "${model_common[@]}" "${init[@]}" \
      --data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" --cache-dir "$FULL_CACHE" \
      --region-views "$DATA_DIR/train.views.jsonl" --dev-region-views "$DATA_DIR/dev.views.jsonl" \
      --region-cache-dir "$REGION_CACHE" --region-condition "$branch" --output-dir "$TRAIN_ROOT/$branch" \
      --qformer-mode full --train-qformer --qformer-gradient-checkpointing --train-projector \
      --gradient-checkpointing --answer-only-logits --lr 2.5e-6 --projector-lr 2.5e-6 --qformer-lr 5e-6 \
      --epochs "$EPOCHS" --batch-size 1 --grad-accumulation 4 --seed 44 --warmup-ratio 0.03 \
      --save-every 1000 --max-dev-records 64 --max-steps "${REGION_MAX_STEPS:-0}"
    ;;
  train_pair)
    for branch in duplicate roi; do bash "$0" train "$branch"; done
    ;;
  eval_dev)
    branch="${2:-roi}"
    for ((epoch=1; epoch<=EPOCHS; epoch++)); do
      checkpoint="$(printf 'epoch-%02d' "$epoch")"
      evaluate "${branch}_$checkpoint" "$branch" "$branch" "$checkpoint" dev
    done
    report report
    ;;
  report) report report ;;
  eval_selected)
    branch="${2:-roi}"
    report report
    checkpoint="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"][sys.argv[2]]["checkpoint"])' "$EVAL_DIR/report.json" "$branch")"
    evaluate frozen_full frozen full init eval
    evaluate "${branch}_$checkpoint" "$branch" "$branch" "$checkpoint" eval
    report report
    ;;
  ask)
    branch="${2:?roi/duplicate/full/frozen}"; checkpoint="${3:?epoch-01/epoch-02/final/init}"
    weights_for "$branch" "$checkpoint"
    "$PYTHON_BIN" -u scripts/ask_region_video.py "${model_common[@]}" "${weights[@]}" --qformer-mode auto \
      --video "${4:?video path}" --question "${5:?question}" --condition "$([[ "$branch" == frozen ]] && echo roi || echo "$branch")" \
      --detector-python "$DETECTOR_PYTHON" --detector-model "$DETECTOR_MODEL" --detector-cache "$DETECTOR_HUB_CACHE" \
      --planner-manifest "$DATA_DIR/manifest.json" \
      --output-root "$EVAL_DIR/user_clips"
    ;;
  *) echo "usage: bash $0 install_detector|propose|prepare|preview [dev/train]|cache|preflight|status|eval_frozen|train [roi/duplicate/full]|train_pair|eval_dev [BRANCH]|report|eval_selected [BRANCH]|ask BRANCH CHECKPOINT VIDEO QUESTION" >&2; exit 2 ;;
esac
