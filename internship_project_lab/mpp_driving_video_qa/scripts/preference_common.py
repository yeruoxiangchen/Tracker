"""Shared construction/identity helpers for offline preference runs."""
import json
from pathlib import Path

import torch

from mpp_video_qa.core import TASK_INSTRUCTIONS, file_sha256, stable_key
from mpp_video_qa.model import CachedVideoQA, build_text_model, mpp_config, load_projector
from mpp_video_qa.opd import disable_dropout
from scripts.generate_answers import directory_fingerprint


def add_model_arguments(parser):
    for name in ("data", "cache-dir", "upstream-root", "checkpoint-root", "init-adapter", "projector"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-txt-len", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=8)
    parser.set_defaults(feature_stage="qformer", train_projector=True, train_qformer=False,
                        dtype="bf16", gradient_checkpointing=True, lora_r=16, lora_alpha=32,
                        lora_dropout=0.0, lora_target_modules=["c_attn"], visual_checkpoint_root="")


def initialization_identity(args):
    config = mpp_config(args)
    checkpoint = Path(config["finetuned"])
    checkpoint_index = checkpoint / "model.safetensors.index.json" if checkpoint.is_dir() else checkpoint
    # Shard index binds checkpoint layout; weight files are the installed official base.
    return {"adapter": directory_fingerprint(args.init_adapter),
            "projector": file_sha256(args.projector), "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_index_sha256": file_sha256(checkpoint_index),
            "qwen_config_sha256": file_sha256(Path(config["llm_model"]) / "config.json"),
            "feature_stage": "qformer", "view": "full", "task_instruction_version": "legacy_v1",
            "task_instruction_sha256": stable_key(TASK_INSTRUCTIONS["legacy_v1"]),
            "max_txt_len": args.max_txt_len, "chunk_tokens": args.chunk_tokens,
            "trainable_precision": "fp32_adapter_and_projector_from_saved_weights"}


def validate_selection(args, rows):
    manifest = json.loads((Path(args.data).parent / "manifest.json").read_text())
    if manifest["data_sha256"] != file_sha256(args.data) or manifest["pairs"] < 1:
        raise ValueError("empty or modified prepared preference dataset")
    identity = initialization_identity(args)
    generation = manifest["generation_identity"]
    if any(generation.get(k) != identity[k] for k in ("adapter", "projector", "feature_stage", "view", "task_instruction_version")):
        raise ValueError("initial joint differs from the student used to construct preference pairs")
    if len(rows) != manifest["records"]:
        raise ValueError("selection record count differs")
    return manifest, identity


def make_model(args, train, device, adapter=None, projector=None):
    model = CachedVideoQA(
        build_text_model(args, train=train, adapter=adapter or args.init_adapter,
                         projector=projector or args.projector),
        task_instruction=TASK_INSTRUCTIONS["legacy_v1"], feature_stage="qformer").to(device)
    if train:
        # build_text_model casts the backbone through bf16 before upcasting trainable
        # parameters. Reload the small fp32 projector so resume keeps AdamW updates.
        load_projector(model.backbone, projector or args.projector)
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        # PEFT creates LoRA layers at the base dtype during initial construction.
        # Reload after build_text_model's fp32 promotion to avoid rounding a
        # resumed adapter through bf16 as well.
        state = load_peft_weights(adapter or args.init_adapter, device="cpu")
        set_peft_model_state_dict(model.backbone.llm_model, state, adapter_name="default")
    disable_dropout(model)
    model.train(train)
    return model
