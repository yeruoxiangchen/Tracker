"""Expand Qwen's trained QKV adapter without changing its initial function."""
import copy

import torch
from torch import nn


LANGUAGE_LORA_SCOPES = ("qkv", "decoder_linear")
DECODER_LINEAR_TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj")
EXPANSION_PROTOCOL = "qwen_qkv_to_decoder_linear_preserve_fp32_v1"


def scope_identity(scope):
    if scope not in LANGUAGE_LORA_SCOPES:
        raise ValueError("unknown language LoRA scope: %s" % scope)
    return {"scope": scope, "r": 16, "alpha": 32, "dropout": 0.0,
            "targets": list(DECODER_LINEAR_TARGETS if scope == "decoder_linear" else ("attn.c_attn",)),
            "initialization": EXPANSION_PROTOCOL if scope == "decoder_linear" else "load_existing_adapter"}


def decoder_target_modules(base):
    """Use full decoder paths so that embeddings and the LM head stay frozen."""
    names = [name for name, module in base.named_modules()
             if isinstance(module, nn.Linear) and
             any(name.endswith("." + suffix) for suffix in DECODER_LINEAR_TARGETS)]
    layers = getattr(base.config, "num_hidden_layers", None)
    if not names or (layers is not None and len(names) != layers * len(DECODER_LINEAR_TARGETS)):
        raise ValueError("Qwen decoder must expose five target linear layers per block")
    return sorted(names)


def expand_qkv_adapter(base, source, dropout=0.0):
    """Create one larger adapter, copy old A/B exactly, leave new B at zero.

    This is a weights-only initialization. Resume and inference should load the
    saved expanded adapter normally, including its target_modules configuration.
    """
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import (
        get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict)

    original = LoraConfig.from_pretrained(source)
    if (original.r != 16 or original.lora_alpha != 32 or
            set(original.target_modules) != {"c_attn"} or original.bias != "none" or
            original.modules_to_save or original.use_dora or original.use_rslora or
            original.rank_pattern or original.alpha_pattern):
        raise ValueError("expansion requires the original rank-16, alpha-32 QKV-only adapter")
    targets = decoder_target_modules(base)
    config = copy.deepcopy(original)
    config.target_modules = targets
    config.inference_mode = False
    config.lora_dropout = dropout
    config.init_lora_weights = True
    expanded = get_peft_model(base, config)
    # PEFT can construct adapters at the frozen base dtype. Promote before the
    # first copy, so the existing trained FP32 weights are never rounded to BF16.
    for name, parameter in expanded.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    old = load_peft_weights(source, device="cpu")
    expected = get_peft_model_state_dict(expanded, save_embedding_layers=False)
    if len(old) != 2 * sum(name.endswith(".attn.c_attn") for name in targets):
        raise ValueError("source adapter must contain both QKV LoRA matrices in every decoder block")
    for name, value in old.items():
        if (".attn.c_attn.lora_" not in name or name not in expected or
                value.shape != expected[name].shape):
            raise ValueError("source QKV adapter tensor does not match: %s" % name)
    set_peft_model_state_dict(expanded, old, adapter_name="default")
    actual = get_peft_model_state_dict(expanded, save_embedding_layers=False)
    if any(not torch.equal(actual[name].detach().cpu(), value.float()) for name, value in old.items()):
        raise RuntimeError("old FP32 QKV adapter weights were not preserved")
    new_keys = sorted(set(actual) - set(old))
    if any(torch.count_nonzero(actual[name]).item() for name in new_keys if ".lora_B." in name):
        raise RuntimeError("new LoRA modules must start with zero update")
    report = {"protocol": EXPANSION_PROTOCOL, "source_adapter": str(source),
              "r": config.r, "alpha": config.lora_alpha, "dropout": config.lora_dropout,
              "target_modules": targets, "target_module_count": len(targets),
              "inherited_tensor_count": len(old), "new_tensor_count": len(new_keys),
              "inherited_weights_exact": True, "new_B_zero": True,
              "language_trainable_parameters": sum(p.numel() for p in expanded.parameters() if p.requires_grad)}
    expanded._lora_expansion_report = report
    return expanded
