"""Regression using the locally installed Qwen implementation and real PEFT."""
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

from mpp_video_qa.model import (FEATURE_STAGE_QFORMER, CachedVideoQA,
                                load_projector, qwen_core, save_projector)


def check_qwen_lora_forward_backward_and_reload(tmp_path):
    source = Path(os.environ.get("QWEN_TEST_MODEL", "/tmp/zjr_mpp_fast/checkpoints/ckpt/Qwen7B-chat"))
    if not (source / "config.json").exists():
        raise unittest.SkipTest("set QWEN_TEST_MODEL to local Qwen config and modeling files")
    config = AutoConfig.from_pretrained(str(source), trust_remote_code=True,
                                        local_files_only=True)
    config.update(dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=2, kv_channels=16, vocab_size=128,
                       bf16=False, fp16=False, fp32=True, use_flash_attn=False))
    base = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    assert base.base_model is base.transformer  # The old traversal went too far.
    assert qwen_core(base) is base
    wrapped = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                              target_modules=["c_attn"], task_type="CAUSAL_LM"))
    assert qwen_core(wrapped) is base
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    assert base.transformer.gradient_checkpointing

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm_model = wrapped
            self.replace_image_token_id = 3
            self.llm_tokenizer = type("Tokenizer", (), {
                "im_end_id": 2,
                "eod_id": 0,
                "decode": staticmethod(
                    lambda row, skip_special_tokens=True:
                    " ".join(str(int(token)) for token in row)),
            })()

        def preprocess(self, prompts, answers, image_len, for_generation=False):
            ids = (torch.tensor([[1, 3, 3, 4]]) if for_generation else
                   torch.tensor([[1, 3, 3, 4, 5, 6]]))
            if for_generation:
                return dict(input_ids=ids, labels=torch.full_like(ids, -100),
                            attention_mask=torch.ones_like(ids))
            return dict(input_ids=ids, labels=torch.tensor([[-100]*4+[5, 6]]),
                        attention_mask=torch.ones_like(ids))

    model = CachedVideoQA(Backbone())
    parameter = next(p for n, p in wrapped.named_parameters() if "lora_B" in n)
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = model(torch.randn(1, 1, 2, 32), ["What happened?"], ["Braking"]).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert parameter.grad is not None and parameter.grad.abs().max() > 0
    optimizer.step()
    assert not torch.equal(before, parameter)
    generated = model.generate(
        torch.randn(1, 1, 2, 32), ["What happened?"], max_new_tokens=2)
    assert len(generated) == 1
    # The four generation-prompt ids must not leak into the decoded answer.
    assert len(generated[0].split()) <= 2
    wrapped.save_pretrained(tmp_path / "adapter")
    fresh = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    reloaded = PeftModel.from_pretrained(fresh, tmp_path / "adapter")
    assert qwen_core(reloaded) is fresh
    actual = next(p for n, p in reloaded.named_parameters() if "lora_B" in n)
    torch.testing.assert_close(actual, parameter)

    # The joint path must propagate answer loss through the official-style
    # 768->LLM projection and persist it independently from the PEFT adapter.
    projection = nn.Linear(8, 32)
    for item in projection.parameters():
        item.requires_grad = True
    projected_backbone = Backbone()
    projected_backbone.llm_proj = projection
    joint = CachedVideoQA(projected_backbone, feature_stage=FEATURE_STAGE_QFORMER)
    joint.zero_grad(set_to_none=True)
    joint_loss = joint(torch.randn(1, 1, 2, 8), ["What happened?"], ["Braking"]).loss
    joint_loss.backward()
    assert projection.weight.grad is not None and projection.weight.grad.abs().max() > 0
    before_reload = projection.weight.detach().clone()
    save_projector(projected_backbone, tmp_path / "projector.pt")
    with torch.no_grad():
        projection.weight.zero_()
    load_projector(projected_backbone, tmp_path / "projector.pt")
    torch.testing.assert_close(projection.weight, before_reload)


class QwenCoreTest(unittest.TestCase):
    def test_lora_forward_backward_and_reload(self):
        with tempfile.TemporaryDirectory(prefix="qwen-core-test-") as root:
            check_qwen_lora_forward_backward_and_reload(Path(root))


if __name__ == "__main__":
    unittest.main()
