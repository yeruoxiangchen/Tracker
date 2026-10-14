"""MPP's original visual bridge: frozen, LoRA, or complete fine-tuning."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .question_qformer import (QUESTION_FORMAT, add_question_arguments, is_text_parameter,
                              load_text_parameters, question_sidecar, text_identity, tokenizer_identity)


class VisualLoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        for parameter in base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        update = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(inputs) + (update * self.scaling).to(inputs.dtype)


def extract_bridge_init(blip_checkpoint, destination):
    """Read the large BLIP checkpoint once; subsequent jobs load only this small file."""
    destination = Path(destination)
    if destination.exists():
        payload = torch.load(destination, map_location="cpu")
        if payload.get("format") != "mpp_visual_bridge_init_v1":
            raise ValueError("not an MPP bridge initialization: %s" % destination)
        if payload["source"] != str(Path(blip_checkpoint).resolve()):
            raise ValueError("bridge initialization source differs")
        return
    payload = torch.load(blip_checkpoint, map_location="cpu")
    state = {key: value for key, value in payload["model"].items()
             if key.startswith(("Qformer.", "ln_vision.")) or key == "query_tokens"}
    if "query_tokens" not in state or "ln_vision.weight" not in state:
        raise ValueError("BLIP checkpoint is missing visual bridge weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save({"format": "mpp_visual_bridge_init_v1", "state_dict": state,
                "source": str(Path(blip_checkpoint).resolve()),
                "vision_width": int(state["ln_vision.weight"].numel())}, temporary)
    temporary.replace(destination)


class VisualBridge(nn.Module):
    def __init__(self, init_path, bert_path, rank=8, alpha=16, trainable=False,
                 weights="", mode="auto", compute_dtype="fp32", gradient_checkpointing=False,
                 text_input="auto", question_max_len=128, text_init=""):
        super().__init__()
        from transformers import BertConfig
        from lavis.models.minigpt4qwen_models.Qformer import BertLMHeadModel

        initial = torch.load(init_path, map_location="cpu")
        if initial.get("format") != "mpp_visual_bridge_init_v1":
            raise ValueError("unsupported bridge initialization")
        saved = torch.load(weights, map_location="cpu") if weights else None
        saved_mode = None
        if saved is not None:
            formats = {"mpp_visual_bridge_adapter_v1": "lora", "mpp_visual_bridge_full_v1": "full",
                       QUESTION_FORMAT: "full"}
            saved_mode = formats.get(saved.get("format"))
            if saved_mode is None:
                raise ValueError("unsupported Q-Former adapter")
            rank, alpha = saved.get("rank", 0), saved.get("alpha", 0)
        if mode == "frozen":
            trainable = False
        if mode in ("auto", "frozen"):
            mode = saved_mode or ("frozen" if mode == "frozen" else "lora")
        if mode not in ("frozen", "lora", "full") or (saved_mode and saved_mode != mode):
            raise ValueError("Q-Former training mode does not match saved weights")
        self.mode, self.compute_dtype = mode, compute_dtype
        if text_input not in ("auto", "none", "question"):
            raise ValueError("unsupported Q-Former text input")
        saved_question = saved is not None and saved.get("format") == QUESTION_FORMAT
        self.question_input = saved_question if text_input == "auto" else text_input == "question"
        if saved_question and not self.question_input:
            raise ValueError("question checkpoint cannot be downgraded to a visual-only bridge")
        if self.question_input and mode != "full":
            raise ValueError("question input requires complete Q-Former training/weights")
        self.question_identity = None
        self.question_max_len = question_max_len
        self.rank, self.alpha = rank, alpha
        self.vision_width = initial["vision_width"]
        config = BertConfig.from_pretrained(str(bert_path))
        config.encoder_width = self.vision_width
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 32
        config.gradient_checkpointing = bool(gradient_checkpointing)
        self.Qformer = BertLMHeadModel(config)
        if not self.question_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            from transformers import BertTokenizer
            self.question_tokenizer = BertTokenizer.from_pretrained(
                str(bert_path), do_lower_case=True, local_files_only=True)
            if len(self.question_tokenizer) != config.vocab_size:
                raise ValueError("BERT vocabulary size differs from restored word embeddings")
            if saved_question:
                self.question_identity = saved["question_identity"]
                self.question_max_len = self.question_identity["max_question_tokens"]
                if text_input == "question" and question_max_len != self.question_max_len:
                    raise ValueError("saved question length differs from requested length")
                if self.question_identity["tokenizer"] != tokenizer_identity(bert_path):
                    raise ValueError("question tokenizer/config differs from the saved bridge")
                sidecar = question_sidecar(weights)
                if sidecar.is_file():
                    import json
                    if json.loads(sidecar.read_text()).get("question_identity") != self.question_identity:
                        raise ValueError("question sidecar differs from tensor checkpoint")
            else:
                self.question_identity = text_identity(bert_path, question_max_len, text_init)
        self.Qformer.cls = None
        self.query_tokens = nn.Parameter(initial["state_dict"]["query_tokens"].clone())
        expected = self.state_dict()
        initial_state = {key: initial["state_dict"][key] for key in expected if key in initial["state_dict"]}
        if self.question_input:
            if saved_question:
                if set(saved["state_dict"]) != set(expected):
                    raise ValueError("question bridge checkpoint keys differ")
                initial_state = saved["state_dict"]
            else:
                initial_state.update(load_text_parameters(bert_path, expected, text_init))
        missing = set(expected) - set(initial_state)
        if missing:
            raise ValueError("missing Q-Former initialization: %s" % sorted(missing))
        self.load_state_dict(initial_state, strict=True)
        self.initialization_report = None
        for parameter in self.parameters():
            parameter.requires_grad = False
        if mode == "lora" and rank > 0:
            targets = [(name, layer) for name, layer in self.Qformer.named_modules()
                       if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in ("query", "value")]
            for name, layer in targets:
                parent_name, leaf = name.rsplit(".", 1)
                setattr(self.Qformer.get_submodule(parent_name), leaf,
                        VisualLoRALinear(layer, rank, alpha))
        if saved is not None:
            wanted = (set(self.state_dict()) if mode == "full" else
                      {key for key in self.state_dict() if "lora_" in key or key == "query_tokens"})
            upgrade = self.question_input and not saved_question
            inherited = wanted - {key for key in wanted if is_text_parameter(key)} if upgrade else wanted
            if inherited != set(saved["state_dict"]):
                raise ValueError("Q-Former adapter keys differ")
            self.load_state_dict(saved["state_dict"], strict=False)
            if upgrade:
                exact = all(torch.equal(self.state_dict()[key], value) for key, value in saved["state_dict"].items())
                if not exact:
                    raise ValueError("trained visual bridge parameters were not inherited exactly")
                self.initialization_report = {"protocol": self.question_identity["protocol"],
                    "inherited_visual_keys": len(inherited), "inherited_visual_weights_exact": exact,
                    "text_keys_initialized": len(wanted - inherited), "question_identity": self.question_identity}
        for name, parameter in self.named_parameters():
            parameter.requires_grad = bool(trainable and (
                mode == "full" or (mode == "lora" and ("lora_" in name or name == "query_tokens"))))
        self.trainable_bridge = trainable and mode != "frozen"
        self.train(trainable)

    def train(self, mode=True):
        # A frozen bridge must reproduce the old deterministic cache path.
        return super().train(bool(mode and self.trainable_bridge))

    def forward(self, features, questions=None):
        if features.ndim != 4 or features.shape[-1] != self.vision_width:
            raise ValueError("expected [batch,frames,patches,%d] EVA features" % self.vision_width)
        batch, frames, patches, width = features.shape
        dtype = self.Qformer.bert.embeddings.LayerNorm.weight.dtype
        hidden = features.reshape(batch * frames, patches, width).to(dtype)
        queries = self.query_tokens.to(dtype).expand(batch * frames, -1, -1)
        text_kwargs = {}
        if self.question_input:
            if questions is None or len(questions) != batch or any(not isinstance(q, str) or not q.strip() for q in questions):
                raise ValueError("question Q-Former requires one nonempty original question per video")
            tokens = self.question_tokenizer(questions, padding=True, truncation=False, return_tensors="pt")
            if tokens.input_ids.shape[1] > self.question_max_len:
                raise ValueError("question exceeds Q-Former max length; refusing silent truncation")
            ids = tokens.input_ids.to(features.device).repeat_interleave(frames, dim=0)
            mask = tokens.attention_mask.to(features.device).repeat_interleave(frames, dim=0)
            text_kwargs = {"input_ids": ids, "attention_mask": torch.cat([
                torch.ones((batch * frames, 32), dtype=mask.dtype, device=features.device), mask], dim=1)}
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16,
                            enabled=features.is_cuda and self.compute_dtype == "bf16"):
            result = self.Qformer.bert(
                query_embeds=queries, encoder_hidden_states=hidden,
                encoder_attention_mask=torch.ones(hidden.shape[:-1], dtype=torch.long, device=hidden.device),
                use_cache=False, return_dict=True, is_decoder=False, **text_kwargs)
        return result.last_hidden_state[:, :32].reshape(batch, frames, 32, -1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full = self.mode in ("full", "frozen")
        payload = {"format": (QUESTION_FORMAT if self.question_input else
                    "mpp_visual_bridge_full_v1" if full else "mpp_visual_bridge_adapter_v1"), "rank": self.rank,
                    "alpha": self.alpha, "vision_width": self.vision_width,
                    "state_dict": {key: value.detach().float().cpu()
                                   for key, value in self.state_dict().items()
                                   if full or "lora_" in key or key == "query_tokens"}}
        if self.question_input:
            payload["question_identity"] = self.question_identity
        torch.save(payload, path)
        if self.question_input:
            import json
            metadata = {"format": QUESTION_FORMAT, "question_identity": self.question_identity}
            temporary = question_sidecar(path).with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(question_sidecar(path))


def add_bridge_arguments(parser):
    add_question_arguments(parser)
    parser.add_argument("--visual-checkpoint-root", default="")
    parser.add_argument("--bridge-init", default="", help="extracted BLIP Q-Former/query/LN weights")
    parser.add_argument("--qformer-adapter", default="", help="saved visual adapter including query tokens")
    parser.add_argument("--train-qformer", action="store_true")
    parser.add_argument("--qformer-r", type=int, default=8)
    parser.add_argument("--qformer-alpha", type=int, default=16)
    parser.add_argument("--qformer-lr", type=float, default=1e-5)
    parser.add_argument("--qformer-mode", choices=("auto", "frozen", "lora", "full"), default="auto")
    parser.add_argument("--qformer-gradient-checkpointing", action="store_true")
    parser.add_argument("--precision-protocol", choices=("legacy", "fp32_trainables_v1"),
                        default="legacy", help="FP32 bridge/projector/LoRA weights in training and evaluation")


def save_bridge(backbone, path):
    if hasattr(backbone, "visual_bridge"):
        backbone.visual_bridge.save(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    print("Extracting/loading BLIP visual initialization: %s" % cli.output, flush=True)
    extract_bridge_init(cli.source, cli.output)
    print("Visual initialization ready: %s" % cli.output, flush=True)
