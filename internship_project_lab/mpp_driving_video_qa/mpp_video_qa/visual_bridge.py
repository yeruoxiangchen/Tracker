"""MPP's original visual bridge: frozen, LoRA, or complete fine-tuning."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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
                 weights="", mode="auto", compute_dtype="fp32", gradient_checkpointing=False):
        super().__init__()
        from transformers import BertConfig
        from lavis.models.minigpt4qwen_models.Qformer import BertLMHeadModel

        initial = torch.load(init_path, map_location="cpu")
        if initial.get("format") != "mpp_visual_bridge_init_v1":
            raise ValueError("unsupported bridge initialization")
        saved = torch.load(weights, map_location="cpu") if weights else None
        saved_mode = None
        if saved is not None:
            formats = {"mpp_visual_bridge_adapter_v1": "lora", "mpp_visual_bridge_full_v1": "full"}
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
        self.rank, self.alpha = rank, alpha
        self.vision_width = initial["vision_width"]
        config = BertConfig.from_pretrained(str(bert_path))
        config.encoder_width = self.vision_width
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 32
        config.gradient_checkpointing = bool(gradient_checkpointing)
        self.Qformer = BertLMHeadModel(config)
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None
        self.query_tokens = nn.Parameter(initial["state_dict"]["query_tokens"].clone())
        expected = self.state_dict()
        missing = set(expected) - set(initial["state_dict"])
        if missing:
            raise ValueError("missing Q-Former initialization: %s" % sorted(missing))
        self.load_state_dict({key: initial["state_dict"][key] for key in expected}, strict=True)
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
            if wanted != set(saved["state_dict"]):
                raise ValueError("Q-Former adapter keys differ")
            self.load_state_dict(saved["state_dict"], strict=False)
        for name, parameter in self.named_parameters():
            parameter.requires_grad = bool(trainable and (
                mode == "full" or (mode == "lora" and ("lora_" in name or name == "query_tokens"))))
        self.trainable_bridge = trainable and mode != "frozen"
        self.train(trainable)

    def train(self, mode=True):
        # A frozen bridge must reproduce the old deterministic cache path.
        return super().train(bool(mode and self.trainable_bridge))

    def forward(self, features):
        if features.ndim != 4 or features.shape[-1] != self.vision_width:
            raise ValueError("expected [batch,frames,patches,%d] EVA features" % self.vision_width)
        batch, frames, patches, width = features.shape
        dtype = self.Qformer.bert.embeddings.LayerNorm.weight.dtype
        hidden = features.reshape(batch * frames, patches, width).to(dtype)
        queries = self.query_tokens.to(dtype).expand(batch * frames, -1, -1)
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16,
                            enabled=features.is_cuda and self.compute_dtype == "bf16"):
            result = self.Qformer.bert(
                query_embeds=queries, encoder_hidden_states=hidden,
                encoder_attention_mask=torch.ones(hidden.shape[:-1], dtype=torch.long, device=hidden.device),
                use_cache=False, return_dict=True)
        return result.last_hidden_state[:, :32].reshape(batch, frames, 32, -1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full = self.mode in ("full", "frozen")
        torch.save({"format": "mpp_visual_bridge_full_v1" if full else "mpp_visual_bridge_adapter_v1", "rank": self.rank,
                    "alpha": self.alpha, "vision_width": self.vision_width,
                    "state_dict": {key: value.detach().float().cpu()
                                   for key, value in self.state_dict().items()
                                   if full or "lora_" in key or key == "query_tokens"}}, path)


def add_bridge_arguments(parser):
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
