"""Question-input Q-Former initialization and reproducible text contracts."""
from functools import lru_cache
import json
from pathlib import Path
import re

import torch

from .core import file_sha256

QUESTION_PROTOCOL = "mpp_question_qformer_v1"
QUESTION_FORMAT = "mpp_visual_bridge_question_full_v1"


def add_question_arguments(parser):
    parser.add_argument("--qformer-text-input", choices=("auto", "none", "question"), default="auto",
                        help="auto follows saved bridge metadata; question upgrades a full visual bridge")
    parser.add_argument("--qformer-question-max-len", type=int, default=128)
    parser.add_argument("--qformer-text-init", default="", help="BERT model file or directory; default is local bert-base-uncased")


def is_text_parameter(key):
    return (key.startswith("Qformer.bert.embeddings.word_embeddings.") or
            key.startswith("Qformer.bert.embeddings.position_embeddings.") or
            re.match(r"^Qformer\.bert\.encoder\.layer\.\d+\.(intermediate|output)\.", key) is not None)


def text_weight_file(bert_path, override=""):
    path = Path(override or bert_path)
    if path.is_file():
        return path
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("BERT text initialization weights are missing: %s" % path)


@lru_cache(maxsize=8)
def _hashed_file(path, size, mtime_ns):
    return file_sha256(path)


def cached_hash(path):
    path = Path(path).resolve()
    info = path.stat()
    return _hashed_file(str(path), info.st_size, info.st_mtime_ns)


def tokenizer_identity(bert_path):
    root = Path(bert_path)
    return {"class": "BertTokenizer", "do_lower_case": True,
            "vocab_sha256": cached_hash(root / "vocab.txt"),
            "config_sha256": cached_hash(root / "config.json")}


def text_identity(bert_path, max_len=128, override=""):
    if max_len < 2 or max_len > 512:
        raise ValueError("question length must fit BERT positions (2..512)")
    path = text_weight_file(bert_path, override)
    return {"protocol": QUESTION_PROTOCOL, "mode": "question", "max_question_tokens": max_len,
            "tokenizer": tokenizer_identity(bert_path),
            "text_initializer_sha256": cached_hash(path),
            "initialization": "inherit_trained_visual_keys_fill_text_from_bert_v1",
            "attention": "bidirectional_query_and_question", "visual_tokens_per_frame": 32}


def normalize_bert_key(key):
    if key.startswith("module."):
        key = key[len("module."):]
    if not key.startswith("Qformer."):
        key = "Qformer." + key
    if key.endswith(".gamma"):
        key = key[:-len("gamma")] + "weight"
    elif key.endswith(".beta"):
        key = key[:-len("beta")] + "bias"
    return key


def load_text_parameters(bert_path, expected, override=""):
    """Read only restored text modules; never overwrite trained query/visual parameters."""
    wanted = {key for key in expected if is_text_parameter(key)}
    if not wanted:
        raise ValueError("question Q-Former did not create text modules")
    source = text_weight_file(bert_path, override)
    result = {}

    def accept(key, tensor):
        normalized = normalize_bert_key(key)
        if normalized not in wanted:
            return
        if normalized in result:
            raise ValueError("duplicate normalized BERT key: %s" % normalized)
        if tensor.shape != expected[normalized].shape:
            raise ValueError("BERT text shape mismatch: %s" % normalized)
        result[normalized] = tensor.detach().float().clone()

    if source.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if normalize_bert_key(key) in wanted:
                    accept(key, handle.get_tensor(key))
    else:
        payload = torch.load(source, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload))
        for key, tensor in state.items():
            if normalize_bert_key(key) in wanted:
                accept(key, tensor)
    missing = wanted - set(result)
    if missing:
        raise ValueError("BERT text initialization is incomplete: %s" % sorted(missing))
    return result


def question_sidecar(path):
    return Path(path).with_suffix(".question.json")


def question_identity_from_args(args):
    """Light metadata lets generation identify auto-loaded question checkpoints."""
    requested = getattr(args, "qformer_text_input", "auto")
    weights = getattr(args, "qformer_adapter", "")
    sidecar = question_sidecar(weights) if weights else None
    if sidecar and sidecar.is_file():
        value = json.loads(sidecar.read_text())
        if value.get("format") != QUESTION_FORMAT:
            raise ValueError("invalid question bridge metadata")
        if requested == "none":
            raise ValueError("question checkpoint cannot run with text input disabled")
        identity = value["question_identity"]
        if requested == "question" and identity["max_question_tokens"] != args.qformer_question_max_len:
            raise ValueError("saved question length differs from the requested contract")
        return identity
    if requested != "question":
        return None
    root = Path(getattr(args, "visual_checkpoint_root", "") or args.checkpoint_root) / "ckpt/bert-base-uncased"
    return text_identity(root, getattr(args, "qformer_question_max_len", 128),
                         getattr(args, "qformer_text_init", ""))
