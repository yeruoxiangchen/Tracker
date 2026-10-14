"""Offline preference training on MPP's existing five-frame Q-Former cache."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .core import file_sha256, load_jsonl, stable_key
from .data import CachedQADataset, cache_file
from .opd import completion_logits


PREFERENCE_PROTOCOL = "mpp_fact_preference_v1"
REFERENCE_PROTOCOL = "mpp_joint_completion_sum_logp_eos_v1"


def row_identity(row):
    keys = ("question_id", "segment_id", "question", "frames", "answers", "kind",
            "chosen", "rejected", "source_subset")
    return stable_key(json.dumps({k: row.get(k) for k in keys}, sort_keys=True, ensure_ascii=False))


def tokenizer_identity(tokenizer):
    vocab = sorted((repr(k), int(v)) for k, v in tokenizer.get_vocab().items())
    return stable_key(json.dumps(vocab), tokenizer.im_end_id, tokenizer.pad_token_id)


def encode_completion(tokenizer, text):
    if not str(text).strip() or "<|" in str(text):
        raise ValueError("completion must be nonempty plain assistant text")
    ids = list(tokenizer(str(text)).input_ids)
    if not ids or tokenizer.im_end_id in ids:
        raise ValueError("invalid completion tokens")
    return ids + [tokenizer.im_end_id]


def sum_token_logps(logits, targets, chunk_tokens=8):
    """Sum answer log probabilities in fp32, including EOS; no prompt or pad."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.numel() or not targets.numel():
        raise ValueError("expected aligned nonempty [answer_tokens,vocab] and [answer_tokens]")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_tokens):
        x, y = logits[start:start + chunk_tokens], targets[start:start + chunk_tokens]
        def score(a, b):
            return -F.cross_entropy(a.float(), b, reduction="sum")
        if torch.is_grad_enabled() and x.requires_grad:
            total = total + checkpoint(score, x, y, use_reentrant=False)
        else:
            total = total + score(x, y)
    return total


def completion_logprob(model, features, question, token_ids, chunk_tokens=8):
    targets = torch.as_tensor(token_ids, device=features.device, dtype=torch.long)
    logits = completion_logits(model, features, question, targets)
    return sum_token_logps(logits, targets, chunk_tokens)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    ref_chosen = torch.as_tensor(ref_chosen, device=chosen.device, dtype=chosen.dtype).detach()
    ref_rejected = torch.as_tensor(ref_rejected, device=chosen.device, dtype=chosen.dtype).detach()
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margin).mean(), margin.detach()


class PreferenceDataset(CachedQADataset):
    def __init__(self, path, cache_dir):
        super().__init__(path, cache_dir, feature_stage="qformer")
        ids = [r["example_id"] for r in self.rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("preference examples must be nonempty and unique")
        for row in self.rows:
            if (row.get("preference_protocol") != PREFERENCE_PROTOCOL or
                    row.get("training_source") != "train" or len(row["frames"]) != 5 or
                    row.get("kind") not in ("preference", "replay")):
                raise ValueError("expected train-only five-frame preference/replay rows")
            if row.get("row_sha256") != row_identity(row):
                raise ValueError("preference row identity mismatch")

    def __getitem__(self, index):
        result = super().__getitem__(index)
        row = result["row"]
        payload = torch.load(cache_file(self.cache_dir, row["segment_id"]), map_location="cpu")
        if payload.get("view", "full") != "full" or payload.get("frame_paths") != row["frames"]:
            raise ValueError("cache must contain the selected original full frames")
        return result


def cache_inventory(rows, cache_dir):
    """Hash only selected visual files, once per segment (not the whole cache)."""
    return {segment: file_sha256(cache_file(cache_dir, segment))
            for segment in sorted({r["segment_id"] for r in rows})}


def read_reference_bundle(folder, data_path):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if (manifest.get("protocol") != REFERENCE_PROTOCOL or not summary.get("completed") or
            manifest["data_sha256"] != file_sha256(data_path) or
            summary["reference_sha256"] != file_sha256(folder / "reference.jsonl")):
        raise ValueError("reference cache incomplete or belongs to different data")
    records = load_jsonl(folder / "reference.jsonl")
    refs = {r["example_id"]: r for r in records}
    rows = load_jsonl(data_path)
    if len(refs) != len(records) or set(refs) != {r["example_id"] for r in rows}:
        raise ValueError("reference cache example coverage differs")
    for row in rows:
        if refs[row["example_id"]]["row_sha256"] != row_identity(row):
            raise ValueError("reference response identity mismatch")
    return manifest, summary, refs
