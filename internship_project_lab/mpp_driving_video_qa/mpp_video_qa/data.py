from pathlib import Path

import torch
from torch.utils.data import Dataset

from .core import PROTOCOL, TASK_INSTRUCTION, load_jsonl, stable_key
from .model import FEATURE_STAGE_PROJECTED, FEATURE_STAGE_QFORMER, FEATURE_STAGE_EVA, FEATURE_STAGES


def cache_file(cache_dir, segment_id):
    key = stable_key(segment_id)
    return Path(cache_dir) / key[:2] / (key + ".pt")


def training_answer(row, target_mode="reference"):
    """Generated targets never replace the original references used for evaluation."""
    if target_mode == "reference":
        return row["answers"][0]
    if target_mode != "distillation":
        raise ValueError("unknown target mode: %s" % target_mode)
    if row.get("distillation_source") != "train":
        raise ValueError("distillation targets must come from training data")
    if row.get("target_source") == "teacher":
        answer = row.get("teacher_answer", "").strip()
        if not answer:
            raise ValueError("missing teacher answer")
        return answer
    if row.get("target_source") != "reference_replay":
        raise ValueError("unknown distillation target source")
    return row["answers"][0]


class CachedQADataset(Dataset):
    def __init__(self, data_path, cache_dir, feature_stage=FEATURE_STAGE_PROJECTED,
                 target_mode="reference", validate_full_frames=False,
                 region_views="", region_cache_dir="", region_condition="roi",
                 task_instruction=TASK_INSTRUCTION):
        self.rows = load_jsonl(data_path)
        self.target_mode = target_mode
        self.validate_full_frames = validate_full_frames or bool(region_views)
        for row in self.rows:
            training_answer(row, target_mode)
        self.cache_dir = Path(cache_dir)
        if feature_stage not in FEATURE_STAGES:
            raise ValueError("unknown feature stage: %s" % feature_stage)
        self.feature_stage = feature_stage
        self.region_condition = region_condition
        self.region_cache_dir = Path(region_cache_dir) if region_cache_dir else None
        self.task_instruction = task_instruction
        self.plans = None
        self.region_encoder_identity = None
        if region_views:
            import json
            from .region_views import CONDITIONS, validate_plan
            if feature_stage != FEATURE_STAGE_EVA or region_condition not in CONDITIONS:
                raise ValueError("automatic regions require EVA features and a valid condition")
            plans = load_jsonl(region_views)
            self.plans = {p["question_id"]: p for p in plans}
            if len(plans) != len(self.plans) or set(self.plans) != {r["question_id"] for r in self.rows}:
                raise ValueError("view plans must cover each dataset question exactly once")
            for row in self.rows:
                validate_plan(self.plans[row["question_id"]], row)
            if region_condition == "roi":
                if self.region_cache_dir is None:
                    raise ValueError("ROI views require a region EVA cache")
                manifest = json.loads((self.region_cache_dir / "cache_manifest.json").read_text())
                self.region_encoder_identity = manifest["encoder_identity"]

    def _load_region(self, view):
        from .region_views import REGION_FEATURE_PROTOCOL, region_file
        path = region_file(self.region_cache_dir, view["region_key"])
        payload = torch.load(path, map_location="cpu")
        if (payload.get("protocol") != REGION_FEATURE_PROTOCOL or
                payload.get("region_key") != view["region_key"] or
                payload.get("pixel_box") != view["pixel_box"] or
                payload.get("image_sha256") != view["image_sha256"] or
                payload.get("encoder_identity") != self.region_encoder_identity):
            raise RuntimeError("stale or mismatched region EVA feature: %s" % path)
        features = payload["features"]
        if features.shape != (257, 1408) or features.dtype != torch.bfloat16:
            raise RuntimeError("region must contain frozen BF16 EVA patches [257,1408]")
        return features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = cache_file(self.cache_dir, row["segment_id"])
        payload = torch.load(path, map_location="cpu")
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError("visual cache protocol mismatch: %s" % path)
        if payload.get("segment_id") != row["segment_id"]:
            raise RuntimeError("visual cache segment mismatch: %s" % path)
        features = payload["features"]
        if self.validate_full_frames and (payload.get("view", "full") != "full" or
                payload.get("frame_paths") != row["frames"] or features.shape[0] != len(row["frames"])):
            raise RuntimeError("cache does not contain this sample's original full frames: %s" % path)
        if features.ndim != 3 or (self.feature_stage != FEATURE_STAGE_EVA and features.shape[1] != 32):
            raise RuntimeError("unexpected cached feature shape %s" %
                               (tuple(features.shape),))
        # Old caches predate this field and are unambiguously 4096-dimensional.
        cache_stage = payload.get("feature_stage")
        if cache_stage is None and features.shape[-1] == 4096:
            cache_stage = FEATURE_STAGE_PROJECTED
        expected_hidden = {FEATURE_STAGE_QFORMER: 768, FEATURE_STAGE_PROJECTED: 4096,
                           FEATURE_STAGE_EVA: 1408}[self.feature_stage]
        if cache_stage != self.feature_stage or features.shape[-1] != expected_hidden:
            raise RuntimeError(
                "visual cache stage mismatch for %s: stage=%s hidden=%d, expected %s/%d" %
                (path, cache_stage, features.shape[-1], self.feature_stage, expected_hidden))
        result = {"row": row, "features": features, "answer": training_answer(row, self.target_mode)}
        if self.plans is not None:
            from .region_views import compose_features, view_prompt
            plan = self.plans[row["question_id"]]
            result["features"] = compose_features(features, plan, self.region_condition, self._load_region)
            result["prompt"] = view_prompt(row["question"], plan, self.region_condition, self.task_instruction)
            result["region_metadata"] = {"plan_sha256": plan["plan_sha256"],
                                         "condition": self.region_condition,
                                         "views": len(result["features"]),
                                         "region_slots": sum(v["kind"] == "region" for v in plan["views"])
                                         if self.region_condition != "full" else 0}
            if "evidence_search" in plan:
                result["region_metadata"]["evidence_search"] = plan["evidence_search"]
        return result


def qa_collate(batch):
    shapes = {tuple(item["features"].shape) for item in batch}
    if len(shapes) != 1:
        raise RuntimeError("all cached feature shapes in a batch must match: %s" % shapes)
    result = {
        "rows": [item["row"] for item in batch],
        "features": torch.stack([item["features"] for item in batch]),
        "answers": [item["answer"] for item in batch],
    }
    if any("prompt" in item for item in batch):
        if not all("prompt" in item for item in batch):
            raise ValueError("cannot mix legacy and region observations in one batch")
        result["prompts"] = [item["prompt"] for item in batch]
        result["region_metadata"] = [item["region_metadata"] for item in batch]
    return result
