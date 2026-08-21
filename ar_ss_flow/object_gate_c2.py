from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ar_ss_flow.pairwise_object_selfref_gate import (
    config_from_dict,
    contrastive_object_score,
)


C2_GATE_TABLE_VERSION = "ar_ss_flow.object_selfref_gate_table.v1"
HYPOTHESES = (
    "correct",
    "pose_cyclic1",
    "pose_cyclic2",
    "pose_reverse",
    "visual_shuffle",
)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}


def stable_sigmoid(value: float | np.ndarray, temperature: float) -> float | np.ndarray:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    data = np.asarray(value, dtype=np.float64)
    scaled = data / temperature
    output = np.empty_like(scaled)
    positive = scaled >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exp_value = np.exp(scaled[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    if np.isscalar(value):
        return float(output)
    return output


@dataclass(frozen=True)
class ObjectGateRow:
    object_uid: str
    scores: dict[str, float]
    gates: dict[str, float]
    valid_voxels: dict[str, int]


class SelfReferenceObjectGateTable:
    def __init__(
        self,
        *,
        rows: dict[str, ObjectGateRow],
        calibration: dict[str, Any],
        protocol: dict[str, Any],
    ) -> None:
        if not rows:
            raise ValueError("gate table must contain at least one object")
        self.rows = dict(rows)
        self.calibration = dict(calibration)
        self.protocol = dict(protocol)
        self.temperature = float(calibration["temperature"])
        self.config = config_from_dict(calibration["config"])

    @classmethod
    def load(
        cls,
        *,
        report_path: str | Path,
        samples_path: str | Path,
        calibration_path: str | Path,
    ) -> "SelfReferenceObjectGateTable":
        report = load_json(report_path)
        arrays = load_npz(samples_path)
        calibration = load_json(calibration_path)
        protocol = dict(report.get("protocol", {}))
        hypotheses = tuple(protocol.get("hypotheses", ()))
        if hypotheses != HYPOTHESES:
            raise ValueError(f"unexpected hypotheses={hypotheses}")
        if not bool(protocol.get("visual_only_pairwise", False)):
            raise ValueError("gate records are not visual-only pairwise")
        if not bool(protocol.get("geometry_pair_scale_forced_zero", False)):
            raise ValueError("geometry pair scale was not forced to zero")
        required = {
            "selfref_object_index",
            "selfref_confidence",
            "selfref_common_support",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise KeyError(f"selfref archive missing arrays={missing}")
        confidence = np.asarray(arrays["selfref_confidence"])
        support = np.asarray(arrays["selfref_common_support"])
        indices = np.asarray(arrays["selfref_object_index"]).reshape(-1)
        if confidence.ndim != 4:
            raise ValueError(
                "selfref_confidence must have [objects,hypotheses,variants,voxels]"
            )
        if support.ndim != 3:
            raise ValueError(
                "selfref_common_support must have [objects,hypotheses,voxels]"
            )
        if confidence.shape[:2] != support.shape[:2]:
            raise ValueError("confidence/support object-hypothesis shapes differ")
        if confidence.shape[0] != indices.size:
            raise ValueError("object index count differs from records")
        if confidence.shape[1] != len(HYPOTHESES):
            raise ValueError("hypothesis count differs from protocol")
        temperature = float(calibration["temperature"])
        config = config_from_dict(calibration["config"])
        min_valid = int(calibration.get("minimum_valid_voxels", 1))
        uid_map = report.get("object_uids", {})
        rows: dict[str, ObjectGateRow] = {}
        for row_index, object_index in enumerate(indices.tolist()):
            uid = str(uid_map.get(str(int(object_index)), ""))
            if not uid:
                raise KeyError(f"missing object uid for selfref index={object_index}")
            scores: dict[str, float] = {}
            gates: dict[str, float] = {}
            valid_voxels: dict[str, int] = {}
            for hypothesis_index, hypothesis in enumerate(HYPOTHESES):
                result = contrastive_object_score(
                    confidence[row_index, hypothesis_index],
                    support[row_index, hypothesis_index],
                    config,
                    min_valid_voxels=min_valid,
                )
                score = float(result["score"])
                gate = float(stable_sigmoid(score, temperature))
                if not np.isfinite(score) or not np.isfinite(gate):
                    raise ValueError(f"non-finite gate for uid={uid} hypothesis={hypothesis}")
                scores[hypothesis] = score
                gates[hypothesis] = gate
                valid_voxels[hypothesis] = int(result["valid_voxel_count"])
            if uid in rows:
                raise ValueError(f"duplicate gate uid={uid}")
            rows[uid] = ObjectGateRow(
                object_uid=uid,
                scores=scores,
                gates=gates,
                valid_voxels=valid_voxels,
            )
        return cls(rows=rows, calibration=calibration, protocol=protocol)

    def __contains__(self, object_uid: str) -> bool:
        return str(object_uid) in self.rows

    def gate(self, object_uid: str, hypothesis: str = "correct") -> float:
        if hypothesis not in HYPOTHESES:
            raise KeyError(f"unsupported hypothesis={hypothesis}")
        return float(self.rows[str(object_uid)].gates[hypothesis])

    def score(self, object_uid: str, hypothesis: str = "correct") -> float:
        if hypothesis not in HYPOTHESES:
            raise KeyError(f"unsupported hypothesis={hypothesis}")
        return float(self.rows[str(object_uid)].scores[hypothesis])

    def mean_gate(
        self,
        hypothesis: str = "correct",
        object_uids: Iterable[str] | None = None,
    ) -> float:
        keys = list(self.rows) if object_uids is None else [str(uid) for uid in object_uids]
        values = [self.gate(uid, hypothesis) for uid in keys if uid in self.rows]
        if not values:
            raise ValueError("no gates selected")
        return float(np.mean(values))

    def summary(self) -> dict[str, Any]:
        return {
            "format": C2_GATE_TABLE_VERSION,
            "object_count": len(self.rows),
            "config": self.config.to_dict(),
            "temperature": self.temperature,
            "gate_means": {
                hypothesis: self.mean_gate(hypothesis)
                for hypothesis in HYPOTHESES
            },
        }


def gate_tensor(
    gate: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(gate, device=reference.device, dtype=torch.float32)
    batch = int(reference.shape[0])
    if value.ndim == 0:
        value = value.expand(batch)
    if value.ndim != 1 or int(value.numel()) != batch:
        raise ValueError(f"object gate must be scalar or [B], got={tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("object gate contains non-finite values")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError("object gate must be in [0,1]")
    return value.reshape(batch, 1, 1, 1, 1)


def apply_object_gate_exact(
    stock_velocity: torch.Tensor,
    raw_delta: torch.Tensor,
    gate: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stock_velocity.shape != raw_delta.shape:
        raise ValueError("stock velocity and delta shapes differ")
    value = gate_tensor(gate, stock_velocity)
    if int(torch.count_nonzero(value).item()) == 0:
        zero = torch.zeros_like(raw_delta)
        return stock_velocity, zero
    scaled = raw_delta * value.to(dtype=raw_delta.dtype)
    return stock_velocity + scaled, scaled


def deterministic_permuted_gates(
    object_uids: Iterable[str],
    gate_by_uid: dict[str, float],
    *,
    seed: int,
) -> dict[str, float]:
    keys = sorted(str(uid) for uid in object_uids)
    values = np.asarray([float(gate_by_uid[uid]) for uid in keys], dtype=np.float64)
    if values.size < 2:
        return {uid: float(value) for uid, value in zip(keys, values)}
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(values.size)
    if np.all(permutation == np.arange(values.size)):
        permutation = np.roll(permutation, 1)
    return {uid: float(values[permutation[index]]) for index, uid in enumerate(keys)}
