"""Strict with-VGGT VSS -> V endpoint bindings for predicted-support tests.

This module deliberately contains only the pieces which differ from the
already-audited official predicted-support evaluator.  Noise construction,
sampling order, mesh decoding, surface metrics and bootstrap aggregation stay
in :mod:`pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

from official_slat_with_vggt_perf_v1.dataset import (
    StrictWithVGGTNativeConditionSLatDataset,
)
from official_ss_with_vggt_perf_v1.cache import (
    MODEL_CONTEXT_CONTRACT,
    WithVGGTOfficialSSDataset,
)
from official_ss_with_vggt_perf_v1.model import (
    EVAL_AGGREGATE_FORMAT,
    VERSION as VSS_CHECKPOINT_FORMAT,
    build_components as build_vss_components,
    validate_checkpoint as validate_vss_checkpoint,
)
from pose_point_depth_mv.dino_only_condition import (
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.evaluate_native_ss_genrecon import sampling_params
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import (
    make_sampling_namespace,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    _validate_ss_evidence_domain,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    load_trainable_state_dict as load_slat_trainable_state_dict,
)
from pose_point_depth_mv.native_slat_genrecon_with_vggt_official import (
    NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
    OFFICIAL_WITH_VGGT_SLAT_CONTRACT,
    build_native_slat_official_with_vggt_components,
    validate_native_slat_official_with_vggt_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import (
    canonical_json_sha256,
    load_trainable_state_dict as load_ss_trainable_state_dict,
    require_disjoint_object_uids,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256
from pose_point_depth_mv.slat_checkpoint_evaluation_membership import (
    audit_checkpoint_evaluation_membership,
)
from pose_point_depth_mv.proobjaverse_official_ss import (
    validate_official_ss_domain_contract,
)


JOINT_ENDPOINT_VERSION = "official_ss_with_vggt_perf_v1.predicted_ss_slat_endpoint.v2"
WORKER_FORMAT = f"{JOINT_ENDPOINT_VERSION}.worker"
REPORT_FORMAT = f"{JOINT_ENDPOINT_VERSION}.aggregate"

TARGET_JOIN_CONTRACT = {
    "version": "official_ss_with_vggt_perf_v1.official_target_join.v2",
    "shared_source_identity": "exact_official_lh_slat_sha256",
    "slat_metric_target": "raw_official_lh_slat_coords_and_features",
    "ss_flow_target": "frozen_ss_decoder_projected_coords",
    "exact_cross_target_coordinate_equality_required": False,
    "observed_cross_target_iou_must_match_frozen_ss_roundtrip_iou": True,
    "slat_runtime_target_preserved": True,
}

_ACTIVE_SS_CACHE_MANIFEST: Path | None = None


def activate_ss_cache_manifest(path: str | Path) -> Path:
    """Bind the explicit SS sidecar used by the base evaluator's dataset hook."""

    global _ACTIVE_SS_CACHE_MANIFEST
    _ACTIVE_SS_CACHE_MANIFEST = Path(path).expanduser().resolve(strict=True)
    return _ACTIVE_SS_CACHE_MANIFEST


def _tensor_exact(label: str, left: Any, right: Any, *, uid: str) -> None:
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        raise RuntimeError(f"uid={uid} joint endpoint lacks tensor {label}")
    if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(
        left, right
    ):
        raise RuntimeError(f"uid={uid} SS/SLat {label} differs")


def _canonical_coords(value: torch.Tensor) -> list[tuple[int, int, int]]:
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] not in (3, 4):
        raise RuntimeError("official target coordinates have an invalid shape")
    xyz = value[:, -3:].to(torch.int64).cpu().tolist()
    return sorted(tuple(int(component) for component in row) for row in xyz)


def _require_sha256(value: Any, *, label: str, uid: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"uid={uid} {label} is not a lowercase SHA256")
    return text


def _validate_official_source_pair(
    slat_row: dict[str, Any],
    ss_row: dict[str, Any],
    *,
    minimum_roundtrip_iou: float,
) -> dict[str, Any]:
    """Bind two different target domains to the same frozen LH-SLat source.

    Native-SLat consumes the raw official sparse coordinates and features.
    Native-SS deliberately trains on coordinates projected back through its
    frozen encoder/decoder.  Those supports can differ slightly, so equality
    is required for their source content hash rather than for the two derived
    coordinate tensors.
    """

    uid = str(slat_row.get("uid", ""))
    if (
        not uid
        or uid != str(ss_row.get("uid", ""))
        or str(slat_row.get("object_uid", ""))
        != str(ss_row.get("object_uid", ""))
    ):
        raise RuntimeError(f"uid={uid} official target source identity differs")
    target_sha = _require_sha256(
        slat_row.get("target_file_sha256"),
        label="SLat target SHA256",
        uid=uid,
    )
    slat_source_sha = _require_sha256(
        slat_row.get("source_lh_slat_sha256"),
        label="SLat source SHA256",
        uid=uid,
    )
    ss_source_sha = _require_sha256(
        ss_row.get("official_lh_slat_sha256"),
        label="SS source SHA256",
        uid=uid,
    )
    if len({target_sha, slat_source_sha, ss_source_sha}) != 1:
        raise RuntimeError(f"uid={uid} SS/SLat official LH-SLat SHA256 differs")
    roundtrip_iou = float(ss_row.get("official_ss_roundtrip_iou", float("nan")))
    if (
        not math.isfinite(roundtrip_iou)
        or not 0.0 <= roundtrip_iou <= 1.0
        or roundtrip_iou < minimum_roundtrip_iou
    ):
        raise RuntimeError(f"uid={uid} frozen SS round-trip IoU is invalid")
    return {
        "uid": uid,
        "object_uid": str(slat_row["object_uid"]),
        "official_lh_slat_sha256": target_sha,
        "frozen_ss_roundtrip_iou": roundtrip_iou,
    }


def _validate_runtime_target_relation(
    slat_coords: torch.Tensor,
    ss_coords: torch.Tensor,
    *,
    binding: dict[str, Any],
) -> dict[str, Any]:
    uid = str(binding["uid"])
    slat_values = _canonical_coords(slat_coords)
    ss_values = _canonical_coords(ss_coords)
    slat_set = set(slat_values)
    ss_set = set(ss_values)
    if len(slat_set) != len(slat_values) or len(ss_set) != len(ss_values):
        raise RuntimeError(f"uid={uid} official target support contains duplicates")
    intersection = len(slat_set.intersection(ss_set))
    union = len(slat_set.union(ss_set))
    if union <= 0:
        raise RuntimeError(f"uid={uid} official target support is empty")
    observed_iou = intersection / union
    expected_iou = float(binding["frozen_ss_roundtrip_iou"])
    if not math.isclose(observed_iou, expected_iou, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            f"uid={uid} SS decoder-projected support no longer matches its "
            f"frozen round-trip audit: observed={observed_iou:.17g} "
            f"expected={expected_iou:.17g}"
        )
    return {
        "contract": TARGET_JOIN_CONTRACT["version"],
        "official_lh_slat_sha256": str(binding["official_lh_slat_sha256"]),
        "slat_raw_support_count": len(slat_set),
        "ss_decoder_projected_support_count": len(ss_set),
        "intersection_count": intersection,
        "union_count": union,
        "roundtrip_iou": observed_iou,
        "supports_exact": slat_values == ss_values,
        "runtime_target_source": "slat_raw_official_lh_slat",
    }


class WithVGGTSSSLatEndpointDataset:
    """Join frozen with-VGGT SS and SLat sidecars by exact sample identity.

    The returned SLat sample keeps its native ``slat_vggt_cond`` condition.
    Only ``lifting_sample.stock_condition`` is replaced by the corresponding
    native ``ss_vggt_cond`` so the unchanged base evaluator can run VSS0/VSS.
    The DINO/K/T tensors used by both posed branches must be bit-exact.
    """

    slat_dataset_factory: Callable[..., Any] = StrictWithVGGTNativeConditionSLatDataset
    ss_dataset_factory: Callable[..., Any] = WithVGGTOfficialSSDataset

    def __init__(
        self,
        slat_manifest: str | Path,
        lifting_manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
        ss_cache_manifest: str | Path | None = None,
    ) -> None:
        manifest = ss_cache_manifest or _ACTIVE_SS_CACHE_MANIFEST
        if manifest is None:
            raise RuntimeError("with-VGGT endpoint SS cache manifest was not activated")
        self.slat_endpoint = self.slat_dataset_factory(
            slat_manifest,
            lifting_manifest,
            indices=indices,
            verify_hashes=verify_hashes,
        )
        self.ss_endpoint = self.ss_dataset_factory(manifest, indices="all")
        ss_by_identity: dict[tuple[str, str], int] = {}
        for position, row in enumerate(self.ss_endpoint.rows):
            identity = (str(row.get("uid", "")), str(row.get("object_uid", "")))
            if not all(identity) or identity in ss_by_identity:
                raise RuntimeError(f"with-VGGT SS cache duplicate identity={identity}")
            ss_by_identity[identity] = position
        identities = [
            (str(row.get("uid", "")), str(row.get("object_uid", "")))
            for row in self.slat_endpoint.rows
        ]
        missing = [identity for identity in identities if identity not in ss_by_identity]
        if missing:
            raise RuntimeError(
                f"with-VGGT SS/SLat UID join incomplete: {missing[:8]} count={len(missing)}"
            )
        self.ss_indices = [ss_by_identity[identity] for identity in identities]
        self.rows = self.slat_endpoint.rows
        self.config = self.slat_endpoint.config
        self.config_hash = self.slat_endpoint.config_hash
        self.slat_normalization = self.slat_endpoint.slat_normalization
        self.slat_normalization_hash = self.slat_endpoint.slat_normalization_hash
        self.lifting = self.slat_endpoint.lifting
        self.slat = self.slat_endpoint.slat
        self.pair_identity = self.slat_endpoint.pair_identity
        self.endpoint_identity = {
            "version": JOINT_ENDPOINT_VERSION,
            "slat_pair_identity": str(self.slat_endpoint.pair_identity),
            "slat_sidecar_contract_hash": str(
                self.slat_endpoint.sidecar_contract_hash
            ),
            "ss_pair_identity": str(self.ss_endpoint.pair_identity),
            "ss_sidecar_contract_hash": str(self.ss_endpoint.sidecar_contract_hash),
            "ss_cache_manifest": str(Path(manifest).expanduser().resolve()),
            "ss_cache_manifest_sha256": sha256_file(manifest),
            "object_count": len(self.rows),
            "ordered_identity_sha256": canonical_sha256(identities),
            "slat_support_input": "predicted_only",
            "gt_support_used_as_slat_input": False,
            "official_target_join_contract": copy.deepcopy(TARGET_JOIN_CONTRACT),
        }
        ss_protocol = self.ss_endpoint.sidecar_contract.get("protocol_sha256")
        slat_protocol = self.config.get("target_source", {}).get("protocol_sha256")
        if not ss_protocol or ss_protocol != slat_protocol:
            raise RuntimeError("with-VGGT SS/SLat official protocols differ")
        ss_target_binding = self.ss_endpoint.config.get("official_ss_targets")
        ss_domain = (
            ss_target_binding.get("domain_contract")
            if isinstance(ss_target_binding, dict)
            else None
        )
        if not isinstance(ss_domain, dict):
            raise RuntimeError("with-VGGT SS cache lacks its official target domain")
        validate_official_ss_domain_contract(ss_domain)
        if str(ss_domain.get("official_slat_protocol_sha256", "")) != slat_protocol:
            raise RuntimeError("with-VGGT SS target-domain protocol differs")
        minimum_roundtrip_iou = float(ss_domain["minimum_roundtrip_iou"])
        self.target_source_bindings = [
            _validate_official_source_pair(
                self.slat_endpoint.rows[position],
                self.ss_endpoint.rows[ss_position],
                minimum_roundtrip_iou=minimum_roundtrip_iou,
            )
            for position, ss_position in enumerate(self.ss_indices)
        ]
        self.endpoint_identity["official_target_source_identity_sha256"] = (
            canonical_sha256(self.target_source_bindings)
        )
        self.endpoint_identity["minimum_ss_roundtrip_iou"] = minimum_roundtrip_iou

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        slat_sample = self.slat_endpoint[index]
        ss_sample = self.ss_endpoint[self.ss_indices[index]]
        uid = str(slat_sample.get("uid", ""))
        if (
            uid != str(ss_sample.get("uid", ""))
            or str(slat_sample.get("object_uid", ""))
            != str(ss_sample.get("object_uid", ""))
        ):
            raise RuntimeError(f"uid={uid} with-VGGT runtime join changed")
        lifting = dict(slat_sample["lifting_sample"])
        for key in ("view_ids", "visual_patch_features", "intrinsics", "extrinsics"):
            _tensor_exact(key, lifting.get(key), ss_sample.get(key), uid=uid)
        target_relation = _validate_runtime_target_relation(
            slat_sample["target_coords"],
            ss_sample["target_coords"],
            binding=self.target_source_bindings[index],
        )
        lifting["stock_condition"] = ss_sample["stock_condition"]
        lifting["stock_condition_source"] = "native_ss_vggt_cond_sidecar"
        lifting["with_vggt_ss_sidecar_path"] = ss_sample[
            "with_vggt_sidecar_path"
        ]
        return {
            **slat_sample,
            "lifting_sample": lifting,
            "with_vggt_joint_endpoint": copy.deepcopy(self.endpoint_identity),
            "with_vggt_target_join": target_relation,
        }


def official_target_contract(dataset: WithVGGTSSSLatEndpointDataset) -> dict[str, Any]:
    target = dict(dataset.config.get("target_source", {}))
    if target.get("support_policy") != "official_gt_slat_coordinates":
        raise RuntimeError("cache is not bound to the official target protocol")
    split = str(target.get("split", ""))
    if split not in {"train", "dev"}:
        raise RuntimeError(f"unsupported official endpoint split={split!r}")
    if int(target.get("coordinate_resolution", -1)) != 64:
        raise RuntimeError("official target coordinates are not on the 64^3 grid")
    if not str(target.get("protocol_sha256", "")):
        raise RuntimeError("official target protocol hash is missing")
    validate_dino_only_lifting_contract(dataset.lifting)
    target["evaluation_role"] = (
        "training_overlap_fit_diagnosis"
        if split == "train"
        else "held_out_development_generalization"
    )
    target["slat_support_input"] = "predicted_only"
    target["gt_support_used_as_slat_input"] = False
    return target


def load_vss_deployment(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a frozen VSS aggregate without conflating science and integrity.

    A scientifically negative SS result must still be evaluable downstream.
    Therefore the four runtime-integrity gates are mandatory, while other false
    gates are recorded in ``false_checks`` instead of silently rejecting the
    checkpoint.
    """

    report_path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("with-VGGT SS report root must be an object")
    body = dict(payload)
    saved_hash = str(body.pop("report_sha256", ""))
    if (
        payload.get("format") != EVAL_AGGREGATE_FORMAT
        or payload.get("formal") is not False
        or not saved_hash
        or canonical_json_sha256(body) != saved_hash
    ):
        raise ValueError("with-VGGT SS aggregate identity/hash differs")
    checks = payload.get("checks")
    deployment = payload.get("deployment")
    object_uids = payload.get("object_uids")
    domain = payload.get("official_ss_domain_contract")
    if not isinstance(checks, dict) or not isinstance(deployment, dict):
        raise RuntimeError("with-VGGT SS aggregate lacks checks/deployment")
    integrity_gates = (
        "correct_record_matrix_exact",
        "pose_control_record_matrix_exact",
        "stock_baseline_nonempty",
        "disabled_stock_equivalence",
    )
    false_integrity = [key for key in integrity_gates if checks.get(key) is not True]
    if false_integrity:
        raise RuntimeError(
            f"with-VGGT SS aggregate failed runtime integrity={false_integrity}"
        )
    if (
        not isinstance(object_uids, list)
        or len(object_uids) != int(payload.get("object_count", -1))
        or len(object_uids) != len(set(str(value) for value in object_uids))
        or canonical_json_sha256(sorted(str(value) for value in object_uids))
        != str(payload.get("object_uid_hash", ""))
        or not isinstance(domain, dict)
    ):
        raise RuntimeError("with-VGGT SS aggregate object/domain identity differs")
    validate_official_ss_domain_contract(domain)
    required = {
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "weights",
        "cfg_strength",
        "steps",
        "cfg_interval",
        "guidance_rescale",
        "rescale_t",
        "amp_dtype",
    }
    if set(deployment) != required:
        raise ValueError("with-VGGT SS deployment fields differ")
    checkpoint = Path(str(deployment["checkpoint"])).expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != str(deployment["checkpoint_sha256"]):
        raise RuntimeError("with-VGGT SS checkpoint hash differs")
    cfg_interval = [float(value) for value in deployment["cfg_interval"]]
    false_checks = sorted(key for key, value in checks.items() if value is not True)
    binding = {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(deployment["checkpoint_sha256"]),
        "checkpoint_step": int(deployment["checkpoint_step"]),
        "weights": str(deployment["weights"]),
        "cfg_strength": float(deployment["cfg_strength"]),
        "steps": int(deployment["steps"]),
        "cfg_interval": cfg_interval,
        "guidance_rescale": float(deployment["guidance_rescale"]),
        "rescale_t": float(deployment["rescale_t"]),
        "amp_dtype": str(deployment["amp_dtype"]),
        "science_passed": payload.get("passed") is True,
        "false_checks": false_checks,
    }
    if (
        binding["checkpoint_step"] <= 0
        or binding["steps"] <= 0
        or binding["weights"] != "ema"
        or binding["cfg_strength"] <= 0.0
        or len(cfg_interval) != 2
        or not 0.0 <= cfg_interval[0] <= cfg_interval[1] <= 1.0
        or binding["guidance_rescale"] != 0.0
        or binding["rescale_t"] <= 0.0
        or binding["amp_dtype"] != "bf16"
    ):
        raise ValueError("with-VGGT SS deployment semantics are invalid")
    return payload, binding


def load_ss_runtime(
    args: Any,
    dataset: WithVGGTSSSLatEndpointDataset,
    selected_indices: list[int],
    target_contract: dict[str, Any],
    device: torch.device,
):
    report, binding = load_vss_deployment(args.native_ss_report)
    _validate_ss_evidence_domain(
        report,
        target_contract=target_contract,
        pretrained=str(args.pretrained),
    )
    if str(args.weights) != binding["weights"] or str(args.amp_dtype) != binding[
        "amp_dtype"
    ]:
        raise RuntimeError("worker weights/AMP differ from frozen VSS deployment")
    checkpoint = torch.load(binding["checkpoint"], map_location="cpu", weights_only=False)
    validate_vss_checkpoint(checkpoint, pretrained=args.pretrained)
    if (
        checkpoint.get("format") != VSS_CHECKPOINT_FORMAT
        or int(checkpoint.get("step", -1)) != int(binding["checkpoint_step"])
    ):
        raise RuntimeError("VSS checkpoint format/step differs from deployment")
    training_uids = checkpoint.get("data_identity", {}).get("object_uids")
    if not isinstance(training_uids, list):
        raise RuntimeError("VSS checkpoint lacks training object identities")
    selected_uids = {
        str(dataset.rows[index]["object_uid"]) for index in selected_indices
    }
    split = str(target_contract["split"])
    if split == "train":
        if not selected_uids or not selected_uids.issubset(
            {str(value) for value in training_uids}
        ):
            raise RuntimeError("Train64 is not a subset of VSS training identities")
    else:
        evidence_uids = {str(value) for value in report["object_uids"]}
        if not selected_uids or not selected_uids.issubset(evidence_uids):
            raise RuntimeError("Dev objects are not covered by frozen VSS evidence")
        require_disjoint_object_uids(selected_uids, training_uids)
    saved = checkpoint["args"]
    sampler, model, decoder, summary, defaults = build_vss_components(
        pretrained=args.pretrained,
        lora_rank=int(saved["lora_rank"]),
        lora_alpha=int(saved["lora_alpha"]),
        condition_channels=int(saved["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("with-VGGT endpoint requires the frozen SS decoder")
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_ss_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    params = sampling_params(
        defaults,
        make_sampling_namespace(binding),
        float(binding["cfg_strength"]),
    )
    binding = {
        **binding,
        "evaluation_split": split,
        "training_overlap_diagnosis": split == "train",
    }
    return binding, checkpoint, sampler, model, decoder, summary, params


def build_trained_slat_pipeline(
    *,
    checkpoint_path: str | Path,
    weights: str,
    pretrained: str,
    stock_freeze: dict[str, Any],
    dataset: WithVGGTSSSLatEndpointDataset,
    expected_step: int,
    device: torch.device,
    evaluation_object_uids: list[str],
    allow_target_protocol_mismatch: bool,
    expected_training_membership: str,
    checkpoint_payload: dict[str, Any] | None = None,
):
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = (
        torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint_payload is None
        else checkpoint_payload
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("with-VGGT SLat checkpoint payload is not a dictionary")
    if (
        checkpoint.get("format") != NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION
        or int(checkpoint.get("step", -1)) != int(expected_step)
    ):
        raise RuntimeError("with-VGGT SLat checkpoint format/step differs")
    saved_upstream = checkpoint.get("data_identity", {}).get("native_ss")
    summary_upstream = checkpoint.get("model_summary", {}).get("upstream_native_ss")
    if not isinstance(saved_upstream, dict) or saved_upstream != summary_upstream:
        raise RuntimeError("with-VGGT SLat checkpoint upstream identity is inconsistent")
    saved_protocol = (
        checkpoint.get("data_identity", {})
        .get("target_decoder_audit", {})
        .get("protocol_sha256")
    )
    current_protocol = str(
        dataset.config.get("target_source", {}).get("protocol_sha256", "")
    )
    membership = audit_checkpoint_evaluation_membership(
        checkpoint,
        evaluation_protocol_sha256=current_protocol,
        evaluation_object_uids=evaluation_object_uids,
        expected_membership=expected_training_membership,
    )
    if not saved_protocol:
        raise RuntimeError("with-VGGT SLat checkpoint official protocol is missing")
    if (
        membership["protocol_relation"] != "same"
        and not allow_target_protocol_mismatch
    ):
        raise RuntimeError("with-VGGT SLat checkpoint official protocol differs")
    validate_native_slat_official_with_vggt_checkpoint(
        checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=saved_upstream,
    )
    saved = checkpoint["args"]
    sampler, model, decoder, summary, defaults, normalization = (
        build_native_slat_official_with_vggt_components(
            pretrained=pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=saved_upstream,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("with-VGGT endpoint requires the Stock Mesh decoder")
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    load_slat_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    runtime_normalization = {
        key: [float(value) for value in values]
        for key, values in normalization.items()
    }
    if canonical_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("with-VGGT SLat runtime/cache normalization differs")
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    result = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": str(weights),
        "sampler": sampler,
        "model": model,
        "decoder": decoder,
        "summary": summary,
        "params": params,
        "mean": torch.tensor(runtime_normalization["mean"], device=device)[None],
        "std": torch.tensor(runtime_normalization["std"], device=device)[None],
        "training_upstream_native_ss": copy.deepcopy(saved_upstream),
        "evaluation_support_deployment": "VSS",
        "support_deployment_differs_from_training_upstream": True,
        "checkpoint_evaluation_membership": membership,
    }
    del checkpoint
    return result


def endpoint_contract() -> dict[str, Any]:
    return {
        "version": JOINT_ENDPOINT_VERSION,
        "branches": {
            "A": "VSS0 predicted support -> V0 Stock SLat",
            "B": "VSS predicted support -> V0 Stock SLat",
            "C": "VSS predicted support -> V trained SLat",
        },
        "comparisons": {
            "B_minus_A": "VSS support increment through identical V0",
            "C_minus_B": "trained V SLat increment on identical VSS support",
            "C_minus_A": "full trained with-VGGT endpoint increment",
        },
        "slat_support_input": "predicted_only",
        "gt_support_used_as_slat_input": False,
        "official_gt_role": "metric_target_only",
        "official_target_join_contract": copy.deepcopy(TARGET_JOIN_CONTRACT),
        "ss_stock_floor": "VSS0",
        "slat_stock_floor": "V0",
        "slat_context_contract": copy.deepcopy(OFFICIAL_WITH_VGGT_SLAT_CONTRACT),
        "ss_context_contract": copy.deepcopy(MODEL_CONTEXT_CONTRACT),
        "same_ss_initial_noise": True,
        "same_coordinate_keyed_slat_noise": True,
        "same_stock_mesh_decoder": True,
    }


__all__ = [
    "JOINT_ENDPOINT_VERSION",
    "REPORT_FORMAT",
    "TARGET_JOIN_CONTRACT",
    "WORKER_FORMAT",
    "WithVGGTSSSLatEndpointDataset",
    "activate_ss_cache_manifest",
    "build_trained_slat_pipeline",
    "endpoint_contract",
    "load_ss_runtime",
    "load_vss_deployment",
    "official_target_contract",
]
