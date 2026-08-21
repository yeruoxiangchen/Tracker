#!/usr/bin/env python3
"""Immutable mixed-domain datasets and deterministic balanced DDP sampling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, parse_indices
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.dino_only_condition import validate_dino_only_lifting_contract
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.native_ss_genrecon_no_vggt import NO_VGGT_MODEL_CONTRACT
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


MIXED_LIFTING_MANIFEST_VERSION = (
    "pose_point_depth_mv.mixed_no_vggt_lifting_manifest.v1"
)
MIXED_SLAT_MANIFEST_VERSION = "pose_point_depth_mv.mixed_no_vggt_slat_manifest.v1"
DOMAIN_BALANCED_SAMPLER_VERSION = (
    "pose_point_depth_mv.domain_balanced_distributed_sampler.v1"
)
REQUIRED_DOMAINS = ("synthetic", "real")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _resolve_bound_path(value: str, owner: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else owner.parent / path).resolve()


def _validate_domain_names(domains: list[dict[str, Any]]) -> None:
    names = tuple(str(row.get("name", "")) for row in domains)
    if tuple(sorted(names)) != tuple(sorted(REQUIRED_DOMAINS)):
        raise ValueError(f"mixed domains must be exactly {REQUIRED_DOMAINS}, got {names}")
    if any(int(row.get("weight", 0)) != 1 for row in domains):
        raise ValueError("v1 freezes equal synthetic/real domain weights at 1:1")


class MixedPoseLiftingCacheDataset(Dataset):
    """Read two immutable DINO-only lifting manifests without copying samples."""

    def __init__(self, manifest: str | Path, *, indices: str = "all") -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        payload = _load_json(self.manifest_path)
        if payload.get("format") != MIXED_LIFTING_MANIFEST_VERSION:
            raise ValueError(f"unsupported mixed lifting manifest={payload.get('format')!r}")
        domains = payload.get("domains")
        if not isinstance(domains, list) or len(domains) != 2:
            raise ValueError("mixed lifting manifest must contain two domains")
        _validate_domain_names(domains)

        self.domain_datasets: dict[str, PoseLiftingCacheDataset] = {}
        self.domain_contracts: dict[str, dict[str, Any]] = {}
        all_rows: list[dict[str, Any]] = []
        lookup: list[tuple[str, int]] = []
        seen_uids: set[str] = set()
        object_domains: dict[str, str] = {}
        for domain in domains:
            name = str(domain["name"])
            source_path = _resolve_bound_path(str(domain["manifest"]), self.manifest_path)
            if sha256_file(source_path) != str(domain.get("manifest_sha256", "")):
                raise RuntimeError(f"{name} lifting manifest hash changed: {source_path}")
            dataset = PoseLiftingCacheDataset(source_path, indices="all")
            contract = validate_dino_only_lifting_contract(dataset)
            expected = {
                "sample_count": len(dataset.rows),
                "object_count": len(
                    {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
                ),
                "config_hash": dataset.config_hash,
            }
            mismatch = {
                key: (domain.get(key), value)
                for key, value in expected.items()
                if domain.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"{name} mixed lifting binding differs: {mismatch}")
            self.domain_datasets[name] = dataset
            self.domain_contracts[name] = contract
            for source_index, source_row in enumerate(dataset.rows):
                uid = str(source_row.get("uid", ""))
                object_uid = str(source_row.get("object_uid", uid))
                if uid in seen_uids:
                    raise ValueError(
                        f"mixed domains must be UID/object-disjoint: {name}:{uid}:{object_uid}"
                    )
                seen_uids.add(uid)
                previous_domain = object_domains.setdefault(object_uid, name)
                if previous_domain != name:
                    raise ValueError(
                        f"mixed domains overlap object: {name}:{uid}:{object_uid}"
                    )
                row = {
                    **source_row,
                    "_mixed_domain": name,
                    "_mixed_domain_weight": 1,
                    "_mixed_source_index": source_index,
                }
                all_rows.append(row)
                lookup.append((name, source_index))

        selected = parse_indices(indices, len(all_rows))
        self.rows = [all_rows[index] for index in selected]
        self._lookup = [lookup[index] for index in selected]
        self.visual_feature_dim = 1024
        first = self.domain_datasets[REQUIRED_DOMAINS[0]]
        self.feature_metadata = dict(first.feature_metadata)
        common = {
            key: self.domain_contracts[REQUIRED_DOMAINS[0]][key]
            for key in (
                "version",
                "visual_feature_dim",
                "vggt_feature_dim",
                "dino_feature_dim",
                "patch_count",
                "context_source",
                "vggt_model_executed",
                "stock_condition_source",
                "slat_condition_source",
                "depth_policy",
            )
        }
        for name, contract in self.domain_contracts.items():
            changed = {
                key: (common[key], contract[key])
                for key in common
                if contract.get(key) != common[key]
            }
            if changed:
                raise ValueError(f"DINO-only contracts differ for {name}: {changed}")
        self.config = {
            "version": MIXED_LIFTING_MANIFEST_VERSION,
            "domain_order": list(REQUIRED_DOMAINS),
            "domain_weights": {name: 1 for name in REQUIRED_DOMAINS},
            "sampler": {
                "version": DOMAIN_BALANCED_SAMPLER_VERSION,
                "policy": "equal_domain_object_cycles",
                "ratio": {"synthetic": 1, "real": 1},
            },
            "component_config_hashes": {
                name: dataset.config_hash
                for name, dataset in sorted(self.domain_datasets.items())
            },
            "no_vggt": common,
        }
        self.config_hash = canonical_json_sha256(self.config)
        if self.config_hash != str(payload.get("config_hash", "")):
            raise RuntimeError("mixed lifting config hash differs")
        self.root = self.manifest_path.parent
        self.source_cache_manifest = str(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        domain, source_index = self._lookup[index]
        sample = dict(self.domain_datasets[domain][source_index])
        sample["mixed_domain"] = domain
        sample["mixed_source_index"] = int(source_index)
        return sample


def validate_mixed_no_vggt_cache_contract(
    dataset: Any, *, training_config_hash: str | None = None
) -> dict[str, Any]:
    if not isinstance(dataset, MixedPoseLiftingCacheDataset):
        raise TypeError("mixed no-VGGT training requires MixedPoseLiftingCacheDataset")
    if training_config_hash is not None and str(training_config_hash) != dataset.config_hash:
        raise RuntimeError("mixed training config hash differs")
    no_vggt = dict(dataset.config["no_vggt"])
    if (
        no_vggt.get("vggt_feature_dim") != 0
        or no_vggt.get("dino_feature_dim") != 1024
        or no_vggt.get("vggt_model_executed") is not False
    ):
        raise RuntimeError("mixed lifting cache is not strictly DINO-only")
    domain_summary = {
        name: {
            "sample_count": len(component.rows),
            "object_count": len(
                {str(row.get("object_uid", row["uid"])) for row in component.rows}
            ),
            "config_hash": component.config_hash,
            "contract": dataset.domain_contracts[name],
        }
        for name, component in sorted(dataset.domain_datasets.items())
    }
    return {
        "visual_feature_dim": 1024,
        "dino_feature_dim": 1024,
        "patch_count": int(no_vggt["patch_count"]),
        "config_hash": dataset.config_hash,
        "no_vggt": no_vggt,
        "mixed_domains": domain_summary,
        "sampler": dict(dataset.config["sampler"]),
        "model_context": NO_VGGT_MODEL_CONTRACT,
    }


class DomainBalancedDistributedSampler(Sampler[int]):
    """Equal-domain object cycles, deterministic across ranks and resumes."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        resume_micro_step: int = 0,
    ) -> None:
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.resume_micro_step = int(resume_micro_step)
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        if self.resume_micro_step < 0:
            raise ValueError("resume_micro_step must be non-negative")
        self.by_domain_object: dict[str, dict[str, list[int]]] = {}
        for index, row in enumerate(rows):
            domain = str(row.get("_mixed_domain", ""))
            object_uid = str(row.get("object_uid", row.get("uid", "")))
            if domain not in REQUIRED_DOMAINS or not object_uid:
                raise ValueError("mixed sampler row lacks frozen domain/object identity")
            self.by_domain_object.setdefault(domain, {}).setdefault(object_uid, []).append(index)
        if tuple(sorted(self.by_domain_object)) != tuple(sorted(REQUIRED_DOMAINS)):
            raise ValueError("mixed sampler requires both synthetic and real rows")
        maximum = max(len(values) for values in self.by_domain_object.values())
        rank_factor = self.num_replicas // math.gcd(self.num_replicas, len(REQUIRED_DOMAINS))
        self.per_domain_size = math.ceil(maximum / rank_factor) * rank_factor
        self.total_size = self.per_domain_size * len(REQUIRED_DOMAINS)
        if self.total_size % self.num_replicas:
            raise RuntimeError("balanced global schedule is not rank-divisible")
        self.num_samples = self.total_size // self.num_replicas
        self.resume_epoch, self.resume_offset = divmod(
            self.resume_micro_step, self.num_samples
        )
        self.local_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.local_epoch = int(epoch)

    def _domain_indices(self, domain: str, epoch: int) -> list[int]:
        by_object = self.by_domain_object[domain]
        object_uids = sorted(by_object)
        selected: list[int] = []
        cycle = 0
        domain_seed = 0 if domain == "synthetic" else 1_000_003
        while len(selected) < self.per_domain_size:
            generator = torch.Generator().manual_seed(
                self.seed + epoch * 10_000_019 + domain_seed + cycle * 97_409
            )
            order = torch.randperm(len(object_uids), generator=generator).tolist()
            for object_index in order:
                candidates = by_object[object_uids[object_index]]
                choice = int(torch.randint(len(candidates), (1,), generator=generator).item())
                selected.append(candidates[choice])
                if len(selected) == self.per_domain_size:
                    break
            cycle += 1
        return selected

    def __iter__(self) -> Iterator[int]:
        epoch = self.resume_epoch + self.local_epoch
        by_domain = {
            domain: self._domain_indices(domain, epoch) for domain in REQUIRED_DOMAINS
        }
        global_order = [
            by_domain[domain][position]
            for position in range(self.per_domain_size)
            for domain in REQUIRED_DOMAINS
        ]
        local = global_order[self.rank : self.total_size : self.num_replicas]
        if self.local_epoch == 0 and self.resume_offset:
            local = local[self.resume_offset :]
        return iter(local)

    def __len__(self) -> int:
        if self.local_epoch == 0:
            return self.num_samples - self.resume_offset
        return self.num_samples

    def identity(self) -> dict[str, Any]:
        return {
            "version": DOMAIN_BALANCED_SAMPLER_VERSION,
            "domains": list(REQUIRED_DOMAINS),
            "ratio": {"synthetic": 1, "real": 1},
            "object_counts": {
                name: len(rows) for name, rows in sorted(self.by_domain_object.items())
            },
            "per_domain_size": self.per_domain_size,
            "num_replicas": self.num_replicas,
            "resume_micro_step": self.resume_micro_step,
        }


class MixedNativeConditionSLatDataset(Dataset):
    """Join per-domain SLat/lifting caches through immutable meta manifests."""

    def __init__(
        self,
        slat_manifest: str | Path,
        lifting_manifest: str | Path,
        *,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        self.manifest_path = Path(slat_manifest).expanduser().resolve()
        self.lifting_manifest_path = Path(lifting_manifest).expanduser().resolve()
        slat_payload = _load_json(self.manifest_path)
        lifting_payload = _load_json(self.lifting_manifest_path)
        if slat_payload.get("format") != MIXED_SLAT_MANIFEST_VERSION:
            raise ValueError("unsupported mixed SLat manifest")
        if lifting_payload.get("format") != MIXED_LIFTING_MANIFEST_VERSION:
            raise ValueError("mixed SLat training requires a mixed lifting manifest")
        if sha256_file(self.lifting_manifest_path) != str(
            slat_payload.get("lifting_manifest_sha256", "")
        ):
            raise RuntimeError("mixed SLat/lifting meta-manifest binding changed")
        slat_domains = slat_payload.get("domains")
        lifting_domains = lifting_payload.get("domains")
        if not isinstance(slat_domains, list) or not isinstance(lifting_domains, list):
            raise ValueError("mixed SLat manifests lack domains")
        _validate_domain_names(slat_domains)
        _validate_domain_names(lifting_domains)
        lifting_by_name = {str(row["name"]): row for row in lifting_domains}

        self.domain_datasets: dict[str, NativeConditionSLatDataset] = {}
        all_rows: list[dict[str, Any]] = []
        lookup: list[tuple[str, int]] = []
        configs: dict[str, dict[str, Any]] = {}
        seen: set[tuple[str, int]] = set()
        object_domains: dict[str, str] = {}
        for domain in slat_domains:
            name = str(domain["name"])
            slat_path = _resolve_bound_path(str(domain["manifest"]), self.manifest_path)
            lifting_row = lifting_by_name[name]
            lifting_path = _resolve_bound_path(
                str(lifting_row["manifest"]), self.lifting_manifest_path
            )
            if sha256_file(slat_path) != str(domain.get("manifest_sha256", "")):
                raise RuntimeError(f"{name} SLat manifest hash changed")
            dataset = NativeConditionSLatDataset(
                slat_path,
                lifting_path,
                indices="all",
                verify_hashes=verify_hashes,
            )
            configs[name] = dict(dataset.config)
            self.domain_datasets[name] = dataset
            for source_index, source_row in enumerate(dataset.rows):
                identity = (str(source_row["uid"]), int(source_row["support_seed"]))
                object_uid = str(source_row["object_uid"])
                if identity in seen:
                    raise ValueError(f"duplicate mixed SLat identity={identity}")
                seen.add(identity)
                previous_domain = object_domains.setdefault(object_uid, name)
                if previous_domain != name:
                    raise ValueError(f"mixed SLat domains overlap object={object_uid}")
                all_rows.append(
                    {
                        **source_row,
                        "_mixed_domain": name,
                        "_mixed_domain_weight": 1,
                        "_mixed_source_index": source_index,
                    }
                )
                lookup.append((name, source_index))
        first = self.domain_datasets[REQUIRED_DOMAINS[0]]
        common_fields = ("condition_arch", "native_ss_deployment")
        for field in common_fields:
            values = {canonical_json_sha256(config.get(field)) for config in configs.values()}
            if len(values) != 1:
                raise RuntimeError(f"mixed SLat domain configs differ for {field}")
        normalization_hashes = {
            dataset.slat_normalization_hash for dataset in self.domain_datasets.values()
        }
        if len(normalization_hashes) != 1:
            raise RuntimeError("mixed SLat domains use different normalization")
        selected = parse_indices(indices, len(all_rows))
        self.rows = [all_rows[index] for index in selected]
        self._lookup = [lookup[index] for index in selected]
        self.config = dict(first.config)
        self.config["mixed_no_vggt"] = {
            "version": MIXED_SLAT_MANIFEST_VERSION,
            "domains": list(REQUIRED_DOMAINS),
            "ratio": {"synthetic": 1, "real": 1},
            "component_config_hashes": {
                name: dataset.config_hash
                for name, dataset in sorted(self.domain_datasets.items())
            },
        }
        self.config_hash = canonical_json_sha256(self.config["mixed_no_vggt"])
        if self.config_hash != str(slat_payload.get("config_hash", "")):
            raise RuntimeError("mixed SLat config hash differs")
        self.slat_normalization = dict(first.slat_normalization)
        self.slat_normalization_hash = first.slat_normalization_hash
        self.identity = {
            "version": MIXED_SLAT_MANIFEST_VERSION,
            "slat_manifest": str(self.manifest_path),
            "slat_manifest_sha256": sha256_file(self.manifest_path),
            "lifting_manifest": str(self.lifting_manifest_path),
            "lifting_manifest_sha256": sha256_file(self.lifting_manifest_path),
            "uid_count": len(self.rows),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        domain, source_index = self._lookup[index]
        sample = dict(self.domain_datasets[domain][source_index])
        sample["mixed_domain"] = domain
        sample["mixed_source_index"] = int(source_index)
        return sample


__all__ = [
    "DOMAIN_BALANCED_SAMPLER_VERSION",
    "DomainBalancedDistributedSampler",
    "MIXED_LIFTING_MANIFEST_VERSION",
    "MIXED_SLAT_MANIFEST_VERSION",
    "MixedNativeConditionSLatDataset",
    "MixedPoseLiftingCacheDataset",
    "validate_mixed_no_vggt_cache_contract",
]
