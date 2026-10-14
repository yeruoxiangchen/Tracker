"""Exact global QA batches across different replica counts, without repeated QA."""
import math

import torch


GLOBAL_BATCH_PROTOCOL = "exact_global_batch_v1"


class ExactGlobalBatchSampler:
    """Shard each global update, using None for unused synchronization slots.

    The permutation is the same seed+epoch permutation as DistributedSampler.
    Four replicas/batch16 give its original strided order, two use eight slots,
    and three use six slots with only 6+5+5 real examples. Empty slots perform
    no forward/backward, rollout or logging, but never skip the update collective.
    """
    def __init__(self, records, global_batch_size, world, rank, seed):
        if (records < 1 or global_batch_size < 1 or world < 1 or world > global_batch_size or
                not 0 <= rank < world):
            raise ValueError("invalid exact global batch layout")
        self.records, self.global_batch_size = records, global_batch_size
        self.world, self.rank, self.seed = world, rank, seed
        self.accumulation = math.ceil(global_batch_size / world)
        self.updates_per_epoch = math.ceil(records / global_batch_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.updates_per_epoch * self.accumulation

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.records, generator=generator).tolist()
        for start in range(0, self.records, self.global_batch_size):
            count = min(self.global_batch_size, self.records - start)
            for micro in range(self.accumulation):
                offset = self.rank + micro * self.world
                yield permutation[start + offset] if offset < count else None

    def loss_divisor(self, batch_index):
        """SUM/world gradient reduction must produce SUM(real losses)/real count."""
        if not 0 <= batch_index < len(self):
            raise IndexError("micro slot outside epoch")
        start = (batch_index // self.accumulation) * self.global_batch_size
        return min(self.global_batch_size, self.records - start) / self.world


def weight_batch_layout(world):
    if world not in (2, 3, 4):
        raise ValueError("weight comparison supports two, three or four training GPUs")
    return {"world_size": world, "grad_accumulation": math.ceil(16 / world),
            "global_batch_size": 16, "global_batch_protocol": GLOBAL_BATCH_PROTOCOL}


def normalized_weight_config(config):
    """Normalize physical layout only after validating an equal real batch budget."""
    world, accumulation = config.get("world_size"), config.get("grad_accumulation")
    expected = weight_batch_layout(world)
    if "global_batch_size" in config or "global_batch_protocol" in config:
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("weight comparison requires exact batch16 with the correct slot count")
    elif world * accumulation != 16:
        raise ValueError("legacy layout does not have effective batch16")
    result = dict(config, world_size=4, grad_accumulation=4)
    result.pop("global_batch_size", None)
    result.pop("global_batch_protocol", None)
    return result
