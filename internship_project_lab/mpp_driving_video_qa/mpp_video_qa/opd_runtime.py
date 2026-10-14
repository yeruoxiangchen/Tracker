"""Execution optimizations; training identities and token objectives live elsewhere."""
import contextlib
from collections import OrderedDict, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch


class GradientBuckets:
    """Fixed-order SUM/world reduction, with one reusable buffer per dtype/device."""
    def __init__(self, parameters, world, bucket_mb=25):
        self.parameters, self.world = list(parameters), world
        self.buckets, self.buffers = [], {}
        if world <= 1:
            return
        limit = int(bucket_mb * 1024 * 1024)
        if limit <= 0:
            raise ValueError("gradient bucket size must be positive")
        bucket, size, key = [], 0, None
        for parameter in self.parameters:
            current = (parameter.device, parameter.dtype)
            nbytes = parameter.numel() * parameter.element_size()
            if bucket and (current != key or size + nbytes > limit):
                self.buckets.append((key, bucket))
                bucket, size = [], 0
            key = current
            bucket.append(parameter)
            size += nbytes
        if bucket:
            self.buckets.append((key, bucket))
        for key, parameters in self.buckets:
            size = sum(p.numel() for p in parameters)
            if key not in self.buffers or self.buffers[key].numel() < size:
                self.buffers[key] = torch.empty(size, device=key[0], dtype=key[1])

    @torch.no_grad()
    def reduce(self):
        for key, parameters in self.buckets:
            size = sum(p.numel() for p in parameters)
            buffer = self.buffers[key][:size]
            offset = 0
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                buffer[offset:offset + parameter.numel()].copy_(parameter.grad.reshape(-1))
                offset += parameter.numel()
            torch.distributed.all_reduce(buffer)
            buffer.div_(self.world)
            offset = 0
            for parameter in parameters:
                parameter.grad.copy_(buffer[offset:offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()


class PhaseTimer:
    """Sample update windows. CUDA synchronization happens only when collecting a sample."""
    def __init__(self, device, every=0):
        self.device, self.every, self.active = device, every, False
        self.wall, self.events = defaultdict(lambda: [0.0, 0]), []

    def begin(self, step):
        self.active = bool(self.every and step % self.every == 0)
        self.wall.clear()
        self.events.clear()
        self.started = time.perf_counter()

    @contextlib.contextmanager
    def phase(self, name, device=None):
        if not self.active:
            yield
            return
        device = torch.device(device) if device is not None else self.device
        events = None
        if device.type == "cuda":
            with torch.cuda.device(device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name][0] += (time.perf_counter() - started) * 1000
            self.wall[name][1] += 1
            if events:
                with torch.cuda.device(device):
                    events[1].record()
                self.events.append((name, events))

    def finish(self):
        if not self.active:
            return None
        gpu = defaultdict(float)
        for name, (start, end) in self.events:
            end.synchronize()
            gpu[name] += start.elapsed_time(end)
        return {"update_wall_ms": (time.perf_counter() - self.started) * 1000,
                "phases": {name: {"wall_ms": value[0], "calls": value[1],
                                  "gpu_ms": gpu.get(name)} for name, value in self.wall.items()}}


def phase(timer, name, device=None):
    return timer.phase(name, device) if timer is not None else contextlib.nullcontext()


class TeacherGateCache:
    """Reference-only greedy correction/gate cache, never student-prefix probabilities."""
    def __init__(self, directory, identity, memory_entries=512):
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.directory = Path(directory) / namespace
        self.memory, self.memory_entries = OrderedDict(), memory_entries
        self.source_cache = identity.get("cache_dir")

    def key(self, row, prompt, max_new_tokens):
        payload = {"question_id": row["question_id"], "segment_id": row["segment_id"],
                   "frames": row["frames"], "prompt": prompt, "max_new_tokens": max_new_tokens}
        if self.source_cache:
            from .data import cache_file
            stamp = cache_file(self.source_cache, row["segment_id"]).stat()
            payload["visual_cache_stamp"] = [stamp.st_size, stamp.st_mtime_ns]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key):
        value = self.memory.get(key)
        if value is None:
            path = self.directory / key[:2] / (key + ".json")
            if not path.exists():
                return None
            value = json.loads(path.read_text())
        self._remember(key, value)
        return value

    def _remember(self, key, value):
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.memory_entries:
            self.memory.popitem(last=False)

    def put(self, key, value):
        path = self.directory / key[:2] / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
        temporary.write_text(json.dumps(value, allow_nan=False))
        temporary.replace(path)
        self._remember(key, value)


def validate_resume_identity(saved, requested):
    # Execution settings are stored separately. All historical semantic keys remain strict.
    if saved != requested:
        changed = sorted(k for k in saved.keys() | requested.keys() if saved.get(k) != requested.get(k))
        raise ValueError("resume data, teacher or training configuration differs: %s" % changed)
