"""Fixed-sample OPD diagnostics, not a trainer or an automatic fact annotator."""
from collections import Counter
import hashlib
import json
import math
import random

import torch

PROTOCOL = "fixed_teacher_feedback_opd_diagnostics_v1"
ARMS = ("parent", "ce", "opd")
QA_FIELDS = ("question_id", "segment_id", "source_subset", "question", "frames")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_original(draw, original):
    if any(draw.get(k) != original.get(k) for k in QA_FIELDS):
        raise ValueError("cached draw changed original QA/frames: %s" % original["question_id"])
    if draw.get("references", original["answers"]) != original["answers"]:
        raise ValueError("cached draw changed original reference answers")
    if original.get("feedback_training_source") != "train" or len(original["frames"]) != 5:
        raise ValueError("diagnostic trajectories must be original train-only five-frame QA")
    ids = draw.get("student_token_ids")
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("missing exact cached student token IDs")


def assert_split_isolation(train, heldout):
    result = {}
    for name, rows in heldout.items():
        counts = {}
        for key in ("question_id", "segment_id", "frames"):
            values = lambda rs: {v for r in rs for v in
                                (r[key] if key == "frames" else [r[key]])}
            counts[key] = len(values(train) & values(rows))
        if any(counts.values()):
            raise ValueError("training diagnostic overlaps %s: %s" % (name, counts))
        result[name] = counts
    return result


def select_trajectories(comparison, probe, originals, controls=100, seed=44):
    """All existing paired errors + independent source-proportional correct controls."""
    cases, used = [], set()
    for draw in comparison:
        original = originals[draw["question_id"]]
        check_original(draw, original)
        if (draw["question_id"] in used or not draw["student_finished"] or
                draw["student_judge"]["correct"]):
            raise ValueError("paired comparison must contain unique completed training errors")
        teacher = draw["modes"]["reference_only"]
        cases.append(dict(original, case_id=draw["question_id"], cohort="cached_error",
                          student_answer=draw["student_answer"],
                          student_token_ids=draw["student_token_ids"], sample_id=draw["sample_id"],
                          student_judge=draw["student_judge"],
                          teacher_correction=teacher["teacher_correction"],
                          teacher_judge=teacher["teacher_judge"],
                          teacher_finished=teacher["teacher_finished"],
                          eligible=teacher["eligible"], gate_reason=teacher["gate_reason"]))
        used.add(draw["question_id"])
    rng = random.Random(seed)
    pool = list(probe)
    rng.shuffle(pool)
    by_source = {}
    for draw in pool:
        qid = draw["question_id"]
        if qid not in originals:
            continue
        check_original(draw, originals[qid])
        if (qid in used or not draw["student_finished"] or
                not draw["student_judge"]["correct"] or
                not math.isfinite(draw["student_judge"]["logit"])):
            continue
        used.add(qid)
        by_source.setdefault(draw["source_subset"], []).append(draw)
    sizes = Counter(r["source_subset"] for r in originals.values())
    total = sum(sizes.values())
    quotas = {s: int(controls * n / total) for s, n in sizes.items()}
    remainder = sorted(sizes, key=lambda s: (-(controls * sizes[s] / total - quotas[s]), s))
    for source in remainder[:controls - sum(quotas.values())]:
        quotas[source] += 1
    for source, count in sorted(quotas.items()):
        if len(by_source.get(source, [])) < count:
            raise ValueError("not enough independent correct controls: %s" % source)
        for draw in by_source[source][:count]:
            cases.append(dict(originals[draw["question_id"]], case_id=draw["question_id"],
                              cohort="cached_correct_control", student_answer=draw["student_answer"],
                              student_token_ids=draw["student_token_ids"], sample_id=draw["sample_id"],
                              student_judge=draw["student_judge"], teacher_correction=None,
                              teacher_judge=None, teacher_finished=False, eligible=False,
                              gate_reason="student_correct_ce_only"))
    return cases


def select_effective_batches(logs_by_rank, originals, steps, accumulation, world):
    """Keep last resumed draw per (rank,micro_step), replay the actual effective batch."""
    if len(set(steps)) != len(steps) or any(s <= 0 for s in steps):
        raise ValueError("gradient steps must be unique positive optimizer-step numbers")
    batches, duplicates = [], 0
    for rank in range(world):
        selected = {}
        for draw in logs_by_rank[rank]:
            micro = draw["micro_step"]
            step = micro // accumulation + 1
            if step in steps:
                duplicates += int(micro in selected)
                selected[micro] = draw
        for step in steps:
            for micro in range((step - 1) * accumulation, step * accumulation):
                if micro not in selected:
                    raise ValueError("incomplete effective batch step=%d rank=%d micro=%d" % (step, rank, micro))
                draw = selected[micro]
                original = originals[draw["question_id"]]
                if draw.get("source_subset") != original["source_subset"]:
                    raise ValueError("rollout source differs from original training QA")
                check_original(dict(original, **{k: draw[k] for k in
                               ("student_token_ids",)}), original)
                if draw.get("teacher_mode") != "reference_only":
                    raise ValueError("gradient replay requires the original reference_only teacher")
                if draw["eligible"] != (draw["gate_reason"] == "eligible"):
                    raise ValueError("inconsistent cached gate")
                batches.append(dict(original, **draw, effective_step=step, rank=rank,
                                    case_id="%d:%d:%d" % (step, rank, micro),
                                    divisor=world * accumulation))
    batches.sort(key=lambda r: (steps.index(r["effective_step"]), r["rank"], r["micro_step"]))
    return batches, duplicates


def parameter_group(name):
    for group, marker in (("language", "llm_model."), ("projector", "llm_proj."),
                          ("qformer", "visual_bridge.")):
        if marker in name:
            return group
    raise ValueError("unclassified trainable parameter: %s" % name)


def capture_gradients(named):
    return {n: p.grad.detach().float().cpu().clone() for n, p in named if p.grad is not None}


def add_gradients(total, part):
    for name, value in part.items():
        if name not in total:
            total[name] = value.clone()
        else:
            total[name].add_(value)


def gradient_geometry(ce, jsd, named_shapes, clip_norm=1.0):
    """Missing gradients are zeros. Inputs already include loss weights and /batch."""
    accum = {g: dict(ce2=0., jsd2=0., dot=0., parameters=0, ce_nonzero=0, jsd_nonzero=0)
             for g in ("language", "projector", "qformer", "all")}
    for name, count in named_shapes.items():
        a, b = ce.get(name), jsd.get(name)
        aa = float(a.square().sum()) if a is not None else 0.
        bb = float(b.square().sum()) if b is not None else 0.
        dot = float((a * b).sum()) if a is not None and b is not None else 0.
        if not all(math.isfinite(x) for x in (aa, bb, dot)):
            raise ValueError("nonfinite gradient: %s" % name)
        for group in (parameter_group(name), "all"):
            v = accum[group]
            v["parameters"] += count
            v["ce2"] += aa
            v["jsd2"] += bb
            v["dot"] += dot
            v["ce_nonzero"] += int(aa > 0)
            v["jsd_nonzero"] += int(bb > 0)
    result = {}
    for group, v in accum.items():
        a, b = math.sqrt(v["ce2"]), math.sqrt(v["jsd2"])
        result[group] = {"ce_l2": a, "weighted_jsd_l2": b, "dot": v["dot"],
                         "cosine": max(-1., min(1., v["dot"] / (a * b))) if a and b else None,
                         "jsd_to_ce_norm_ratio": b / a if a else None,
                         "combined_l2_before_clip": math.sqrt(max(0., v["ce2"] + v["jsd2"] + 2 * v["dot"])),
                         "parameters": v["parameters"], "ce_nonzero_tensors": v["ce_nonzero"],
                         "jsd_nonzero_tensors": v["jsd_nonzero"]}
    norm = result["all"]["combined_l2_before_clip"]
    result["all"]["combined_global_clip_scale"] = min(1., clip_norm / (norm + 1e-6))
    result["all"]["ce_only_global_clip_scale"] = min(1., clip_norm / (result["all"]["ce_l2"] + 1e-6))
    return result


def tensor_fingerprint(named):
    h = hashlib.sha256()
    for name, tensor in sorted(named, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def review_category(question):
    """Question-only review strata; these are NOT ground-truth factual token labels."""
    q = question.lower()
    if q.startswith("why") or "reason" in q:
        return "cause"
    if any(s in q for s in ("left", "right", "direction", "lane", "steer", "turn")):
        return "direction"
    if any(s in q for s in ("vehicle", "pedestrian", "car", "truck", "traffic", "describe", "color")):
        return "object"
    return "action"


def select_review(cases, count=40):
    pools = {}
    for row in cases:
        pools.setdefault((row["cohort"], review_category(row["question"])), []).append(row)
    selected = []
    for index in range(max((len(v) for v in pools.values()), default=0)):
        for key in sorted(pools):
            if index < len(pools[key]) and len(selected) < count:
                selected.append(pools[key][index]["case_id"])
    return selected


def validate_annotations(rows, cases):
    """Reviewed positions explicitly supplied by a person; no lexical automatic truth mask."""
    index = {c["case_id"]: c for c in cases}
    seen = set()
    for row in rows:
        key = row["case_id"]
        if key not in index or key in seen:
            raise ValueError("unknown or duplicate reviewed case")
        seen.add(key)
        if not row.get("reviewed"):
            continue
        if not row.get("reviewer") or row.get("reference_consistent") not in (True, False):
            raise ValueError("reviewed cases need reviewer and explicit reference consistency")
        used = set()
        for span in row.get("spans", []):
            if span.get("label") not in ("wrong_fact", "correct_fact", "wording"):
                raise ValueError("unknown reviewed span label")
            positions = span.get("positions", [])
            if not positions or len(set(positions)) != len(positions):
                raise ValueError("span requires distinct explicit token positions")
            for pos in positions:
                if type(pos) is not int or not 0 <= pos < len(index[key]["student_token_ids"]) or pos in used:
                    raise ValueError("invalid or overlapping factual token positions")
                used.add(pos)
    return rows
