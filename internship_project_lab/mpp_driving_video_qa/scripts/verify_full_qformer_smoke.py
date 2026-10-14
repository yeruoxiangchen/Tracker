#!/usr/bin/env python3
"""Summarize real training, resumption and generation smoke artifacts on CPU."""
import argparse
import json
from pathlib import Path

import torch
from peft.utils.save_and_load import load_peft_weights


def differences(left, right):
    if set(left) != set(right):
        raise ValueError("saved weight keys differ")
    maximum = max(float((left[k].float() - right[k].float()).abs().max()) for k in left)
    return {"tensors": len(left), "max_abs_delta": maximum, "bitwise_equal": maximum == 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--continuous", type=Path, required=True)
    p.add_argument("--resumed", type=Path, required=True)
    p.add_argument("--generation", type=Path, required=True)
    args = p.parse_args()
    result = {"training": json.loads((args.continuous / "summary.json").read_text()),
              "resumed": json.loads((args.resumed / "summary.json").read_text()),
              "generation": json.loads((args.generation / "generation_summary.json").read_text())}
    result["gradients_by_rank"] = {f.stem: json.loads(f.read_text())
                                   for f in sorted(args.continuous.glob("gradients_rank_*.json"))}
    if len(result["gradients_by_rank"]) != 4:
        raise ValueError("four-rank gradient records are required")
    for groups in result["gradients_by_rank"].values():
        if any(g["parameters"] != g["with_gradient"] or not g["nonzero"] for g in groups.values()):
            raise ValueError("trainable parameters did not receive gradients")
    for run in ("training", "resumed"):
        if not result[run]["completed"] or result[run]["optimizer_steps"] != 2:
            raise ValueError("two-step smoke is incomplete")
        if result[run]["max_parameter_checksum_spread"] != 0:
            raise ValueError("distributed parameter checksums differ")
    if result["generation"]["completed"] != result["generation"]["total"]:
        raise ValueError("reload generation is incomplete")
    comparison = {}
    for key, filename in (("qformer", "final_qformer_adapter.pt"), ("projector", "final_projector.pt")):
        left = torch.load(args.continuous / filename, map_location="cpu")["state_dict"]
        right = torch.load(args.resumed / filename, map_location="cpu")["state_dict"]
        comparison[key] = differences(left, right)
        del left, right
    comparison["language_lora"] = differences(
        load_peft_weights(str(args.continuous / "final_adapter"), device="cpu"),
        load_peft_weights(str(args.resumed / "final_adapter"), device="cpu"))
    result["resume_weight_comparison"] = comparison
    result["functional_checks_passed"] = True
    result["interpretation"] = "Engineering checks only; not held-out quality. Exact resume equality is reported, not assumed."
    destination = args.resumed / "verification.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verification": str(destination), "resume_weight_comparison": comparison,
                      "functional_checks_passed": True}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
