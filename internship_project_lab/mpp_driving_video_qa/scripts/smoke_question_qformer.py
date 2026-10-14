#!/usr/bin/env python3
"""Two real MPP CE updates, original train QA, question effects and complete bridge reload."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feedback_opd_common import add_runtime_arguments, configure_model, build_student, training_dataset
from scripts.train_sft import seed_all, gpu_memory
from scripts.generate_answers import write_json_atomic
from scripts.train_feedback_opd import tracked_parameters
from mpp_video_qa.visual_bridge import VisualBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    args = parser.parse_args()
    args.qformer_text_input = "question"
    args = configure_model(args, train=True)
    if args.updates < 2 or args.accumulation < 1:
        parser.error("verify at least two updates and nonzero accumulation")
    device = torch.device("cuda", 0)
    torch.set_num_threads(1)
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"status": "loading_question_mpp", "output_dir": str(output)}), flush=True)
    dataset = training_dataset(args)
    model = build_student(args, device, train=True)
    bridge = model.backbone.visual_bridge
    tracked = tracked_parameters(model)
    initial = {name: p.detach().float().cpu().clone() for name, (_, p) in tracked.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    question_embedding_gradient = None
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.accumulation):
            item = dataset[step * args.accumulation + micro]
            features = item["features"].unsqueeze(0).to(device)
            loss = model(features, [item["row"]["question"]], [item["answer"]]).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite smoke CE")
            losses.append(float(loss.detach()))
            (loss / args.accumulation).backward()
        grad = bridge.Qformer.bert.embeddings.word_embeddings.weight.grad
        question_embedding_gradient = float(grad.abs().sum()) if grad is not None else 0.
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        print(json.dumps({"optimizer_step": step + 1, "ce": losses[-1],
                          "question_embedding_gradient": question_embedding_gradient}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, loss
    model.eval()
    first = dataset[0]
    images = first["features"].unsqueeze(0).to(device)
    question = first["row"]["question"]
    with torch.no_grad():
        original = bridge(images, [question])
        altered = bridge(images, ["What is the colour of the traffic light?"])
        delta = float((original.float() - altered.float()).abs().max())
        generated = model.generate(images, [question], max_new_tokens=32)[0]
    visual_root = Path(args.visual_checkpoint_root or args.checkpoint_root)
    # The large temporary smoke checkpoint is removed automatically; publish only evidence.
    with tempfile.TemporaryDirectory(prefix="mpp_question_smoke_") as temporary:
        weights = Path(temporary) / "qformer.pt"
        bridge.save(weights)
        restored = VisualBridge(args.bridge_init, visual_root / "ckpt/bert-base-uncased", weights=weights)
        state = restored.state_dict()
        complete = set(state) == set(bridge.state_dict())
        exact = complete and all(torch.equal(value.detach().float().cpu(), state[key])
                                 for key, value in bridge.state_dict().items())
    updates = {name: float((p.detach().float().cpu() - initial[name]).abs().max())
               for name, (_, p) in tracked.items()}
    result = {"completed": True, "optimizer_steps": args.updates, "accumulation": args.accumulation,
              "real_qa_visits": len(losses), "ce_losses": losses, "question_qformer": bridge.question_identity,
              "initialization": bridge.initialization_report, "tracked_parameter_deltas": updates,
              "question_embedding_gradient": question_embedding_gradient,
              "visual_token_change_max": delta, "complete_checkpoint_keys": complete,
              "checkpoint_reload_weights_exact": exact, "generation": generated,
              "gpu_memory": gpu_memory(device),
              "note": "Single-GPU implementation smoke, not four-GPU formal training or benchmark accuracy. Temporary tensor checkpoint was removed."}
    if (not exact or delta <= 0 or question_embedding_gradient <= 0 or
            any(value <= 0 for value in updates.values())):
        raise ValueError("question bridge smoke did not verify updates/input effects/reload")
    write_json_atomic(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
