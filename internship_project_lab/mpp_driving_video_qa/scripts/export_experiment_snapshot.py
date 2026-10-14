#!/usr/bin/env python3
"""Publish small experiment evidence and checkpoint locations, never tensor weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "sft_parent": ("mpp_lingoqa_qformer_all_1epoch_v1", ""),
    "retained_sft": ("mpp_feedback_sft_control_40k_2epoch_v2", "final"),
    "resume_opd": ("mpp_feedback_opd_40k_2epoch_v2", "final"),
    "expanded_language_lora": ("mpp_language_lora_decoder_40k_2epoch_v1", "final"),
    "opd_weight3": ("mpp_feedback_opd_weight3_40k_2epoch_v1", "checkpoint-002500"),
}
REPORTS = ("full_qformer_scale_eval_v1", "full_qformer_continue_eval_v1",
           "feedback_opd_eval_v2", "language_lora_eval_v1", "feedback_opd_weight_eval_v1")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "mpp_resume_experiment_archive_v1",
                "exported_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(ROOT), "files": [], "checkpoints": {},
                "note": "No weights, optimizer/RNG tensors, images, caches or per-question predictions are published. "
                        "Reported weight identities remain in the original reports; inventory below uses file sizes, not newly computed tensor hashes."}
    sources = []
    for key, (run, label) in RUNS.items():
        folder = ROOT / "outputs" / run
        for name in ("run_config.json", "summary.json"):
            sources.append((folder / name, key + "__" + name))
        bundle = folder / label if label else folder
        sources.append((bundle / "metadata.json", key + "__metadata.json"))
        adapter = bundle / ("adapter" if label else "final_adapter")
        sources.append((adapter / "adapter_config.json", key + "__adapter_config.json"))
        wanted = [(adapter / "adapter_model.safetensors", "language_lora"),
                  (bundle / ("projector.pt" if label else "final_projector.pt"), "projector"),
                  (bundle / ("qformer_adapter.pt" if label else "final_qformer_adapter.pt"), "complete_qformer"),
                  (bundle / "training_state.pt", "optimizer_scheduler_and_rank_rng")]
        manifest["checkpoints"][key] = {"run": str(folder), "selected_bundle": str(bundle),
            "files": [{"path": str(path), "role": role, "present": path.is_file(),
                       "bytes": path.stat().st_size if path.is_file() else None}
                      for path, role in wanted]}
    for report in REPORTS:
        for name in ("report.json", "report.md", "official_report.json", "official_report.md"):
            sources.append((ROOT / "outputs" / report / name, report + "__" + name))
    for source, name in sources:
        if not source.is_file():
            continue
        data = source.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("archive accepts only compact metadata: %s" % source)
        if source.suffix == ".json":
            json.loads(data)
        target = destination / name
        target.write_bytes(data)
        manifest["files"].append({"source": str(source.relative_to(ROOT)), "published": name,
                                  "bytes": len(data), "sha256": digest(data)})
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"destination": str(destination), "metadata_files": len(manifest["files"]),
            "tensor_files_published": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/artifacts/20261014")
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir), ensure_ascii=False, indent=2))
