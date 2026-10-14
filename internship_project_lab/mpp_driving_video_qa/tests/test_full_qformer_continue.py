import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256
from scripts.report_full_qformer_continue import (
    CANDIDATES, JUDGE_PROTOCOL, build_report, identity, select_model, summarize, validate_training,
    parent_evaluation_identity,
)
from scripts.train_sft import continuation_provenance


class FullQFormerContinuationTests(unittest.TestCase):
    def fixture(self, root):
        args = types.SimpleNamespace(**{k: root / k for k in
            ("data_dir", "train_dir", "eval_dir", "parent_dir", "parent_eval_dir")})
        for path in vars(args).values():
            path.mkdir()
        dump_jsonl(args.data_dir / "train.jsonl", [{"question_id": "training-only"}])
        train_hash = file_sha256(args.data_dir / "train.jsonl")
        (args.data_dir / "manifest.json").write_text(json.dumps({
            "sampling": "all_original_training_rows", "split_sha256": {"train": train_hash}}))
        parent_config = dict(data_sha256=train_hash, total_steps=25389, records=406212, epochs=1,
                             qformer_mode="full", feature_stage="eva", precision_protocol="fp32_trainables_v1")
        (args.parent_dir / "run_config.json").write_text(json.dumps(parent_config))
        (args.parent_dir / "summary.json").write_text(json.dumps({"completed": True, "optimizer_steps": 25389}))
        for model in ("baseline",) + CANDIDATES:
            final = model in ("baseline", "final")
            weights = args.parent_dir if model == "baseline" else (
                args.train_dir if final else args.train_dir / model)
            weights.mkdir(exist_ok=True)
            adapter = weights / ("final_adapter" if final else "adapter")
            adapter.mkdir()
            (adapter / "weights.bin").write_bytes(model.encode())
            (weights / ("final_projector.pt" if final else "projector.pt")).write_bytes(b"projection" + model.encode())
            (weights / ("final_qformer_adapter.pt" if final else "qformer_adapter.pt")).write_bytes(b"bridge" + model.encode())
            if not final:
                (weights / "metadata.json").write_text(json.dumps({"global_step": int(model.split("-")[1]),
                    "data_sha256": train_hash, "total_steps": 25389}))
            hashes = identity(weights, final)
            for split in ("action", "scenery"):
                path = args.data_dir / ("dev_%s.jsonl" % split)
                dump_jsonl(path, [{"question_id": split, "segment_id": split, "question": "q", "answers": ["yes"]}])
                folder = (args.parent_eval_dir / ("epoch-01_" + split) if model == "baseline" else
                          args.eval_dir / (model + "_" + split))
                (folder / "scored").mkdir(parents=True)
                correct = model == CANDIDATES[0] or (model == "baseline" and split == "scenery")
                row = dict(question_id=split, segment_id=split, question="q", references=["yes"],
                           prediction="yes" if correct else "no")
                dump_jsonl(folder / "predictions.jsonl", [row])
                scored = dict(row, lingo_judge_correct=correct, token_f1=float(correct), normalized_exact=float(correct))
                dump_jsonl(folder / "scored/scored_predictions.jsonl", [scored])
                (folder / "scored/summary.json").write_text(json.dumps(dict(summarize([scored]), judge_protocol=JUDGE_PROTOCOL)))
                run = dict(hashes, data_sha256=file_sha256(path), protocol="mpp_driving_video_qa_v1",
                           cache_dir="/same/cache", feature_stage="eva", precision_protocol="fp32_trainables_v1",
                           task_instruction_version="legacy_v1", task_instruction="same", dtype="bf16",
                           max_new_tokens=96, max_records=0, batch_size=1, bridge_init_sha256="same")
                (folder / "run_manifest.json").write_text(json.dumps(run))
        hashes = identity(args.parent_dir, True)
        epoch = args.parent_dir / "epoch-01"
        epoch.mkdir()
        shutil.copytree(args.parent_dir / "final_adapter", epoch / "adapter")
        for name in ("projector.pt", "qformer_adapter.pt"):
            shutil.copyfile(args.parent_dir / ("final_" + name), epoch / name)
        child = dict(parent_config, parent_run=str(args.parent_dir), parent_optimizer_steps=25389,
                     save_every=6000, max_steps=0, seed=43, lr=2.5e-6, projector_lr=2.5e-6, qformer_lr=5e-6,
                     initial_adapter_sha256=hashes["adapter_identity"], initial_projector_sha256=hashes["projector_identity"],
                     initial_qformer_sha256=hashes["qformer_adapter_sha256"])
        (args.train_dir / "run_config.json").write_text(json.dumps(child))
        (args.train_dir / "summary.json").write_text(json.dumps({"completed": True, "optimizer_steps": 25389}))
        return args

    def test_best_middle_checkpoint_selected_without_official_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            (args.data_dir / "eval.jsonl").write_text("not consulted for development selection")
            self.assertTrue(validate_training(args, required=True)["config_present"])
            report = build_report(args)
            self.assertEqual(report["selected_model"], "checkpoint-006000")
            self.assertEqual(report["dev_combined"]["paired"]["baseline_vs_checkpoint-006000"]
                             ["lingo_judge_correct"]["fixed"], 1)
            self.assertNotIn("eval", report["splits"])
            self.assertEqual(select_model({m: {"lingo_judge_accuracy": .5} for m in
                                           ("final", "baseline", "checkpoint-006000")}), "baseline")

    def test_rejects_wrong_bridge_and_unfinished_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            wrong = args.train_dir / "checkpoint-006000/qformer_adapter.pt"
            wrong.write_bytes(b"wrong bridge")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_report(args)
            (args.train_dir / "summary.json").write_text(json.dumps({"completed": False, "optimizer_steps": 24000}))
            with self.assertRaisesRegex(ValueError, "full budget"):
                build_report(args, models=("baseline", "final"))

    def test_parent_bundle_and_continuation_resume_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            train_args = types.SimpleNamespace(parent_run=str(args.parent_dir), output_dir=str(args.train_dir),
                data=str(args.data_dir / "train.jsonl"), resume_from="",
                init_adapter=str(args.parent_dir / "final_adapter"), projector=str(args.parent_dir / "final_projector.pt"),
                qformer_adapter=str(args.parent_dir / "final_qformer_adapter.pt"))
            self.assertEqual(continuation_provenance(train_args)["parent_optimizer_steps"], 25389)
            train_args.qformer_adapter = str(args.train_dir / "final_qformer_adapter.pt")
            with self.assertRaisesRegex(ValueError, "complete parent weight bundle"):
                continuation_provenance(train_args)
            train_args.resume_from = str(args.train_dir / "checkpoint-006000")
            self.assertEqual(continuation_provenance(train_args)["parent_optimizer_steps"], 25389)
            train_args.seed = 99
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                continuation_provenance(train_args)
            del train_args.seed
            train_args.resume_from = str(args.parent_dir / "checkpoint-006000")
            with self.assertRaisesRegex(ValueError, "different continuation"):
                continuation_provenance(train_args)

    def test_reuses_different_archives_only_when_tensors_and_metadata_equal(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            payload = {"format": "full", "state_dict": {"weight": torch.ones(3)}}
            final = args.parent_dir / "final_qformer_adapter.pt"
            epoch = args.parent_dir / "epoch-01/qformer_adapter.pt"
            torch.save(payload, final)
            torch.save(payload, epoch)
            self.assertNotEqual(file_sha256(final), file_sha256(epoch))
            self.assertEqual(parent_evaluation_identity(args.parent_dir)["qformer_adapter_sha256"], file_sha256(epoch))
            payload["state_dict"]["weight"][0] = 2
            torch.save(payload, epoch)
            with self.assertRaisesRegex(ValueError, "tensors differ"):
                parent_evaluation_identity(args.parent_dir)

    def test_launcher_loads_all_weights_and_resumes_only_child_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "python-recorder"
            executable.write_text("#!%s\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n" % sys.executable)
            executable.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(executable), CONTINUE_PARENT_DIR=str(root / "parent"),
                       CONTINUE_TRAIN_DIR=str(root / "child"), CONTINUE_DATA_DIR=str(root / "data"))
            for key in ("SFT_RESUME_FROM", "SFT_LR", "PROJECTOR_LR", "QFORMER_LR", "SFT_SEED"):
                env.pop(key, None)
            command = ["bash", str(ROOT / "scripts/run_full_qformer_continue.sh"), "train"]
            def call():
                output = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                calls = [json.loads(line) for line in output.stdout.splitlines()]
                return next(c for c in calls if "scripts/train_sft.py" in c)
            train = call()
            expected = {"--init-adapter": str(root / "parent/final_adapter"),
                        "--projector": str(root / "parent/final_projector.pt"),
                        "--qformer-adapter": str(root / "parent/final_qformer_adapter.pt"),
                        "--parent-run": str(root / "parent"), "--epochs": "1", "--save-every": "6000",
                        "--seed": "43", "--lr": "2.5e-6", "--projector-lr": "2.5e-6", "--qformer-lr": "5e-6",
                        "--qformer-mode": "full", "--precision-protocol": "fp32_trainables_v1"}
            for k, value in expected.items():
                self.assertEqual(train[train.index(k) + 1], value)
            self.assertNotIn("--resume-from", train)
            env["SFT_RESUME_FROM"] = str(root / "child/checkpoint-006000")
            train = call()
            self.assertIn("--resume-from", train)
            self.assertNotIn("--init-adapter", train)
            self.assertNotIn("--qformer-adapter", train)


if __name__ == "__main__":
    unittest.main()
