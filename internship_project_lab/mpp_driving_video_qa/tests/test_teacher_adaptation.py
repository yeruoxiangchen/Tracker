import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import TASK_INSTRUCTIONS, dump_jsonl, file_sha256
from scripts.generate_vlm_teacher import build_teacher_inputs, model_identity, write_json
from scripts.teacher_adaptation import (build_report, checked_variant, generate_variant, prepare,
                                       VARIANTS, INSTRUCTION)
from scripts.teacher_distillation import check_teacher_settings
from scripts.report_preference import summarize
from scripts.report_answer_opsd import JUDGE_PROTOCOL


class RecordingProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(merge_size=2)
        self.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=2)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "rendered"

    def __call__(self, **kwargs):
        import torch
        self.kwargs = kwargs
        key = "video_grid_thw" if "videos" in kwargs else "image_grid_thw"
        grid = [[3, 4, 6]] if "videos" in kwargs else [[1, 4, 6]] * 5
        return {key: torch.tensor(grid)}


def fixture(root):
    model, source, output = root / "model", root / "source", root / "new"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test-model")
    originals = [dict(question_id=str(i), segment_id="seg%d" % (i // 2),
        source_subset="action" if i < 4 else "scenery", frames=["/%d/%d.jpg" % (i, j) for j in range(5)],
        question="Why are you slowing down?" if i % 2 else "What is ahead?", answers=["reference %d" % i])
        for i in range(6)]
    data = source / "data/dev_probe.jsonl"
    dump_jsonl(data, originals)
    common = dict(data_sha256=file_sha256(data), task_instruction_version="legacy_v1")
    teacher = dict(common, model=model_identity(model), min_pixels=50176, max_pixels=1003520,
                   max_new_tokens=96, decoding="greedy", references_in_prompt=False,
                   transformers_version="4.57.6")
    def save(role, correct, folder, manifest):
        rows = [dict(question_id=r["question_id"], segment_id=r["segment_id"], question=r["question"],
            source_subset=r["source_subset"], references=r["answers"], prediction="reply " + r["question_id"],
            finished=not (role == "multiimage_complete" and i == 4),
            lingo_judge_correct=i in correct, token_f1=float(i in correct), normalized_exact=0.)
            for i, r in enumerate(originals)]
        dump_jsonl(folder / "predictions.jsonl", rows)
        dump_jsonl(folder / "scored/scored_predictions.jsonl", rows)
        write_json(folder / "scored/summary.json", dict(summarize(rows), judge_protocol=JUDGE_PROTOCOL))
        write_json(folder / "run_manifest.json", manifest)
    save("student", {0, 1, 2, 3}, source / "probe_dev/student", common)
    save("teacher", {0}, source / "probe_dev/teacher", teacher)
    for name, mode in VARIANTS.items():
        manifest = dict(teacher, input_mode=mode, task_instruction_version=INSTRUCTION,
                        task_instruction=TASK_INSTRUCTIONS[INSTRUCTION], max_records=0)
        if mode == "video":
            manifest.update(frame_interval_seconds=1.0, do_sample_frames=False,
                            temporal_padding="repeat_last_before_resize")
        save(name, {0, 4} if mode == "multi_image" else {0, 5}, output / name, manifest)
    args = SimpleNamespace(source_run=source, output_dir=output, teacher_model=model,
                           frame_interval_seconds=1.0, review_limit=0,
                           teacher_python="teacher-python", python=sys.executable)
    return args, originals


class TeacherAdaptationTests(unittest.TestCase):
    def test_video_preserves_all_five_frames_padding_and_timestamps_without_answers(self):
        from PIL import Image
        import torch
        images = [Image.new("RGB", (64, 32), color=(i * 20, 0, 0)) for i in range(5)]
        try:
            row = dict(question="Why are you slowing?", frames=["/%d.jpg" % i for i in range(5)],
                       answers=["SECRET_REFERENCE"])
            processor = RecordingProcessor()
            _, observation = build_teacher_inputs(row, images, processor, TASK_INSTRUCTIONS[INSTRUCTION], "video", 1.0)
            tensor = processor.kwargs["videos"][0]
            self.assertEqual(tuple(tensor.shape), (6, 3, 32, 64))
            self.assertEqual(tensor[:, 0, 0, 0].tolist(), [0, 20, 40, 60, 80, 80])
            self.assertTrue(torch.equal(tensor[4], tensor[5]))
            self.assertFalse(processor.kwargs["do_sample_frames"])
            self.assertEqual(processor.kwargs["video_metadata"][0]["frames_indices"], [0, 1, 2, 3, 4, 4])
            self.assertEqual(observation["source_timestamps_seconds"], [0, 1, 2, 3, 4])
            self.assertEqual(observation["visual_tokens"], 18)
            self.assertNotIn("SECRET_REFERENCE", json.dumps(processor.messages))
            self.assertEqual(processor.messages[0]["content"][0]["type"], "video")
            _, multi = build_teacher_inputs(row, images, processor, TASK_INSTRUCTIONS[INSTRUCTION])
            self.assertEqual(multi["visual_tokens"], 30)
            self.assertEqual(processor.kwargs["images"], images)
        finally:
            for image in images:
                image.close()

    def test_invalid_video_interval_is_not_silently_defaulted(self):
        for interval in (0, -1, float("nan")):
            with self.assertRaisesRegex(ValueError, "interval"):
                build_teacher_inputs({}, [None] * 5, RecordingProcessor(), "instruction", "video", interval)

    def test_report_accepts_weaker_complementary_teacher_and_masks_unfinished(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, originals = fixture(Path(temporary))
            with patch("builtins.print"):
                result = build_report(args)
            self.assertEqual(result["records"], 6)
            self.assertEqual(len(result["groups"]["all"]["models"]), 4)
            self.assertEqual(result["groups"]["all"]["paired"]["student_vs_multiimage_complete"]
                             ["lingo_judge_correct"]["fixed"], 1)
            assessment = result["teacher_assessment"]["multiimage_complete"]
            self.assertEqual(assessment["potential_corrections"], 0)
            self.assertFalse(assessment["overall_advantage_detected"])
            self.assertEqual(result["teacher_assessment"]["video_complete"]["potential_corrections"], 1)
            self.assertTrue((args.output_dir / "report.md").is_file())
            self.assertFalse(result["review"]["visually_reviewed"])

    def test_changed_video_timing_cannot_be_compared_as_same_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, originals = fixture(Path(temporary))
            config, data, _, _, _ = prepare(args)
            path = args.output_dir / "video_complete/run_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["frame_interval_seconds"] = .5
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                checked_variant(args, "video_complete", config, data, originals)

    def test_generation_and_existing_distillation_receive_teacher_specific_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, originals = fixture(Path(temporary))
            config, data, _, _, _ = prepare(args)
            with patch("scripts.teacher_adaptation.run") as runner, patch("scripts.teacher_adaptation.score"):
                generate_variant(args, "video_complete", config, data)
                command = [str(x) for x in runner.call_args.args[1:]]
                self.assertEqual(command[command.index("--input-mode") + 1], "video")
                self.assertEqual(command[command.index("--task-instruction-version") + 1], INSTRUCTION)
            settings = SimpleNamespace(teacher_input_mode="video", teacher_task_instruction_version=INSTRUCTION,
                                       teacher_frame_interval_seconds=1.)
            manifest = json.loads((args.output_dir / "video_complete/run_manifest.json").read_text())
            check_teacher_settings(settings, manifest)
            manifest["task_instruction_version"] = "legacy_v1"
            with self.assertRaisesRegex(ValueError, "instruction"):
                check_teacher_settings(settings, manifest)

    def test_status_works_from_outside_project_before_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(["bash", str(ROOT / "scripts/run_teacher_adaptation.sh"), "status",
                "--output-dir", temporary, "--source-run", temporary], cwd=temporary,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(set(value["variants"]), set(VARIANTS))
            self.assertFalse(value["report_ready"])


if __name__ == "__main__":
    unittest.main()
