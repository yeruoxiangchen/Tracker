from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PosttrainEvalScheduleTest(unittest.TestCase):
    def test_two_gpu_registered_route_is_dev48_only(self):
        wrapper = (ROOT / "official_ss_with_vggt_perf_v1/run_source_posttrain_2gpu03.sh").read_text()
        self.assertIn("EVALUATE_TRAIN64=0", wrapper)
        self.assertIn("SLAT_STEPS=10000,15000", wrapper)

    def test_dev48_route_reuses_checkpoint_independent_artifacts(self):
        endpoint = (
            ROOT
            / "official_ss_with_vggt_perf_v1/run_predicted_support_endpoint_train64_dev48.sh"
        ).read_text()
        split = (
            ROOT / "official_ss_with_vggt_perf_v1/run_predicted_support_endpoint_split.sh"
        ).read_text()
        self.assertIn('if (( EVALUATE_TRAIN64 == 1 )); then', endpoint)
        self.assertIn('EVALUATE_TRAIN64=${EVALUATE_TRAIN64:-0}', endpoint)
        self.assertIn('REUSE_INDEPENDENT_ROOT="${REFERENCE_OUTPUT_ROOT}/dev48_predicted"', endpoint)
        self.assertIn("reuse_endpoint_artifacts", split)

    def test_30k_resume_route_excludes_70k(self):
        resume = (
            ROOT
            / "pose_point_depth_mv/background_jobs/"
            "run_proobjaverse_slat29861_legacy_eval_4gpu4567_steps10k30k60k_resume.sh"
        ).read_text()
        self.assertIn("STEPS=10000,30000,60000", resume)
        self.assertNotIn("STEPS=10000,30000,60000,70000", resume)
        self.assertIn("PRESERVE_INTERRUPTED_OUTPUTS=1", resume)


if __name__ == "__main__":
    unittest.main()
