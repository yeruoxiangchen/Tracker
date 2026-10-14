"""Batch budget, loss scaling and real CPU/Gloo collective tests (no MPP load)."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from datetime import timedelta

import torch
from torch.utils.data import DistributedSampler

from mpp_video_qa.global_batch import ExactGlobalBatchSampler, weight_batch_layout
from mpp_video_qa.opd_runtime import GradientBuckets, PhaseTimer, validate_resume_identity
from mpp_video_qa.opd import distillation_loss
from scripts.train_feedback_opd import optimizer_budget, archive_failures, backward_example
from scripts.report_feedback_opd_weight import validate_weight_configs, status, training_progress, main, build_report
from test_feedback_opd_weight import configs, fixture


def toy_backward(parameters, index, divisor):
    weight, gated = parameters
    ce = (weight * ((index + 1) / 10.) - ((index % 7) - 3) / 10.) ** 2
    (ce / divisor).backward()
    if index % 3 == 0:
        # Only some examples/ranks have a second loss and parameter gradient.
        jsd = (gated + weight - index / 100.) ** 2
        (3 * jsd / divisor).backward()


def cpu_collective_worker(rank, world, rendezvous, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous,
                                         rank=rank, world_size=world, timeout=timedelta(seconds=45))
    try:
        parameters = [torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for v in (.2, -.1)]
        optimizer = torch.optim.AdamW(parameters, lr=.003)
        reducer = GradientBuckets(parameters, world, .01)
        sampler = ExactGlobalBatchSampler(33, 16, world, rank, 44)
        gradients, counts, visited = [], [], 0
        for epoch in range(2):
            sampler.set_epoch(epoch)
            batch_count = 0
            for micro, index in enumerate(sampler):
                if index is not None:
                    toy_backward(parameters, index, sampler.loss_divisor(micro))
                    batch_count += 1
                    visited += 1
                if (micro + 1) % sampler.accumulation == 0:
                    reducer.reduce()
                    gradients.append([p.grad.item() for p in parameters])
                    count = torch.tensor(batch_count, dtype=torch.int64)
                    torch.distributed.all_reduce(count)
                    counts.append(count.item())
                    torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    batch_count = 0
        torch.save({'weights': [p.item() for p in parameters], 'gradients': gradients,
                    'batch_counts': counts, 'visited': visited}, Path(directory) / ('rank%d.pt' % rank))
    finally:
        torch.distributed.destroy_process_group()


def toy_reference():
    parameters = [torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for v in (.2, -.1)]
    optimizer = torch.optim.AdamW(parameters, lr=.003)
    gradients = []
    for epoch in range(2):
        permutation = torch.randperm(33, generator=torch.Generator().manual_seed(44 + epoch)).tolist()
        for start in range(0, 33, 16):
            indices = permutation[start:start + 16]
            for index in indices:
                toy_backward(parameters, index, len(indices))
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            gradients.append([p.grad.item() for p in parameters])
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {'weights': [p.item() for p in parameters], 'gradients': gradients}


class FeedbackGPULayoutTests(unittest.TestCase):
    def test_full_pool_budget_and_no_duplicate_padding(self):
        for world, accumulation in ((2, 8), (3, 6), (4, 4)):
            with self.subTest(world=world):
                samplers = [ExactGlobalBatchSampler(40000, 16, world, r, 44) for r in range(world)]
                indices = [list(s) for s in samplers]
                self.assertEqual(optimizer_budget(40000, world, accumulation, 2, global_batch_size=16), 5000)
                self.assertEqual(optimizer_budget(40000, world, accumulation, 2, 2, 16), 2)
                real = [i for rows in indices for i in rows if i is not None]
                self.assertEqual(len(real), 40000)
                self.assertEqual(set(real), set(range(40000)))
                self.assertEqual({len(rows) for rows in indices}, {2500 * accumulation})
                for start in (0, accumulation, len(indices[0]) - accumulation):
                    batch = [i for rows in indices for i in rows[start:start + accumulation] if i is not None]
                    self.assertEqual(len(batch), 16)
                    self.assertEqual(len(set(batch)), 16)
                if world == 3:
                    self.assertEqual([sum(i is not None for i in rows[:6]) for rows in indices], [6, 5, 5])
                    self.assertAlmostEqual(samplers[0].loss_divisor(0), 16 / 3.)

    def test_global_batches_match_four_gpu_sampler_and_resume_order(self):
        for epoch in range(2):
            old = []
            for rank in range(4):
                sampler = DistributedSampler(range(48), num_replicas=4, rank=rank, seed=44)
                sampler.set_epoch(epoch)
                old.append(list(sampler))
                exact = ExactGlobalBatchSampler(48, 16, 4, rank, 44)
                exact.set_epoch(epoch)
                self.assertEqual(list(exact), old[-1])
            expected = [sorted(i for rows in old for i in rows[start:start + 4]) for start in range(0, 12, 4)]
            for world in (2, 3):
                sampler = ExactGlobalBatchSampler(48, 16, world, 0, 44)
                shards = []
                for rank in range(world):
                    s = ExactGlobalBatchSampler(48, 16, world, rank, 44)
                    s.set_epoch(epoch)
                    shards.append(list(s))
                    resumed = list(s)[s.accumulation:]
                    self.assertEqual(resumed, shards[-1][s.accumulation:])
                actual = [sorted(i for rows in shards for i in rows[start:start + sampler.accumulation] if i is not None)
                          for start in range(0, len(shards[0]), sampler.accumulation)]
                self.assertEqual(expected, actual)

    def test_tail_normalizes_by_real_count_and_invalid_layout_rejected(self):
        for world in (2, 3, 4):
            sampler = ExactGlobalBatchSampler(17, 16, world, 1, 44)
            self.assertTrue(all(i is None for i in list(sampler)[sampler.accumulation:]))
            self.assertAlmostEqual(sampler.loss_divisor(sampler.accumulation), 1 / world)
        for arguments in ((0, 16, 3, 0, 44), (16, 2, 3, 0, 44), (16, 16, 3, 3, 44)):
            with self.assertRaises(ValueError):
                ExactGlobalBatchSampler(*arguments)

    def test_report_accepts_exact_layout_only_and_cannot_resume_different_world(self):
        control, reference, candidate = configs()
        for world in (2, 3):
            changed = dict(candidate, **weight_batch_layout(world))
            validate_weight_configs(control, reference, changed)
            with self.assertRaises(ValueError):
                validate_resume_identity(candidate, changed)
            for bad in (dict(changed, global_batch_size=18), dict(changed, grad_accumulation=4),
                        dict(changed, global_batch_protocol='repeated_padding'), dict(changed, lr=1e-3)):
                with self.assertRaises(ValueError):
                    validate_weight_configs(control, reference, bad)
        with self.assertRaises(ValueError):
            validate_weight_configs(control, reference, dict(candidate, world_size=3, grad_accumulation=6))

    def test_archived_failure_is_preserved_and_not_active_status(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'failure_rank_0.json').write_text('{"status":"failed","traceback":"OOM"}')
            (folder / 'progress.json').write_text('{"status":"loading_model"}')
            archive_failures(folder)
            self.assertEqual(training_progress(folder)['status'], 'loading_model')
            archived = list(folder.glob('failure_rank_0.before_retry_*'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(json.loads(archived[0].read_text())['traceback'], 'OOM')

    def test_status_with_existing_training_config_keeps_evaluation_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                status(args)
            value = json.loads(out.getvalue())
            self.assertEqual(value['evaluation']['final_action']['total'], 4)
            self.assertEqual(value['batch_layout']['effective_global_batch_size'], 16)

    def test_actual_two_three_gpu_command_chain_and_reject_mismatched_nproc(self):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / 'python_stub'
            stub.write_text('#!%s\nimport json, sys\nprint(json.dumps({"argv":sys.argv[1:]}))\n' % sys.executable)
            stub.chmod(0o755)
            for devices, world, accumulation in (('5,6', 2, 8), ('5,6,7', 3, 6)):
                env = dict(os.environ, PYTHON_BIN=str(stub), CUDA_VISIBLE_DEVICES=devices)
                env.pop('NPROC_PER_NODE', None)
                env.pop('FEEDBACK_RESUME_FROM', None)
                result = subprocess.run(['bash', 'scripts/run_feedback_opd_weight.sh', 'train'],
                                        cwd=Path(__file__).resolve().parents[1], env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                calls = [json.loads(s)['argv'] for s in result.stdout.splitlines()]
                self.assertIn('--nproc_per_node=%d' % world, calls[-1])
                for flag, val in (('--grad-accumulation', str(accumulation)), ('--global-batch-size', '16'),
                                  ('--opd-weight', '3'), ('--epochs', '2')):
                    self.assertEqual(calls[-1][calls[-1].index(flag) + 1], val)
                self.assertEqual(calls[0][calls[0].index('--launch-world-size') + 1], str(world))
            for devices, count in (('5,6,7', '4'), ('5,5', '2'), ('5', '1')):
                env = dict(os.environ, PYTHON_BIN=str(stub), CUDA_VISIBLE_DEVICES=devices, NPROC_PER_NODE=count)
                result = subprocess.run(['bash', 'scripts/run_feedback_opd_weight.sh', 'train'],
                                        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')

    def test_reference_ce_still_backprops_with_exact_divisor(self):
        p = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        model = SimpleNamespace(answer_only_ce=lambda *a: p * p)
        args = SimpleNamespace(answer_only_logits=True, ce_weight=1.)
        row = {'question_id':'a', 'question':'q', 'source_subset':'action'}
        record = {'row': row, 'features':torch.zeros(1), 'answer':'a'}
        values = backward_example(args, model, None, None, record, torch.device('cpu'),
                                  PhaseTimer('cpu'), None, 16 / 3., io.StringIO(), 0, 0)
        self.assertAlmostEqual(p.grad.item(), .4 * 3 / 16)
        self.assertEqual(values['examples'], 1)
        self.assertEqual(values['rollouts'], 0)

    def test_ce_and_gated_jsd_use_same_global_denominator(self):
        p = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        model = SimpleNamespace(answer_only_ce=lambda *a: p * p)
        teacher = SimpleNamespace(active=lambda: contextlib.nullcontext('frozen'))
        judge = SimpleNamespace(cpu_fallbacks=0)
        args = SimpleNamespace(answer_only_logits=True, ce_weight=1., opd_weight=3., chunk_tokens=1,
                               max_new_tokens=96, teacher_mode='reference_only', runtime_log_flush_every=1)
        record = {'row': {'question_id':'a', 'question':'q', 'source_subset':'action'},
                  'features':torch.zeros(1), 'answer':'a'}
        diagnostic = {'eligible': True, 'teacher_cache_hit':False, 'teacher_generation_called':True,
                      'student_judge':{'correct':False}, 'student_finished':True, 'teacher_correction':'a',
                      'teacher_finished':True, 'teacher_judge':{'correct':True}, 'gate_reason':'eligible'}
        target = torch.tensor([[.1, .9], [.2, .8]], dtype=torch.float64)
        logits = torch.stack((p.expand(2), -p.expand(2)), dim=-1)
        out = io.StringIO()
        with patch('scripts.train_feedback_opd.current_rollout', return_value=(torch.tensor([4, 2]), 'prompt', diagnostic)), \
             patch('scripts.train_feedback_opd.completion_logits', side_effect=[target, logits]):
            values = backward_example(args, model, teacher, judge, record, torch.device('cpu'), PhaseTimer('cpu'),
                                      None, 16 / 3., out, 0, 0)
        q = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        reference_jsd = distillation_loss(torch.stack((q.expand(2), -q.expand(2)), dim=-1), target, 'jsd', 1)
        ((q * q + 3 * reference_jsd) * 3 / 16).backward()
        self.assertAlmostEqual(p.grad.item(), q.grad.item(), places=10)
        self.assertEqual(values['eligible'], 1)
        self.assertEqual(values['examples'], 1)
        self.assertEqual(values['rollouts'], 1)
        self.assertEqual(json.loads(out.getvalue())['question_id'], 'a')

    def test_three_card_report_reuses_old_scores_and_discloses_layout(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('scripts.report_feedback_opd_weight.line_count', return_value=40000):
            args = fixture(Path(directory))
            config = json.loads((args.train_dir / 'run_config.json').read_text())
            config.update(weight_batch_layout(3))
            (args.train_dir / 'run_config.json').write_text(json.dumps(config))
            for label in ('checkpoint-001000', 'checkpoint-002500', 'final'):
                path = args.train_dir / label / 'metadata.json'
                metadata = json.loads(path.read_text())
                metadata['identity'] = config
                path.write_text(json.dumps(metadata))
            report = build_report(args)
            layout = report['runtime_provenance']['batch_layouts']['opd3']
            self.assertEqual(layout['world_size'], 3)
            self.assertEqual(layout['effective_global_batch_size'], 16)
            self.assertEqual(report['selected_checkpoints']['opd3'], 'opd3_final')

    def test_check_discloses_three_card_layout_and_rejects_started_four_card_change(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            argv = ['check', '--action', 'check', '--launch-world-size', '3', '--launch-grad-accumulation', '6',
                    '--launch-global-batch-size', '16']
            for key in ('data_dir','initial_dir','control_dir','reference_dir','reference_eval_dir','train_dir','eval_dir'):
                argv += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            with patch('scripts.report_feedback_opd_weight.line_count', return_value=40000), patch('sys.argv', argv):
                with self.assertRaisesRegex(ValueError, 'cannot change GPU layout'):
                    main()
                (args.train_dir / 'run_config.json').unlink()
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    main()
                result = json.loads(out.getvalue())
                self.assertEqual(result['world_size'], 3)
                self.assertEqual(result['grad_accumulation'], 6)
                self.assertEqual(result['effective_global_batch_size'], 16)

    def test_real_gloo_collectives_match_batch16_gradients_and_adam_updates(self):
        expected = toy_reference()
        for world in (2, 3):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as directory:
                torch.multiprocessing.spawn(cpu_collective_worker,
                    args=(world, str(Path(directory) / 'rendezvous'), directory), nprocs=world, join=True)
                total_visits = 0
                for rank in range(world):
                    actual = torch.load(Path(directory) / ('rank%d.pt' % rank))
                    total_visits += actual['visited']
                    self.assertEqual(actual['batch_counts'], [16, 16, 1, 16, 16, 1])
                    torch.testing.assert_close(torch.tensor(actual['gradients']), torch.tensor(expected['gradients']),
                                               atol=1e-7, rtol=1e-7)
                    torch.testing.assert_close(torch.tensor(actual['weights']), torch.tensor(expected['weights']),
                                               atol=1e-7, rtol=1e-7)
                self.assertEqual(total_visits, 66)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
