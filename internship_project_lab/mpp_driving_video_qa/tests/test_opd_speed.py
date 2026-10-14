import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.model import CachedVideoQA, assistant_only_preprocess
from mpp_video_qa.opd import completion_logits, distillation_loss, average_gradients
from mpp_video_qa.opd_runtime import GradientBuckets, PhaseTimer, TeacherGateCache, validate_resume_identity
from scripts.feedback_opd_common import current_rollout, Judge


class Tokenizer:
    im_start_id, im_end_id, eod_id, pad_token_id = 1, 2, 0, 0

    def __call__(self, text):
        import re
        pieces = re.split('(<\\|extra_0\\|>)', text)
        return types.SimpleNamespace(input_ids=[v for part in pieces for v in
            ([7] if part == '<|extra_0|>' else [8 + ord(c) % 110 for c in part])])

    def decode(self, ids, **kwargs):
        return ' '.join('t%d' % int(i) for i in ids)


def reduce_worker(rank, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', rank=rank, world_size=2,
                                         init_method='file://' + directory + '/rendezvous')
    try:
        params = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3)),
                  nn.Parameter(torch.ones(2, dtype=torch.float64))]
        for index, p in enumerate(params):
            p.grad = None if rank == 0 and index == 0 else torch.full_like(p, (rank + 1) * (index + 1))
        old = [nn.Parameter(p.detach().clone()) for p in params]
        for original, p in zip(params, old):
            p.grad = None if original.grad is None else original.grad.clone()
        average_gradients(old, 2)
        buckets = GradientBuckets(params, 2, bucket_mb=1)
        buckets.reduce()
        for a, b in zip(old, params):
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        # Test reusable storage and a second reduction with a different local missing gradient.
        for a, b in zip(old, params):
            a.grad = None if rank else torch.full_like(a, 8)
            b.grad = None if a.grad is None else a.grad.clone()
        average_gradients(old, 2)
        buckets.reduce()
        for a, b in zip(old, params):
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        (Path(directory) / ('rank%d.json' % rank)).write_text(json.dumps({'passed': True, 'buckets': len(buckets.buckets)}))
    finally:
        torch.distributed.destroy_process_group()


class SpeedTests(unittest.TestCase):
    def model(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(7)
        core = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16,
                                         n_positions=2048, vocab_size=128,
                                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        llm = get_peft_model(core, LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'], task_type='CAUSAL_LM'))
        bridge = nn.Linear(6, 16)
        backbone = nn.Module()
        backbone.llm_model, backbone.llm_proj = llm, bridge
        backbone.llm_tokenizer, backbone.max_txt_len = Tokenizer(), 1800
        backbone.replace_image_string, backbone.replace_image_token_id = '<|extra_0|>', 7
        backbone.preprocess = types.MethodType(assistant_only_preprocess, backbone)
        return CachedVideoQA(backbone, feature_stage='qformer')

    def test_ce_logits_jsd_and_all_trainable_gradients(self):
        model = self.model().train()
        features = torch.randn(1, 5, 32, 6)
        ids = torch.tensor([10, 11, 2])
        teacher = torch.randn(3, 128)
        results = []
        for fast in (False, True):
            model.zero_grad(set_to_none=True)
            logits = completion_logits(model, features, 'q', ids, answer_only=fast)
            jsd = distillation_loss(logits, teacher, chunk_tokens=2)
            ce = model.answer_only_ce(features, ['q'], ['answer']) if fast else model(features, ['q'], ['answer']).loss
            (ce + jsd).backward()
            results.append((logits.detach(), ce.detach(), jsd.detach(),
                            {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        for a, b in zip(results[0][:3], results[1][:3]):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        self.assertEqual(results[0][3].keys(), results[1][3].keys())
        for name, grad in results[0][3].items():
            torch.testing.assert_close(grad, results[1][3][name], atol=1e-6, rtol=1e-5)
        self.assertTrue(any('.lora_B.' in n and g.abs().sum() > 0 for n, g in results[1][3].items()))
        self.assertTrue(results[1][3]['backbone.llm_proj.weight'].abs().sum() > 0)

    def test_two_rank_reduction_missing_gradients_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(reduce_worker, args=(directory,), nprocs=2, join=True)
            self.assertEqual(json.loads((Path(directory)/'rank0.json').read_text())['buckets'], 2)

    def test_teacher_cache_preserves_current_student_gate(self):
        model = types.SimpleNamespace(task_instruction='instruction', backbone=types.SimpleNamespace(llm_tokenizer=Tokenizer()))
        teacher = types.SimpleNamespace(active=lambda: contextlib.nullcontext(model))
        judge = types.SimpleNamespace(score=lambda r, text: {'correct': 't9' in text, 'logit': 2 if 't9' in text else -2})
        row = {'question_id': 'q', 'segment_id': 's', 'question': 'q', 'answers': ['answer'],
               'frames': ['/%d' % i for i in range(5)], 'feedback_training_source': 'train'}
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherGateCache(directory, {'teacher': 'fixed'})
            with patch('scripts.feedback_opd_common.sample_completions', side_effect=[
                    (torch.tensor([8, 2]), True), (torch.tensor([9, 2]), True),
                    (torch.tensor([10, 2]), True), (torch.tensor([9, 2]), True)]) as sample:
                first = current_rollout(model, teacher, judge, torch.zeros(1), row, 10, 'reference_only', cache)[2]
                second = current_rollout(model, teacher, judge, torch.zeros(1), row, 10, 'reference_only',
                                         TeacherGateCache(directory, {'teacher': 'fixed'}))[2]
                third = current_rollout(model, teacher, judge, torch.zeros(1), row, 10, 'reference_only', cache)[2]
            self.assertTrue(first['eligible'] and second['eligible'])
            self.assertFalse(first['teacher_cache_hit'])
            self.assertTrue(second['teacher_cache_hit'])
            self.assertFalse(second['teacher_generation_called'])
            self.assertNotEqual(first['student_token_ids'], second['student_token_ids'])
            self.assertFalse(third['eligible'])
            self.assertEqual(sample.call_count, 4)
            self.assertIsNone(TeacherGateCache(directory, {'teacher': 'changed'}).get(cache.key(row, 'p', 10)))

    def test_cpu_timer_and_strict_semantic_resume(self):
        timer = PhaseTimer(torch.device('cpu'), 2)
        timer.begin(0)
        with timer.phase('judge', 'cpu'):
            sum(range(100))
        self.assertEqual(timer.finish()['phases']['judge']['calls'], 1)
        timer.begin(1)
        self.assertIsNone(timer.finish())
        identity = {'teacher': 'fixed', 'world_size': 4, 'total_steps': 5000}
        validate_resume_identity(identity, copy.deepcopy(identity))
        with self.assertRaises(ValueError):
            validate_resume_identity(identity, dict(identity, world_size=8))

    def test_judge_gpu_boundary_uses_cpu_decision(self):
        judge = Judge.__new__(Judge)
        judge.device, judge.boundary_margin = torch.device('cuda'), .05
        judge.tokenizer, judge.model, judge.cpu_model = None, object(), None
        judge.model_name = 'mock'
        judge.cpu_fallbacks = 0
        with patch('scripts.feedback_opd_common.load_judge', return_value=(None, nn.Linear(1,1))) as load, \
             patch('scripts.feedback_opd_common.judge_score', side_effect=[(.01, True), (-.01, False), (2., True)]) as score:
            row = {'question':'q', 'answers':['a']}
            self.assertFalse(judge.score(row, 'a')['correct'])
            self.assertTrue(judge.score(row, 'a')['correct'])
        self.assertEqual(judge.cpu_fallbacks, 1)
        self.assertEqual(score.call_count, 3)
        load.assert_called_once_with('mock', torch.device('cpu'))

    def test_gate_cache_tracks_feature_file_updates(self):
        from mpp_video_qa.data import cache_file
        with tempfile.TemporaryDirectory() as directory:
            row = {'question_id':'q', 'segment_id':'s', 'frames':['1','2','3','4','5']}
            feature = cache_file(directory, 's')
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b'old')
            cache = TeacherGateCache(Path(directory)/'gates', {'cache_dir':directory})
            key = cache.key(row, 'prompt', 96)
            cache.put(key, {'text':'a', 'finished':True, 'score':{'logit':1.,'correct':True}})
            feature.write_bytes(b'new features')
            self.assertNotEqual(key, cache.key(row, 'prompt', 96))

    def test_fast_options_preserve_old_resume_identity(self):
        from scripts.train_feedback_opd import parse_args, training_config
        argv=['train','--data','train.jsonl','--output-dir','outputs/test','--epochs','2',
              '--teacher-mode','reference_only','--checkpoint-steps','1000','2500']
        with patch('sys.argv', argv):
            legacy=parse_args()
        with patch('sys.argv', argv+['--answer-only-logits','--gradient-bucket-mb','25',
                                     '--runtime-judge-device','local','--teacher-gate-cache-dir','cache']):
            fast=parse_args()
        identity={'teacher':'fixed','judge_device':'cpu'}
        original=training_config(legacy,identity,40000,4,5000,'probe',[1000,2500])
        accelerated=training_config(fast,identity,40000,4,5000,'probe',[1000,2500])
        validate_resume_identity(original,accelerated)
        self.assertEqual(accelerated['world_size']*accelerated['grad_accumulation'],16)

    def test_old_checkpoint_counters_keep_generation_totals(self):
        from scripts.train_feedback_opd import METRICS, restore_counters
        old=[0.]*17
        old[METRICS.index('examples')]=16000
        old[METRICS.index('teacher_attempts')]=7000
        current=torch.zeros(len(METRICS),dtype=torch.float64)
        restore_counters(current,old)
        self.assertEqual(current[METRICS.index('teacher_generations')],7000)
        self.assertEqual(current[METRICS.index('teacher_cache_hits')],0)
        current[METRICS.index('teacher_generations')]=7100
        restore_counters(current,current.tolist())
        self.assertEqual(current[METRICS.index('teacher_generations')],7100)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
