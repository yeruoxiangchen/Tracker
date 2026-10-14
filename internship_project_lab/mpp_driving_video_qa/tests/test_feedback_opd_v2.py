import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, load_jsonl, file_sha256, build_user_prompt, PROTOCOL
from mpp_video_qa.data import cache_file
from mpp_video_qa.feedback_opd import feedback_prompt, feedback_protocol, validate_matched_configs, MATCHED_KEYS
from scripts.feedback_opd_common import current_rollout
from scripts.compare_feedback_teachers import select_cases, comparison_report
from scripts.prepare_feedback_opd import prepare
from scripts.train_feedback_opd import optimizer_budget, validate_probe
from scripts.report_feedback_opd_v2 import (select_checkpoints, model_key, LABELS, BRANCHES,
                                          checkpoint_specs, build_report, save_report)
from scripts.report_feedback_opd import identities
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_preference import summarize


def original(qid, source='action', segment=None):
    segment = segment or qid
    return {'question_id': qid, 'segment_id': segment, 'source_subset': source,
            'frames': ['/%s/%d.jpg' % (segment, i) for i in range(5)],
            'question': 'Which direction?', 'answers': ['turn right'], 'feedback_training_source': 'train'}


def draw(row, sample=0):
    return {'question_id': row['question_id'], 'segment_id': row['segment_id'], 'source_subset': row['source_subset'],
            'question': row['question'], 'references': row['answers'], 'frames': row['frames'],
            'sample_id': sample, 'student_answer': 'turn left', 'student_token_ids': [8, 2],
            'student_finished': True, 'student_judge': {'logit': -2., 'correct': False}, 'gate_reason': 'eligible'}


class FeedbackV2Tests(unittest.TestCase):
    def test_reference_only_removes_entire_wrong_draft_not_student_prefix(self):
        row = original('train')
        draft = 'WRONG_DRAFT_UNIQUE turn left'
        prompt = feedback_prompt(row, draft, teacher_mode='reference_only')
        self.assertNotIn(draft, prompt)
        self.assertNotIn('previous_student_answer', prompt)
        self.assertIn(row['answers'][0], prompt)
        self.assertEqual(prompt.count('<ImageHere>'), 5)
        self.assertNotIn(row['answers'][0], build_user_prompt(row['question'], 5))
        self.assertNotEqual(feedback_protocol('reference_only'), feedback_protocol('reference_draft'))
        with self.assertRaises(ValueError):
            feedback_prompt(dict(row, feedback_training_source='eval'), draft, teacher_mode='reference_only')

    def test_dynamic_rollout_passes_same_mode_prompt_and_keeps_student_ids(self):
        tokenizer = types.SimpleNamespace(decode=lambda ids, **kw: 'turn left' if ids[0] == 8 else 'turn right')
        student = types.SimpleNamespace(task_instruction='instruction', backbone=types.SimpleNamespace(llm_tokenizer=tokenizer))
        teacher = types.SimpleNamespace(active=lambda: contextlib.nullcontext(student))
        judge = types.SimpleNamespace(score=lambda r, text: {'logit': 2. if text == 'turn right' else -2.,
                                                            'correct': text == 'turn right'})
        with patch('scripts.feedback_opd_common.sample_completions', side_effect=[
                (torch.tensor([8, 2]), True), (torch.tensor([9, 2]), True)]) as sampler:
            ids, prompt, result = current_rollout(student, teacher, judge, torch.zeros(1), original('a'), 96, 'reference_only')
        self.assertEqual(ids.tolist(), [8, 2])
        self.assertEqual(sampler.call_args_list[1].kwargs['prompt'], prompt)
        self.assertFalse(sampler.call_args_list[1].kwargs['do_sample'])
        self.assertNotIn('turn left', prompt)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['teacher_mode'], 'reference_only')

    def test_adaptive_sampling_preserves_qa_and_source_quota_beyond_two_per_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cache = root / 'source', root / 'cache'
            train = [original('a%d' % i, segment='a-video-%d' % (i % 4)) for i in range(16)]
            train += [original('s%d' % i, 'scenery', 's-video-%d' % (i % 2)) for i in range(12)]
            for name, rows in (('train', train), ('dev', [original('dev')]), ('dev_action', [original('dev')]),
                               ('dev_scenery', [original('devs', 'scenery')]), ('eval', [original('eval')])):
                dump_jsonl(source / (name + '.jsonl'), rows)
            for row in train:
                target = cache_file(cache, row['segment_id'])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            args = types.SimpleNamespace(source_dir=source, output_dir=root/'out', train_size=28,
                                         probe_size=6, seed=44, max_qa_per_segment=0, cache_dir=cache)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
                prepare(args)
            selected = load_jsonl(root/'out/train.jsonl')
            self.assertEqual({r['question_id'] for r in selected}, {r['question_id'] for r in train})
            manifest = json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(manifest['sources'], {'action': 16, 'scenery': 12})
            self.assertEqual(manifest['segments'], 6)
            self.assertEqual(manifest['actual_max_qa_per_segment'], 6)
            args.train_size = 12
            with self.assertRaises(ValueError):
                prepare(args)  # Immutable v2 pool cannot be silently replaced.

    def test_paired_case_selection_unique_ids_original_tokens_and_no_held_out(self):
        rows = [original('a%d' % i) for i in range(6)] + [original('s%d' % i, 'scenery') for i in range(4)]
        originals = {r['question_id']: r for r in rows}
        samples = [draw(r, i) for r in rows for i in range(2)] + [draw(original('held_out'))]
        before = copy.deepcopy(samples)
        selected = select_cases(samples, originals, 10, 44)
        self.assertEqual(len({r['question_id'] for r in selected}), 10)
        self.assertNotIn('held_out', [r['question_id'] for r in selected])
        self.assertTrue(all(r['student_token_ids'] == [8, 2] for r in selected))
        self.assertEqual(samples, before)
        bad = copy.deepcopy(samples)
        bad[0]['references'] = ['dev answer']
        with self.assertRaises(ValueError):
            select_cases(bad, originals, 10, 44)

    def test_teacher_comparison_never_claims_fact_labels(self):
        row = draw(original('a'))
        row['modes'] = {mode: {'eligible': True, 'gate_reason': 'eligible', 'teacher_correction': 'turn right',
                              'tokens': [{'feedback_minus_plain': .5, 'jsd': .1}]}
                        for mode in ('reference_draft', 'reference_only')}
        report = comparison_report([row], {}, 1, 'sha')
        self.assertTrue(report['completed'])
        self.assertFalse(report['visual_supervision_direction_verified'])
        self.assertEqual(report['modes']['reference_only']['eligible_mean_sampled_logp_increase_draws'], 1)

    def test_paired_report_condition_and_data_identity_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'train.jsonl'
            dump_jsonl(data, [original('a')])
            identity = {'data_sha256': file_sha256(data), 'teacher_mode': 'reference_only'}
            report = {'protocol': 'feedback_teacher_comparison_v2', 'completed': True,
                      'modes': {'reference_only': {'eligible_draws': 1, 'token_diagnostic_draws': 1}},
                      'identity_by_mode': {'reference_only': identity}}
            (root/'report.json').write_text(json.dumps(report))
            (root/'manifest.json').write_text(json.dumps({'split_sha256': {'train': file_sha256(data)}}))
            args = types.SimpleNamespace(probe_report=root/'report.json', data=data, teacher_mode='reference_only')
            self.assertEqual(validate_probe(args, identity), file_sha256(root/'report.json'))
            with self.assertRaises(ValueError):
                validate_probe(args, dict(identity, data_sha256='changed'))
            args.teacher_mode = 'reference_draft'
            with self.assertRaises(ValueError):
                validate_probe(args, identity)

    def test_budget_and_matched_checkpoint_opportunities(self):
        self.assertEqual(optimizer_budget(40000, 4, 4, 2), 5000)
        self.assertEqual(optimizer_budget(10000, 4, 4, 1), 625)
        self.assertEqual(optimizer_budget(40000, 4, 4, 2, 2), 2)
        control = dict.fromkeys(MATCHED_KEYS, 1)
        control.update(training_mode='sft_control', teacher_mode='reference_only', checkpoint_steps=[1000, 2500])
        opd = dict(control, training_mode='feedback_opd')
        validate_matched_configs(control, opd)
        with self.assertRaises(ValueError):
            validate_matched_configs(control, dict(opd, teacher_mode='reference_draft'))
        with self.assertRaises(ValueError):
            validate_matched_configs(control, dict(opd, checkpoint_steps=[2500]))

    def test_each_branch_selects_own_dev_best_with_earlier_ties(self):
        specs = {model_key(b, label): {'branch': b, 'label': label, 'step': n}
                 for b in BRANCHES for label, n in zip(LABELS, (1000, 2500, 5000))}
        metrics = {k: {'lingo_judge_accuracy': .5} for k in specs}
        metrics[model_key('feedback_opd', 'checkpoint-002500')]['lingo_judge_accuracy'] = .6
        # No official scores are an input to this selector.
        selected = select_checkpoints(metrics, specs)
        self.assertEqual(selected['sft_control'], 'sft_control_checkpoint-001000')
        self.assertEqual(selected['feedback_opd'], 'feedback_opd_checkpoint-002500')

    def test_checkpoint_report_official_has_both_branches_and_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = types.SimpleNamespace(data_dir=root/'data', initial_dir=root/'initial',
                                         train_dir=root/'opd', control_dir=root/'sft', eval_dir=root/'evals')
            (args.initial_dir/'final_adapter').mkdir(parents=True)
            (args.initial_dir/'final_adapter/config.json').write_text('base')
            for name in ('final_projector.pt', 'final_qformer_adapter.pt'):
                (args.initial_dir/name).write_text('base')
            dump_jsonl(args.data_dir/'train.jsonl', [original('train')])
            parent = identities(args.initial_dir, True)
            common = dict.fromkeys(MATCHED_KEYS, 1)
            common.update(data_sha256=file_sha256(args.data_dir/'train.jsonl'),
                          initialization={'adapter': parent['adapter_identity'], 'projector': parent['projector_identity'],
                                          'qformer': parent['qformer_adapter_sha256']},
                          records=40000, world_size=4, grad_accumulation=4, epochs=2, total_steps=5000,
                          teacher_mode='reference_only', checkpoint_steps=[1000, 2500])
            for branch, folder in zip(BRANCHES, (args.control_dir, args.train_dir)):
                for label, step in zip(LABELS, (1000, 2500, 5000)):
                    bundle = folder/label
                    (bundle/'adapter').mkdir(parents=True)
                    (bundle/'adapter/config.json').write_text(branch+label)
                    for name in ('projector.pt', 'qformer_adapter.pt'):
                        (bundle/name).write_text(branch+label)
                    (bundle/'metadata.json').write_text(json.dumps({'identity': dict(common, training_mode=branch),
                                                                  'optimizer_steps': step}))
                (folder/'run_config.json').write_text(json.dumps(dict(common, training_mode=branch)))
                (folder/'summary.json').write_text(json.dumps({'completed': True, 'optimizer_steps': 5000}))
            hashes, specs = checkpoint_specs(args)
            for split in ('action', 'scenery', 'eval'):
                data = args.data_dir/('eval.jsonl' if split == 'eval' else 'dev_%s.jsonl' % split)
                originals = [original(split+'1'), original(split+'2')]
                dump_jsonl(data, originals)
                for key in hashes:
                    # Dev chooses each branch's 2500-step snapshot; official prefers the
                    # final snapshots, which must NOT alter that prior selection.
                    correct = ('checkpoint-002500' in key) if split != 'eval' else key.endswith('final')
                    predictions = [dict(question_id=r['question_id'], segment_id=r['segment_id'], question=r['question'],
                        references=r['answers'], prediction='answer', lingo_judge_correct=bool(i == 0 or correct),
                        token_f1=.5, normalized_exact=0.) for i, r in enumerate(originals)]
                    folder = args.eval_dir/(key+'_'+split)
                    dump_jsonl(folder/'predictions.jsonl', predictions)
                    dump_jsonl(folder/'scored/scored_predictions.jsonl', predictions)
                    (folder/'scored/summary.json').write_text(json.dumps(dict(summarize(predictions), judge_protocol=JUDGE_PROTOCOL)))
                    protocol = {'protocol': PROTOCOL, 'cache_dir': 'same', 'feature_stage': 'eva',
                        'task_instruction_version': 'legacy_v1', 'task_instruction': 'same', 'precision_protocol': 'fp32_trainables_v1',
                        'dtype': 'bf16', 'max_new_tokens': 96, 'max_records': 0, 'batch_size': 1,
                        'bridge_init_sha256': 'same', 'data_sha256': file_sha256(data)}
                    (folder/'run_manifest.json').write_text(json.dumps(dict(protocol, **hashes[key])))
            dev = build_report(args)
            self.assertEqual(set(dev['dev_combined']['same_step_control_vs_opd']), set(LABELS))
            self.assertEqual(dev['selected_checkpoints'], {b: model_key(b, 'checkpoint-002500') for b in BRANCHES})
            with contextlib.redirect_stdout(io.StringIO()):
                save_report(args, dev)
            official = build_report(args, official=True)
            self.assertEqual(set(official['splits']['eval']['models']),
                             {'baseline'} | set(dev['selected_checkpoints'].values()))
            self.assertEqual(official['selected_checkpoints'], dev['selected_checkpoints'])
            (args.train_dir/'checkpoint-002500/projector.pt').write_text('changed')
            with self.assertRaises(ValueError):
                build_report(args, official=True)


if __name__ == '__main__':
    unittest.main()
