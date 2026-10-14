import copy
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

from mpp_video_qa.core import dump_jsonl, file_sha256, PROTOCOL
from mpp_video_qa.feedback_opd import MATCHED_KEYS
from mpp_video_qa.opd_diagnostics import digest, gradient_geometry
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_feedback_opd import identities
from scripts.report_feedback_opd_v2 import bundle_identity
from scripts.report_preference import summarize
from scripts.report_feedback_opd_weight import (
    ARMS, STEPS, validate_weight_configs, evaluation_path, select_checkpoints,
    checkpoint_specs, build_report, save_report, checked_configs, runtime_provenance)
from scripts.report_feedback_opd_weight import loss_summary
from scripts.diagnose_feedback_opd_weight import validate_fixed_rows, gradient_summary, prepare, make_report
from scripts import diagnose_feedback_opd as diagnostic
from mpp_video_qa.global_batch import weight_batch_layout
import torch


def configs():
    control = dict.fromkeys(MATCHED_KEYS, 1)
    control.update(records=40000, world_size=4, grad_accumulation=4, epochs=2, total_steps=5000,
                   checkpoint_steps=[1000, 2500], teacher_mode='reference_only', objective='jsd',
                   ce_weight=1., opd_weight=1., training_mode='sft_control', chunk_tokens=8,
                   checkpoint_root='/old', rollout_distribution='unchanged', judge_model='same')
    return control, dict(control, training_mode='feedback_opd'), dict(control, training_mode='feedback_opd', opd_weight=3.)


def original(qid):
    return {'question_id': qid, 'segment_id': qid, 'question': 'Which direction?', 'answers': ['turn right'],
            'frames': ['/%s/%d.jpg' % (qid, n) for n in range(5)]}


def fixture(root):
    args = SimpleNamespace(data_dir=root/'data', initial_dir=root/'parent', control_dir=root/'ce',
                           reference_dir=root/'opd1', reference_eval_dir=root/'historical',
                           train_dir=root/'opd3', eval_dir=root/'new', opd_weight=3.)
    (args.initial_dir/'final_adapter').mkdir(parents=True)
    (args.initial_dir/'final_adapter/adapter_config.json').write_text('parent')
    for name in ('final_projector.pt', 'final_qformer_adapter.pt'):
        (args.initial_dir/name).write_text('parent')
    dump_jsonl(args.data_dir/'train.jsonl', [original('train')])
    parent = identities(args.initial_dir, True)
    for arm, config in zip(ARMS, configs()):
        folder = getattr(args, {'ce': 'control_dir', 'opd1': 'reference_dir', 'opd3': 'train_dir'}[arm])
        config.update(data_sha256=file_sha256(args.data_dir/'train.jsonl'),
                      initialization={'adapter': parent['adapter_identity'], 'projector': parent['projector_identity'],
                                      'qformer': parent['qformer_adapter_sha256']})
        folder.mkdir(parents=True)
        (folder/'run_config.json').write_text(json.dumps(config))
        (folder/'summary.json').write_text(json.dumps({'completed': True, 'optimizer_steps': 5000}))
        for label, step in STEPS.items():
            bundle = folder/label
            (bundle/'adapter').mkdir(parents=True)
            (bundle/'adapter/adapter_config.json').write_text(arm + label)
            (bundle/'projector.pt').write_text(arm + label)
            (bundle/'qformer_adapter.pt').write_text(arm + label)
            (bundle/'metadata.json').write_text(json.dumps({'identity': config, 'optimizer_steps': step}))
    hashes = {'parent': parent}
    for arm in ARMS:
        folder = getattr(args, {'ce': 'control_dir', 'opd1': 'reference_dir', 'opd3': 'train_dir'}[arm])
        for label in STEPS:
            hashes[arm + '_' + label] = bundle_identity(folder/label)
    for split in ('action', 'scenery', 'eval'):
        data = args.data_dir/('eval.jsonl' if split == 'eval' else 'dev_' + split + '.jsonl')
        originals = [original(split + str(i)) for i in range(4)]
        dump_jsonl(data, originals)
        for key in hashes:
            # Each branch has a different best checkpoint. Official deliberately
            # prefers another checkpoint so selection leakage is detectable.
            winners = {'ce': 'checkpoint-001000', 'opd1': 'checkpoint-002500', 'opd3': 'final'}
            arm = key.split('_')[0]
            label = key.split('_', 1)[1] if '_' in key else ''
            good = 3 if label == winners.get(arm) else 1
            if split == 'eval':
                good = 4 if label == 'checkpoint-001000' else 0
            rows = [dict(question_id=r['question_id'], segment_id=r['segment_id'], question=r['question'],
                         references=r['answers'], prediction='right', lingo_judge_correct=i < good,
                         token_f1=.5, normalized_exact=0.) for i, r in enumerate(originals)]
            if arm == 'opd3':
                folder = args.eval_dir/(key + '_' + split)
            else:
                prefix = 'baseline' if arm == 'parent' else {'ce': 'sft_control', 'opd1': 'feedback_opd'}[arm] + '_' + label
                folder = args.reference_eval_dir/(prefix + '_' + split)
            dump_jsonl(folder/'predictions.jsonl', rows)
            dump_jsonl(folder/'scored/scored_predictions.jsonl', rows)
            (folder/'scored/summary.json').write_text(json.dumps(dict(summarize(rows), judge_protocol=JUDGE_PROTOCOL)))
            run = {'protocol': PROTOCOL, 'cache_dir': 'same', 'feature_stage': 'eva',
                   'precision_protocol': 'fp32_trainables_v1', 'task_instruction_version': 'legacy_v1',
                   'task_instruction': 'same', 'dtype': 'bf16', 'max_new_tokens': 96,
                   'max_records': 0, 'batch_size': 1, 'bridge_init_sha256': 'same',
                   'data_sha256': file_sha256(data), **hashes[key]}
            (folder/'run_manifest.json').write_text(json.dumps(run))
    return args


class FeedbackWeightTests(unittest.TestCase):
    def test_only_weight_change_allowed(self):
        control, reference, candidate = configs()
        validate_weight_configs(control, reference, candidate)
        validate_weight_configs(control, reference, dict(candidate, checkpoint_root='/fast'))
        for field, changed in (('lr', 1e-4), ('teacher_mode', 'reference_draft'), ('judge_model', 'new'),
                               ('rollout_distribution', 'new'), ('initialization', 'different'),
                               ('opd_weight', 1.), ('probe_report_sha256', 'changed'), ('seed', 999)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_weight_configs(control, reference, dict(candidate, **{field: changed}))
        with self.assertRaises(ValueError):
            validate_weight_configs(control, dict(reference, opd_weight=3.), candidate)

    def test_weighted_loss_retains_ce_only_examples_in_denominator(self):
        value = loss_summary({'opd_weight': 3.}, {'examples': 16, 'eligible': 4, 'ce_sum': 8., 'jsd_sum': .4})
        self.assertAlmostEqual(value['reference_ce_mean'], .5)
        self.assertAlmostEqual(value['raw_jsd_mean_eligible'], .1)
        self.assertAlmostEqual(value['weighted_jsd_mean_all_examples'], .075)

    def test_historical_paths_are_read_only_and_new_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            self.assertEqual(evaluation_path(args, 'ce_final', 'action'), args.reference_eval_dir/'sft_control_final_action')
            self.assertEqual(evaluation_path(args, 'opd1_final', 'action'), args.reference_eval_dir/'feedback_opd_final_action')
            self.assertEqual(evaluation_path(args, 'opd3_final', 'action'), args.eval_dir/'opd3_final_action')
            with self.assertRaises(ValueError):
                evaluation_path(args, 'opd3_checkpoint-004000', 'action')
            args.train_dir = args.reference_dir
            with patch('scripts.report_feedback_opd_weight.line_count', return_value=40000), self.assertRaises(ValueError):
                checked_configs(args, require_candidate=False)

    def test_equal_selection_budget_and_earlier_ties(self):
        specs = {arm+'_'+label: {'arm': arm, 'step': step} for arm in ARMS for label, step in STEPS.items()}
        metrics = {key: {'lingo_judge_accuracy': .5} for key in specs}
        metrics['opd3_final']['lingo_judge_accuracy'] = .6
        selected = select_checkpoints(metrics, specs)
        self.assertEqual(selected['ce'], 'ce_checkpoint-001000')
        self.assertEqual(selected['opd1'], 'opd1_checkpoint-001000')
        self.assertEqual(selected['opd3'], 'opd3_final')

    def test_report_compares_three_arms_and_official_cannot_reselect(self):
        with tempfile.TemporaryDirectory() as temporary, patch('scripts.report_feedback_opd_weight.line_count', return_value=40000):
            args = fixture(Path(temporary))
            before = {str(p): file_sha256(p) for p in args.reference_eval_dir.rglob('*') if p.is_file()}
            dev = build_report(args)
            self.assertEqual(dev['selected_checkpoints'], {'ce': 'ce_checkpoint-001000',
                             'opd1': 'opd1_checkpoint-002500', 'opd3': 'opd3_final'})
            self.assertEqual(set(dev['dev_combined']['same_step']), set(STEPS))
            for comparisons in dev['dev_combined']['same_step'].values():
                self.assertEqual(set(comparisons), {'ce_vs_opd1', 'ce_vs_opd3', 'opd1_vs_opd3'})
            with contextlib.redirect_stdout(io.StringIO()):
                save_report(args, dev)
            official = build_report(args, official=True)
            self.assertEqual(official['selected_checkpoints'], dev['selected_checkpoints'])
            self.assertEqual(set(official['splits']['eval']['models']), {'parent'} | set(dev['selected_checkpoints'].values()))
            self.assertEqual(before, {str(p): file_sha256(p) for p in args.reference_eval_dir.rglob('*') if p.is_file()})
            (args.train_dir/'final/projector.pt').write_text('changed')
            with self.assertRaises(ValueError):
                build_report(args, official=True)

    def test_generation_and_judge_protocol_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch('scripts.report_feedback_opd_weight.line_count', return_value=40000):
            args = fixture(Path(temporary))
            path = args.eval_dir/'opd3_final_action/run_manifest.json'
            value = json.loads(path.read_text())
            path.write_text(json.dumps(dict(value, max_new_tokens=320)))
            with self.assertRaises(ValueError):
                build_report(args)
            path.write_text(json.dumps(value))
            scored = args.eval_dir/'opd3_final_action/scored/summary.json'
            summary = json.loads(scored.read_text())
            scored.write_text(json.dumps(dict(summary, judge_protocol='changed')))
            with self.assertRaises(ValueError):
                build_report(args)

    def test_runtime_switch_is_disclosed(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            runtime = {'answer_only_logits': True, 'gradient_bucket_mb': 25., 'judge_device': 'local',
                       'judge_boundary_margin': .05, 'judge_equivalence_report_sha256': 'same', 'world_size': 4}
            (args.reference_dir/'runtime_history.jsonl').write_text(json.dumps(dict(runtime, start_step=1000))+'\n')
            (args.train_dir/'runtime_history.jsonl').write_text(json.dumps(dict(runtime, start_step=0))+'\n')
            self.assertFalse(runtime_provenance(args)['same_execution_from_step_zero_recorded'])
            (args.reference_dir/'runtime_history.jsonl').write_text(json.dumps(dict(runtime, start_step=0))+'\n')
            self.assertTrue(runtime_provenance(args)['same_execution_from_step_zero_recorded'])

    def test_diagnostics_require_same_original_tokens(self):
        case = dict(original('a'), case_id='a', cohort='cached_error', student_token_ids=[3, 2])
        row = dict(case, references=case['answers'], cached_token_ids_sha256=digest(case['student_token_ids']))
        validate_fixed_rows([case], [row])
        with self.assertRaises(ValueError):
            validate_fixed_rows([case], [dict(row, cached_token_ids_sha256=digest([4, 2]))])

    def test_weighted_ratio_is_distinguished_from_unweighted_ratio(self):
        ce = {'llm_model.x': torch.tensor([1.]), 'llm_proj.x': torch.tensor([1.]), 'visual_bridge.x': torch.tensor([1.])}
        jsd = {name: .3 * tensor for name, tensor in ce.items()}
        geometry = gradient_geometry(ce, jsd, {name: 1 for name in ce})
        row = {'gradients': geometry, 'loss_weights': {'ce': 1., 'jsd': 3.}}
        summary = gradient_summary([row])
        self.assertAlmostEqual(summary['all']['weighted_jsd_to_ce_norm_ratio_mean'], .3)
        self.assertAlmostEqual(summary['all']['unweighted_jsd_to_ce_norm_ratio_mean'], .1)
        with self.assertRaises(ValueError):
            gradient_summary([row, dict(row, loss_weights={'ce': 1., 'jsd': 1.})])

    def test_four_gpu_training_entry_forces_only_new_weight_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            stub = Path(temporary)/'python_stub'
            stub.write_text('#!%s\nimport json, os, sys\nprint(json.dumps({"argv": sys.argv[1:], '
                            '"env": {k: os.environ.get(k) for k in '
                            '["FEEDBACK_TRAIN_DIR", "FEEDBACK_OPD_WEIGHT", "FEEDBACK_CE_WEIGHT", '
                            '"FEEDBACK_FAST", "FEEDBACK_TEACHER_MODE", "CUDA_VISIBLE_DEVICES"]}}))\n' % sys.executable)
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(stub), CUDA_VISIBLE_DEVICES='0,1,2,3', FEEDBACK_OPD_WEIGHT='9')
            env.pop('FEEDBACK_RESUME_FROM', None)
            result = subprocess.run(['bash', 'scripts/run_feedback_opd_weight.sh', 'train'],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            calls = [json.loads(s) for s in result.stdout.splitlines()]
            training = calls[-1]
            self.assertEqual(training['env']['FEEDBACK_OPD_WEIGHT'], '3')
            self.assertEqual(training['env']['FEEDBACK_CE_WEIGHT'], '1')
            self.assertEqual(training['env']['FEEDBACK_FAST'], '1')
            self.assertEqual(training['env']['FEEDBACK_TEACHER_MODE'], 'reference_only')
            self.assertIn('mpp_feedback_opd_weight3_40k_2epoch_v1', training['env']['FEEDBACK_TRAIN_DIR'])
            self.assertIn('--nproc_per_node=4', training['argv'])
            for option, value in (('--epochs', '2'), ('--grad-accumulation', '4'), ('--opd-weight', '3')):
                self.assertEqual(training['argv'][training['argv'].index(option)+1], value)
            self.assertIn('--answer-only-logits', training['argv'])

    def test_fixed_diagnostic_extension_preserves_prefixes_and_does_not_rerun_old_arms(self):
        with tempfile.TemporaryDirectory() as temporary, patch('scripts.report_feedback_opd_weight.line_count', return_value=40000):
            args = fixture(Path(temporary))
            args.source_diagnostic_dir, args.diagnostic_dir = Path(temporary)/'source_diag', Path(temporary)/'new_diag'
            parent = diagnostic.weight_spec(args.initial_dir, True)
            source = {'protocol': diagnostic.PROTOCOL, 'teacher_mode': 'reference_only', 'teacher': parent,
                      'models': {'parent': parent, 'ce': diagnostic.weight_spec(args.control_dir/'final'),
                                 'opd': diagnostic.weight_spec(args.reference_dir/'final')},
                      'data_path': str((args.data_dir/'train.jsonl').resolve()), 'worker_code_hashes': {},
                      'case_cohorts': {'cached_error': 200, 'cached_correct_control': 100},
                      'gradient_steps': list(range(1, 9)), 'review_case_ids': ['c0'], 'source_hashes': {},
                      'dev_outcomes': {'historical': 1}, 'artifact_hashes': {}}
            cases = [dict(original('c%d' % i), case_id='c%d' % i,
                          cohort='cached_error' if i < 200 else 'cached_correct_control',
                          student_token_ids=[3, 2], student_answer='left') for i in range(300)]
            dump_jsonl(args.source_diagnostic_dir/'cases.jsonl', cases)
            dump_jsonl(args.source_diagnostic_dir/'gradient_draws.jsonl', [{'effective_step': i} for i in range(1, 9)])
            dump_jsonl(args.source_diagnostic_dir/'dev_changes.jsonl', [])
            (args.source_diagnostic_dir/'review_annotations.template.json').write_text('[]')
            source['artifact_hashes'] = {n: file_sha256(args.source_diagnostic_dir/n) for n in
                                        ('cases.jsonl', 'gradient_draws.jsonl', 'dev_changes.jsonl', 'review_annotations.template.json')}
            (args.source_diagnostic_dir/'manifest.json').write_text(json.dumps(source))
            (args.source_diagnostic_dir/'report.json').write_text(json.dumps({'completed': True}))
            dump_jsonl(args.train_dir/'training_log.jsonl', [])
            config = json.loads((args.train_dir/'run_config.json').read_text())
            config.update(weight_batch_layout(3))
            (args.train_dir/'run_config.json').write_text(json.dumps(config))
            for label in STEPS:
                path = args.train_dir/label/'metadata.json'
                metadata = json.loads(path.read_text())
                metadata['identity'] = config
                path.write_text(json.dumps(metadata))
            prepared = prepare(args)
            self.assertEqual(prepared['training_config']['opd_weight'], 3.)
            self.assertEqual(prepared['training_config']['world_size'], 4)
            self.assertEqual(prepared['candidate_training_config']['world_size'], 3)
            self.assertEqual(prepared['teacher'], source['teacher'])
            self.assertEqual(diagnostic.load_jsonl(args.diagnostic_dir/'cases.jsonl'), cases)
            self.assertEqual(prepared, prepare(args))
            for arm, folder, model, manifest_hash in (
                ('ce', args.source_diagnostic_dir/'ce', source['models']['ce'], file_sha256(args.source_diagnostic_dir/'manifest.json')),
                ('opd1', args.source_diagnostic_dir/'opd', source['models']['opd'], file_sha256(args.source_diagnostic_dir/'manifest.json')),
                ('opd3', args.diagnostic_dir/'opd', prepared['models']['opd'], file_sha256(args.diagnostic_dir/'manifest.json'))):
                rows = [dict(case, references=case['answers'], cached_token_ids_sha256=digest(case['student_token_ids']),
                             fixed_trajectory_jsd=.04 if arm == 'opd3' else .06, reference_ce=.5,
                             prediction='right', lingo_judge_correct=True, token_f1=.5, normalized_exact=0.) for case in cases]
                dump_jsonl(folder/'trajectories.jsonl', rows)
                dump_jsonl(folder/'scored/scored_predictions.jsonl', rows)
                (folder/'scored/summary.json').write_text(json.dumps(dict(summarize(rows), judge_protocol=JUDGE_PROTOCOL)))
                (folder/'run_config.json').write_text(json.dumps({'manifest_sha256': manifest_hash, 'student': model, 'teacher': parent}))
                (folder/'worker_summary.json').write_text(json.dumps({'completed': True, 'student_unchanged': True,
                                              'teacher_unchanged': True, 'teacher_tensor_sha256': 'same'}))
                geometry = {g: {'ce_l2': 1., 'weighted_jsd_l2': .3, 'jsd_to_ce_norm_ratio': .3, 'cosine': .2}
                            for g in ('language', 'projector', 'qformer', 'all')}
                dump_jsonl(folder/'gradients.jsonl', [{'effective_step': i, 'gradients': geometry,
                            'loss_weights': {'ce': 1., 'jsd': 3. if arm == 'opd3' else 1.}} for i in range(1, 9)])
            before = {str(p): file_sha256(p) for p in args.source_diagnostic_dir.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                make_report(args)
            report = json.loads((args.diagnostic_dir/'comparison.json').read_text())
            self.assertTrue(report['completed'])
            self.assertAlmostEqual(report['paired']['opd1_vs_opd3']['fixed_jsd_mean_delta'], -.02)
            self.assertTrue((args.diagnostic_dir/'review.html').exists())
            self.assertEqual(before, {str(p): file_sha256(p) for p in args.source_diagnostic_dir.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
