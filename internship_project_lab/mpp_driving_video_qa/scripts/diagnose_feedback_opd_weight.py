#!/usr/bin/env python3
"""Replay the existing 300 cases/eight batches only for the new OPD student."""
import copy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256
from scripts import diagnose_feedback_opd as diagnostic
from scripts.generate_answers import write_json_atomic
from scripts.report_answer_opsd import paired_metrics, read_json, line_count, JUDGE_PROTOCOL
from scripts.report_preference import summarize
from scripts.report_feedback_opd_weight import parser as report_parser, checkpoint_specs, training_progress

PROTOCOL = 'feedback_opd_weight_fixed_diagnostics_v1'


def arguments():
    parser = report_parser(action_choices=('diagnose', 'diagnostic_report', 'diagnostic_status'))
    defaults = {'data_dir': 'OPD_WEIGHT_DATA_DIR', 'initial_dir': 'OPD_WEIGHT_INITIAL_DIR',
                'control_dir': 'OPD_WEIGHT_CONTROL_DIR', 'reference_dir': 'OPD_WEIGHT_REFERENCE_DIR',
                'reference_eval_dir': 'OPD_WEIGHT_REFERENCE_EVAL_DIR', 'train_dir': 'OPD_WEIGHT_TRAIN_DIR',
                'eval_dir': 'OPD_WEIGHT_EVAL_DIR'}
    parser.set_defaults(**{k: Path(os.environ[v]) for k, v in defaults.items() if os.environ.get(v)})
    parser.set_defaults(action='diagnostic_status')
    parser.add_argument('--source-diagnostic-dir', type=Path, default=Path(os.environ.get(
        'OPD_WEIGHT_SOURCE_DIAGNOSTIC_DIR', str(ROOT / 'outputs/feedback_opd_diagnostics_v1'))))
    parser.add_argument('--diagnostic-dir', type=Path, default=Path(os.environ.get(
        'OPD_WEIGHT_DIAGNOSTIC_DIR', str(ROOT / 'outputs/feedback_opd_weight_diagnostics_v1'))))
    return parser.parse_args()


def source_manifest(args):
    manifest = diagnostic.require_manifest(SimpleNamespace(output_dir=args.source_diagnostic_dir,
                                                         data=args.data_dir / 'train.jsonl'))
    report = read_json(args.source_diagnostic_dir / 'report.json') or {}
    if not report.get('completed') or manifest['case_cohorts'] != {'cached_error': 200, 'cached_correct_control': 100}:
        raise ValueError('complete the existing fixed 200-error / 100-control diagnostics first')
    models, _, configs = checkpoint_specs(args)
    for arm, name in (('ce', 'ce_final'), ('opd', 'opd1_final')):
        actual = models[name]
        expected = {'adapter': actual['adapter_identity'], 'projector': actual['projector_identity'],
                    'qformer': actual['qformer_adapter_sha256']}
        if manifest['models'][arm]['identity'] != expected:
            raise ValueError('historical diagnostic weights do not match the comparison: %s' % arm)
        diagnostic.check_weights(manifest['models'][arm])
    if manifest['teacher']['identity'] != configs['opd3']['initialization']:
        raise ValueError('new student and diagnostic must use the same original teacher')
    return manifest, configs


def prepare(args):
    source, configs = source_manifest(args)
    output = args.diagnostic_dir
    if output.resolve() == args.source_diagnostic_dir.resolve():
        raise ValueError('new diagnostics must not overwrite the completed source diagnostics')
    manifest = copy.deepcopy(source)
    manifest['extension_protocol'] = PROTOCOL
    manifest['source_diagnostic_dir'] = str(args.source_diagnostic_dir.resolve())
    manifest['source_manifest_sha256'] = file_sha256(args.source_diagnostic_dir / 'manifest.json')
    manifest['models']['opd'] = diagnostic.weight_spec(args.train_dir / 'final')
    manifest['candidate_training_config'] = configs['opd3']
    # Replay remains the original four-rank batch16, not a re-sharding of the
    # new run's online samples. Keep the historical divisor and use new weights.
    manifest['training_config'] = dict(source.get('training_config', configs['opd1']),
                                       ce_weight=configs['opd3']['ce_weight'],
                                       opd_weight=configs['opd3']['opd_weight'])
    manifest['gradient_replay_note'] = 'Original cached batch16 and prefix; candidate physical GPU layout is not the replay layout.'
    manifest['historical_dev_outcomes'] = manifest.pop('dev_outcomes')
    manifest['training_log'] = diagnostic.training_log_summary(args.train_dir)
    # Both samples and effective batches stay exactly the same as the old replay.
    # They are not this new policy's rollouts and are not fed to its optimizer.
    artifact_names = tuple(source['artifact_hashes'])
    if not (output / 'manifest.json').exists():
        output.mkdir(parents=True, exist_ok=True)
        for name in artifact_names:
            if name.endswith('.jsonl'):
                dump_jsonl(output / name, load_jsonl(args.source_diagnostic_dir / name))
            else:
                write_json_atomic(output / name, read_json(args.source_diagnostic_dir / name))
    manifest['artifact_hashes'] = {n: file_sha256(output / n) for n in artifact_names}
    for name, path in (('run_config', args.train_dir / 'run_config.json'),
                       ('metadata', args.train_dir / 'final/metadata.json')):
        manifest['source_hashes'][str(path.resolve())] = file_sha256(path)
    manifest['worker_code_hashes'][str(Path(__file__).resolve())] = file_sha256(Path(__file__))
    previous = read_json(output / 'manifest.json')
    if previous and previous != manifest:
        raise ValueError('diagnostic inputs/weights changed; use a new OPD_WEIGHT_DIAGNOSTIC_DIR')
    if not previous:
        write_json_atomic(output / 'manifest.json', manifest)
    return manifest


def inference_arguments(args):
    persistent = ROOT.parent / 'mpp_llava_video_workspace/checkpoints'
    checkpoint = Path(os.environ.get('MPP_CHECKPOINT_ROOT', '/tmp/zjr_mpp_fast/checkpoints'))
    if not (checkpoint / 'mpp-qwen-next-sft/sharded/model.safetensors.index.json').exists():
        checkpoint = persistent
    return diagnostic.parser().parse_args([
        '--action', 'infer', '--arm', 'opd', '--part', 'all',
        '--data', str(args.data_dir / 'train.jsonl'), '--output-dir', str(args.diagnostic_dir),
        '--initial-dir', str(args.initial_dir), '--train-dir', str(args.train_dir),
        '--control-dir', str(args.control_dir), '--data-dir', str(args.data_dir),
        '--teacher-mode', 'reference_only', '--seed', '44', '--max-txt-len', '1536', '--max-new-tokens', '96',
        '--cache-dir', os.environ.get('MPP_EVA_CACHE_DIR', '/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1'),
        '--upstream-root', os.environ.get('MPP_UPSTREAM_ROOT', str(ROOT.parent / 'mpp_llava_video_workspace/third_party/MPP-LLaVA')),
        '--checkpoint-root', str(checkpoint), '--visual-checkpoint-root', os.environ.get('MPP_VISUAL_CHECKPOINT_ROOT', str(persistent)),
        '--bridge-init', os.environ.get('MPP_BRIDGE_INIT', str(ROOT / 'checkpoints/visual_bridge_init.pt'))])


def validate_fixed_rows(cases, rows):
    by_id = {r['case_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r['case_id'] for r in cases}:
        raise ValueError('fixed diagnostic case set changed or incomplete')
    for case in cases:
        row = by_id[case['case_id']]
        expected = {'question_id': case['question_id'], 'question': case['question'],
                    'references': case['answers'], 'segment_id': case['segment_id'],
                    'cohort': case['cohort'],
                    'cached_token_ids_sha256': diagnostic.digest(case['student_token_ids'])}
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError('diagnostic no longer scores the same QA/student prefix')
    return by_id


def gradient_summary(rows):
    weights = {r['loss_weights']['jsd'] for r in rows}
    if len(weights) != 1:
        raise ValueError('gradient replay changed the OPD coefficient between batches')
    weight = next(iter(weights))
    return {group: {
        'batches': len(rows),
        'ce_norm_mean': statistics.mean(r['gradients'][group]['ce_l2'] for r in rows),
        'weighted_jsd_norm_mean': statistics.mean(r['gradients'][group]['weighted_jsd_l2'] for r in rows),
        'weighted_jsd_to_ce_norm_ratio_mean': diagnostic.mean(
            r['gradients'][group]['jsd_to_ce_norm_ratio'] for r in rows
            if r['gradients'][group]['jsd_to_ce_norm_ratio'] is not None),
        'unweighted_jsd_to_ce_norm_ratio_mean': diagnostic.mean(
            r['gradients'][group]['jsd_to_ce_norm_ratio'] / weight
            for r in rows if r['gradients'][group]['jsd_to_ce_norm_ratio'] is not None),
        'cosine_mean': diagnostic.mean(r['gradients'][group]['cosine'] for r in rows
                                       if r['gradients'][group]['cosine'] is not None),
        'negative_cosine_batches': sum(r['gradients'][group]['cosine'] is not None and
                                       r['gradients'][group]['cosine'] < 0 for r in rows),
    } for group in ('language', 'projector', 'qformer', 'all')}


def make_report(args):
    manifest = prepare(args)
    cases = load_jsonl(args.source_diagnostic_dir / 'cases.jsonl')
    folders = {'ce': args.source_diagnostic_dir / 'ce', 'opd1': args.source_diagnostic_dir / 'opd',
               'opd3': args.diagnostic_dir / 'opd'}
    trajectories, scored, workers, gradients = {}, {}, {}, {}
    for arm, folder in folders.items():
        config = read_json(folder / 'run_config.json') or {}
        expected_manifest = (file_sha256(args.diagnostic_dir / 'manifest.json') if arm == 'opd3'
                             else manifest['source_manifest_sha256'])
        model_name = 'ce' if arm == 'ce' else 'opd'
        expected_model = (manifest['models'][model_name] if arm != 'opd1'
                          else diagnostic.weight_spec(args.reference_dir / 'final'))
        if (config.get('manifest_sha256') != expected_manifest or config.get('student') != expected_model or
                config.get('teacher') != manifest['teacher']):
            raise ValueError('diagnostic output belongs to different model/sample/teacher: %s' % arm)
        workers[arm] = read_json(folder / 'worker_summary.json') or {}
        if (not workers[arm].get('completed') or not workers[arm].get('student_unchanged') or
                not workers[arm].get('teacher_unchanged')):
            raise ValueError('finish fixed trajectory/gradient replay: %s' % arm)
        trajectories[arm] = validate_fixed_rows(cases, load_jsonl(folder / 'trajectories.jsonl'))
        rows = load_jsonl(folder / 'scored/scored_predictions.jsonl')
        validate_fixed_rows(cases, rows)
        summary = read_json(folder / 'scored/summary.json') or {}
        if summary.get('judge_protocol') != JUDGE_PROTOCOL or any(summary.get(k) != v for k, v in summarize(rows).items()):
            raise ValueError('complete the original Judge scoring: %s' % arm)
        for row in rows:
            if row['prediction'] != trajectories[arm][row['case_id']]['prediction']:
                raise ValueError('scoring belongs to different free answers')
        scored[arm] = rows
        gradients[arm] = diagnostic.expected_rows(folder / 'gradients.jsonl', manifest['gradient_steps'], 'effective_step')
        if len(gradients[arm]) != len(manifest['gradient_steps']):
            raise ValueError('incomplete gradient batches')
    value = {'protocol': PROTOCOL, 'created_utc': datetime.now(timezone.utc).isoformat(), 'completed': True,
             'optimizer_updates': 0, 'checkpoint': 'final_step_5000', 'models': {}, 'paired': {},
             'case_cohorts': manifest['case_cohorts'], 'teacher_identity': manifest['teacher']['identity'],
             'candidate_training_config': manifest['candidate_training_config'],
             'gradient_replay_note': manifest['gradient_replay_note'],
             'source_manifest_sha256': manifest['source_manifest_sha256'],
             'interpretation': 'Same old training prefixes, same frozen reference-only teacher, and same eight old '
                'effective batches replayed at final weights; these are not instantaneous training gradients. '
                'Selected train cases are not development accuracy. Weighted gradient ratios are not AdamW update shares. '
                'No automatic fact-token labels are inferred.'}
    for arm in folders:
        value['models'][arm] = {}
        for cohort in manifest['case_cohorts']:
            chosen = [r for r in scored[arm] if r['cohort'] == cohort]
            fixed = [trajectories[arm][r['case_id']] for r in chosen]
            value['models'][arm][cohort] = {'free_generation': summarize(chosen),
                'reference_ce_mean': statistics.mean(r['reference_ce'] for r in fixed),
                'fixed_trajectory_jsd_mean': statistics.mean(r['fixed_trajectory_jsd'] for r in fixed)}
        value['models'][arm]['gradient_replay'] = gradient_summary(gradients[arm])
        value['models'][arm]['diagnostic_jsd_weight'] = gradients[arm][0]['loss_weights']['jsd']
    if len({w['teacher_tensor_sha256'] for w in workers.values()}) != 1:
        raise ValueError('actual frozen teacher tensors differ across diagnostics')
    for a in ('ce', 'opd1'):
        deltas = [trajectories['opd3'][c['case_id']]['fixed_trajectory_jsd'] -
                  trajectories[a][c['case_id']]['fixed_trajectory_jsd'] for c in cases]
        value['paired'][a + '_vs_opd3'] = {'free_generation': paired_metrics(scored[a], scored['opd3']),
            'fixed_jsd_mean_delta': statistics.mean(deltas), 'fixed_jsd_decreased_cases': sum(d < 0 for d in deltas)}
    write_json_atomic(args.diagnostic_dir / 'comparison.json', value)
    lines = ['# OPD weight fixed diagnostics', '', value['interpretation'], '',
             '| Arm | Cohort | N | Free Judge | Fixed JSD | Reference CE |', '|---|---|---:|---:|---:|---:|']
    for arm in folders:
        for cohort in manifest['case_cohorts']:
            row = value['models'][arm][cohort]
            lines.append('| %s | %s | %d | %.2f%% | %.6f | %.6f |' % (arm, cohort,
                         row['free_generation']['records'], row['free_generation']['lingo_judge_accuracy'] * 100,
                         row['fixed_trajectory_jsd_mean'], row['reference_ce_mean']))
    for key, comparison in value['paired'].items():
        score = comparison['free_generation']['lingo_judge_correct']
        lines.append('\n%s: fixed %d / regressed %d / net %+.2fpp; fixed JSD mean delta %+.6f.' %
                     (key, score['fixed'], score['regressed'], score['net_change_mean'] * 100, comparison['fixed_jsd_mean_delta']))
    (args.diagnostic_dir / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    score_by_id = {a: {r['case_id']: r for r in scored[a]} for a in folders}
    sections = ['<h1>OPD weight 1 / weight 3: fixed training cases</h1>'
                '<p>These selected training cases are diagnostics, not held-out accuracy. '
                'Judge acceptance does not establish factual correctness.</p>']
    for case in cases:
        if case['case_id'] not in manifest['review_case_ids']:
            continue
        sections.append('<section><h2>%s</h2>%s<p>%s</p><p>Reference: %s</p><p>Cached answer: %s</p>' % (
            html.escape(case['case_id']), diagnostic.images_markup(case['frames']), html.escape(case['question']),
            html.escape(' / '.join(case['answers'])), html.escape(case['student_answer'])))
        sections.append('<table><tr><th>Student</th><th>Free answer</th><th>Judge</th><th>Fixed JSD</th></tr>')
        for arm in folders:
            row = score_by_id[arm][case['case_id']]
            sections.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%.6f</td></tr>' % (
                arm, html.escape(row['prediction']), row['lingo_judge_correct'],
                trajectories[arm][case['case_id']]['fixed_trajectory_jsd']))
        sections.append('</table></section>')
    (args.diagnostic_dir / 'review.html').write_text(diagnostic.html_document(''.join(sections)), encoding='utf-8')
    print(json.dumps(value, indent=2), flush=True)


def main():
    args = arguments()
    if args.action == 'diagnostic_status':
        folder = args.diagnostic_dir / 'opd'
        print(json.dumps({'output_dir': str(args.diagnostic_dir), 'progress': training_progress(folder),
                          'trajectories': line_count(folder / 'trajectories.jsonl'), 'cases_total': 300,
                          'gradient_batches': line_count(folder / 'gradients.jsonl'), 'gradient_batches_total': 8,
                          'scored': line_count(folder / 'scored/scored_predictions.jsonl'),
                          'report_ready': (args.diagnostic_dir / 'comparison.json').exists()}, indent=2), flush=True)
        return
    if args.action == 'diagnose':
        prepare(args)
        runtime = inference_arguments(args)
        diagnostic.infer(runtime)
        diagnostic.score(runtime)
    make_report(args)


if __name__ == '__main__':
    main()
