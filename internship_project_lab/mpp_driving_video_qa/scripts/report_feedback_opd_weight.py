#!/usr/bin/env python3
"""Compare matched CE, reference-only OPD weight 1 and OPD weight 3."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from mpp_video_qa.feedback_opd import validate_matched_configs
from mpp_video_qa.global_batch import normalized_weight_config, weight_batch_layout
from scripts.generate_answers import write_json_atomic
from scripts.report_answer_opsd import read_json, line_count, JUDGE_PROTOCOL
from scripts.report_feedback_opd import identities
from scripts.report_feedback_opd_v2 import bundle_identity, pair, LABELS
from scripts.report_full_qformer_scale import load_result
from scripts.report_preference import summarize

ARMS = ('ce', 'opd1', 'opd3')
PAIRS = (('ce', 'opd1'), ('ce', 'opd3'), ('opd1', 'opd3'))
STEPS = dict(zip(LABELS, (1000, 2500, 5000)))
PROTOCOL = 'feedback_opd_weight_comparison_v1'


def arm_folders(args):
    return dict(zip(ARMS, (args.control_dir, args.reference_dir, args.train_dir)))


def validate_weight_configs(control, reference, candidate=None, weight=3.):
    """Same objective/data/budget; replica layout is explicitly disclosed."""
    validate_matched_configs(control, reference)
    for config in (control, reference):
        required = {'records': 40000, 'world_size': 4, 'grad_accumulation': 4,
                    'epochs': 2, 'total_steps': 5000, 'checkpoint_steps': [1000, 2500],
                    'teacher_mode': 'reference_only', 'ce_weight': 1., 'objective': 'jsd'}
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError('requires the original 40k / four-GPU / 5000-step reference-only recipe')
    if reference.get('opd_weight') != 1.:
        raise ValueError('reference OPD must have weight 1')
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('OPD weight must be finite and positive')
    if candidate is not None:
        comparable = normalized_weight_config(candidate)
        validate_matched_configs(control, comparable)
        changed = [k for k in set(reference) | set(comparable)
                   if k not in ('opd_weight', 'checkpoint_root') and reference.get(k) != comparable.get(k)]
        if changed:
            raise ValueError('weight experiment changed other training settings: %s' % sorted(changed))
        if candidate.get('opd_weight') != weight:
            raise ValueError('candidate OPD weight does not match the requested experiment')


def checked_configs(args, require_candidate=True):
    folders = arm_folders(args)
    configs = {arm: read_json(folder / 'run_config.json') for arm, folder in folders.items()}
    if not configs['ce'] or not configs['opd1']:
        raise ValueError('completed historical CE and weight-1 runs are required')
    if require_candidate and not configs['opd3']:
        raise ValueError('weight-3 training has not started')
    validate_weight_configs(configs['ce'], configs['opd1'], configs['opd3'], args.opd_weight)
    parent = identities(args.initial_dir, True)
    expected = {'adapter': parent['adapter_identity'], 'projector': parent['projector_identity'],
                'qformer': parent['qformer_adapter_sha256']}
    train_hash = file_sha256(args.data_dir / 'train.jsonl')
    if line_count(args.data_dir / 'train.jsonl') != 40000:
        raise ValueError('the immutable training pool must contain 40,000 QA records')
    for arm, config in configs.items():
        if config is None:
            continue
        if config['data_sha256'] != train_hash or config['initialization'] != expected:
            raise ValueError('training pool or original SFT initialization changed: %s' % arm)
        if arm != 'opd3' or require_candidate:
            summary = read_json(folders[arm] / 'summary.json') or {}
            if not summary.get('completed') or summary.get('optimizer_steps') != 5000:
                raise ValueError('complete the full 5000-step budget: %s' % arm)
    if args.train_dir.resolve() in {args.control_dir.resolve(), args.reference_dir.resolve(), args.initial_dir.resolve()}:
        raise ValueError('the new training output must be separate from existing weights')
    if args.eval_dir.resolve() == args.reference_eval_dir.resolve():
        raise ValueError('use a separate comparison/evaluation output directory')
    return configs, parent


def checkpoint_specs(args):
    configs, parent = checked_configs(args)
    models, specs = {'parent': parent}, {}
    for arm, folder in arm_folders(args).items():
        for label, step in STEPS.items():
            bundle = folder / label
            metadata = read_json(bundle / 'metadata.json') or {}
            if metadata.get('identity') != configs[arm] or metadata.get('optimizer_steps') != step:
                raise ValueError('checkpoint does not belong to the matched run: %s' % bundle)
            key = arm + '_' + label
            models[key] = bundle_identity(bundle)
            specs[key] = {'arm': arm, 'label': label, 'step': step, 'bundle': str(bundle.resolve()),
                          'metadata_sha256': file_sha256(bundle / 'metadata.json')}
    return models, specs, configs


def evaluation_path(args, key, split):
    if split not in ('action', 'scenery', 'eval'):
        raise ValueError('unknown split: %s' % split)
    if key == 'parent':
        historical = args.reference_eval_dir / ('baseline_' + split)
    else:
        arm, label = key.split('_', 1)
        if arm not in ARMS or label not in LABELS:
            raise ValueError('unknown checkpoint: %s' % key)
        prefix = {'ce': 'sft_control', 'opd1': 'feedback_opd'}.get(arm)
        historical = args.reference_eval_dir / (prefix + '_' + label + '_' + split) if prefix else None
    if historical is not None and (historical / 'scored/summary.json').is_file():
        return historical
    return args.eval_dir / (key + '_' + split)


def select_checkpoints(metrics, specs):
    return {arm: max(sorted((k for k, s in specs.items() if s['arm'] == arm),
                           key=lambda k: specs[k]['step']),
                     key=lambda k: metrics[k]['lingo_judge_accuracy']) for arm in ARMS}


def load_selection(args, models, specs, configs):
    dev = read_json(args.eval_dir / 'report.json') or {}
    split_hashes = {s: file_sha256(args.data_dir / ('dev_%s.jsonl' % s)) for s in ('action', 'scenery')}
    if (dev.get('protocol') != PROTOCOL or dev.get('models') != models or
            dev.get('checkpoint_specs') != specs or dev.get('training_configs') != configs or
            dev.get('split_sha256') != split_hashes):
        raise ValueError('a matching development report is required before official evaluation')
    selected = select_checkpoints(dev['dev_combined']['models'], specs)
    if dev.get('selected_checkpoints') != selected:
        raise ValueError('development checkpoint selection changed')
    return dev


def runtime_provenance(args):
    result = {}
    for arm, folder in arm_folders(args).items():
        history_path = folder / 'runtime_history.jsonl'
        history = [json.loads(s) for s in history_path.read_text().splitlines() if s.strip()] if history_path.exists() else []
        result[arm] = {'current': read_json(folder / 'runtime_config.json'), 'history': history,
                       'history_sha256': file_sha256(history_path) if history_path.exists() else None}
    fields = ('answer_only_logits', 'gradient_bucket_mb', 'judge_device', 'judge_boundary_margin',
              'judge_equivalence_report_sha256', 'world_size')
    histories = [result[a]['history'] for a in ('opd1', 'opd3')]
    covers_start = all(any(r.get('start_step') == 0 for r in h) for h in histories)
    signatures = [{tuple(r.get(k) for k in fields) for r in h} for h in histories]
    layouts = {}
    for arm, folder in arm_folders(args).items():
        config = read_json(folder / 'run_config.json') or {}
        layouts[arm] = {k: config.get(k) for k in ('world_size', 'grad_accumulation', 'global_batch_protocol')}
        layouts[arm]['effective_global_batch_size'] = config.get('global_batch_size') or (
            config.get('world_size', 0) * config.get('grad_accumulation', 0))
    return {'arms': result, 'batch_layouts': layouts, 'same_execution_from_step_zero_recorded':
            covers_start and signatures[0] == signatures[1] and len(signatures[0]) == 1,
            'note': 'Historical weight-1 OPD switched to the fast implementation at step 1000. '
                    'Matched objective/data/budget does not establish identical execution or random trajectories. '
                    'Changing replica count changes rank-specific RNG streams, sampled answers and possibly gates, '
                    'even with the same shuffled global QA batches and effective batch16. '
                    'Use a same-version weight-1 rerun to confirm a small apparent weight-3 advantage.'}


def parent_outcomes(rows, parent, a, b):
    by_parent = {r['question_id']: r for r in parent}
    result = {}
    for name, correct in (('parent_correct', True), ('parent_wrong', False)):
        chosen = {qid for qid, r in by_parent.items() if bool(r['lingo_judge_correct']) == correct}
        subset = {k: [r for r in rows[k] if r['question_id'] in chosen] for k in (a, b)}
        result[name] = pair(subset, a, b) if chosen else {'records': 0}
    return result


def loss_summary(config, totals):
    examples, eligible = totals.get('examples', 0), totals.get('eligible', 0)
    return {'examples': examples, 'eligible_examples': eligible,
            'reference_ce_mean': totals.get('ce_sum', 0) / examples if examples else None,
            'raw_jsd_mean_eligible': totals.get('jsd_sum', 0) / eligible if eligible else None,
            'weighted_jsd_mean_all_examples': config['opd_weight'] * totals.get('jsd_sum', 0) / examples if examples else None,
            'note': 'OPD also averages over CE-only examples; changing trajectories make these losses training diagnostics, not accuracy.'}


def build_report(args, official=False):
    models, specs, configs = checkpoint_specs(args)
    dev = load_selection(args, models, specs, configs) if official else None
    selected = dev['selected_checkpoints'] if official else None
    keys = ['parent'] + [selected[a] for a in ARMS] if official else list(models)
    splits = ('eval',) if official else ('action', 'scenery')
    pooled = {key: [] for key in keys}
    result = {'protocol': PROTOCOL, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'judge_protocol': JUDGE_PROTOCOL, 'models': models, 'checkpoint_specs': specs,
              'training_configs': configs, 'candidate_opd_weight': args.opd_weight,
              'selection_split': 'dev', 'split_sha256': {}, 'splits': {}, 'evaluation_paths': {},
              'runtime_provenance': runtime_provenance(args),
              'interpretation': 'Original five frames and question, no reference/draft in student evaluation. '
                 '40k QA, two epochs, 5000 updates; CE vs OPD weight 1 vs weight 3. '
                 'Historical scores are read without rewriting them. The reused official 500 is not a new blind test.'}
    result['training_loss_summary'] = {arm: loss_summary(configs[arm],
        (read_json(folder / 'summary.json') or {}).get('cumulative_metrics', {}))
        for arm, folder in arm_folders(args).items()}
    all_protocols, split_rows = set(), {}
    for split in splits:
        data = args.data_dir / ('eval.jsonl' if split == 'eval' else 'dev_%s.jsonl' % split)
        result['split_sha256'][split] = file_sha256(data)
        rows = {}
        result['evaluation_paths'][split] = {}
        for key in keys:
            folder = evaluation_path(args, key, split)
            rows[key], generation_protocol = load_result(folder, data, models[key])
            run = read_json(folder / 'run_manifest.json')
            if run.get('max_new_tokens') != 96 or run.get('batch_size') != 1 or run.get('max_records') != 0:
                raise ValueError('evaluation requires the full split, batch 1 and max_new_tokens 96')
            all_protocols.add(generation_protocol)
            pooled[key].extend(rows[key])
            result['evaluation_paths'][split][key] = str(folder.resolve())
        split_rows[split] = rows
        result['splits'][split] = {'models': {k: summarize(v) for k, v in rows.items()}}
    if len(all_protocols) != 1:
        raise ValueError('generation protocols differ between models or splits')
    metrics = {k: summarize(v) for k, v in pooled.items()}
    if not official:
        selected = select_checkpoints(metrics, specs)
        same_step = {}
        for label in LABELS:
            same_step[label] = {a + '_vs_' + b: pair(pooled, a + '_' + label, b + '_' + label)
                                for a, b in PAIRS}
        result['dev_combined'] = {'models': metrics, 'same_step': same_step}
    result['selected_checkpoints'] = selected
    chosen = {'parent': pooled['parent'], **{a: pooled[selected[a]] for a in ARMS}}
    comparisons = {a + '_vs_' + b: pair(chosen, a, b)
                   for a, b in (('parent', 'ce'), ('parent', 'opd1'), ('parent', 'opd3')) + PAIRS}
    result['selected_comparison'] = {'models': {k: summarize(v) for k, v in chosen.items()},
                                    'paired': comparisons,
                                    'by_parent_correctness': {a + '_vs_' + b: parent_outcomes(chosen, chosen['parent'], a, b)
                                                             for a, b in (('ce', 'opd3'), ('opd1', 'opd3'))}}
    for split, rows in split_rows.items():
        selected_rows = {a: rows[selected[a]] for a in ARMS}
        result['splits'][split]['selected_paired'] = {a + '_vs_' + b: pair(selected_rows, a, b) for a, b in PAIRS}
    result['candidate_gain_supported_vs_ce'] = (
        summarize(chosen['opd3'])['lingo_judge_accuracy'] > summarize(chosen['parent'])['lingo_judge_accuracy'] and
        comparisons['ce_vs_opd3']['net_change_95pct_interval'][0] > 0)
    result['candidate_gain_supported_vs_opd1'] = comparisons['opd1_vs_opd3']['net_change_95pct_interval'][0] > 0
    result['selection_note'] = 'Each arm selects its own best combined-dev Judge among steps 1000/2500/5000; ties keep earlier steps. Intervals after dev selection are exploratory.'
    if official:
        result['dev_report_sha256'] = file_sha256(args.eval_dir / 'report.json')
    return result


def save_report(args, value, official=False):
    name = 'official_report' if official else 'report'
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.eval_dir / (name + '.json'), value)
    lines = ['# OPD weight comparison', '', value['interpretation'], '',
             '| Split | Model | N | Judge | Token F1 |', '|---|---|---:|---:|---:|']
    blocks = list(value['splits'].items())
    if 'dev_combined' in value:
        blocks.append(('dev_combined', value['dev_combined']))
    for split, block in blocks:
        for key, metric in block['models'].items():
            lines.append('| %s | %s | %d | %.2f%% | %.4f |' %
                         (split, key, metric['records'], 100 * metric['lingo_judge_accuracy'], metric['token_f1']))
    pairs = list(value['selected_comparison']['paired'].items())
    for label, block in value.get('dev_combined', {}).get('same_step', {}).items():
        pairs.extend((label + '/' + k, v) for k, v in block.items())
    for label, comparison in pairs:
        score, ci = comparison['lingo_judge_correct'], comparison['net_change_95pct_interval']
        lines.append('\n%s: fixed %d / regressed %d / net %+.2fpp; segment-bootstrap 95%% CI [%+.2f, %+.2f]pp.' %
                     (label, score['fixed'], score['regressed'], 100 * score['net_change_mean'], 100 * ci[0], 100 * ci[1]))
    lines += ['', value['selection_note'], '', json.dumps(value['selected_checkpoints'], ensure_ascii=False),
              '', value['runtime_provenance']['note'], '',
              'Same execution recorded from step zero: %s.' % value['runtime_provenance']['same_execution_from_step_zero_recorded']]
    (args.eval_dir / (name + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'json': str(args.eval_dir / (name + '.json')), 'markdown': str(args.eval_dir / (name + '.md')),
                      'selected_checkpoints': value['selected_checkpoints'],
                      'selected_metrics': value['selected_comparison']['models'],
                      'candidate_gain_supported_vs_ce': value['candidate_gain_supported_vs_ce']}, indent=2), flush=True)


def training_progress(folder):
    summary = read_json(folder / 'summary.json')
    if summary and summary.get('completed'):
        return dict(summary, status='completed')
    failures = sorted(folder.glob('failure_rank_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    value = read_json(failures[0]) if failures else read_json(folder / 'progress.json') or {'status': 'not_started'}
    value = dict(value)
    if value.get('updated_unix_seconds'):
        value['seconds_since_update'] = round(time.time() - value['updated_unix_seconds'], 1)
    if value.get('rank0_pid'):
        try:
            os.kill(value['rank0_pid'], 0)
            value['rank0_process_alive'] = True
        except ProcessLookupError:
            value['rank0_process_alive'] = False
        except PermissionError:
            value['rank0_process_alive'] = None
    return value


def status(args):
    data_paths = {s: args.data_dir / ('eval.jsonl' if s == 'eval' else 'dev_%s.jsonl' % s)
                  for s in ('action', 'scenery', 'eval')}
    totals = {s: line_count(p) for s, p in data_paths.items()}
    result = {'train_dir': str(args.train_dir), 'candidate_opd_weight': args.opd_weight,
              'optimizer_steps_total': 5000, 'training': training_progress(args.train_dir), 'evaluation': {},
              'dev_report_ready': (args.eval_dir / 'report.json').exists(),
              'official_report_ready': (args.eval_dir / 'official_report.json').exists()}
    config = read_json(args.train_dir / 'run_config.json')
    if config:
        counters = result['training'].get('cumulative_metrics', {})
        result['loss_summary'] = loss_summary(config, counters)
        result['batch_layout'] = {k: config.get(k) for k in ('world_size', 'grad_accumulation', 'global_batch_protocol')}
        result['batch_layout']['effective_global_batch_size'] = config.get('global_batch_size') or (
            config.get('world_size', 0) * config.get('grad_accumulation', 0))
    for label in LABELS:
        for split in data_paths:
            folder = evaluation_path(args, 'opd3_' + label, split)
            result['evaluation'][label + '_' + split] = {'generated': line_count(folder / 'predictions.jsonl'),
                'total': totals[split], 'scored': (folder / 'scored/summary.json').exists(),
                'score_progress': read_json(folder / 'scored/score_progress.json')}
    print(json.dumps(result, indent=2), flush=True)


def parser(action_choices=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (('data-dir', 'datasets/processed/lingoqa_feedback_opd_40k_v2'),
                          ('initial-dir', 'outputs/mpp_lingoqa_qformer_all_1epoch_v1'),
                          ('control-dir', 'outputs/mpp_feedback_sft_control_40k_2epoch_v2'),
                          ('reference-dir', 'outputs/mpp_feedback_opd_40k_2epoch_v2'),
                          ('reference-eval-dir', 'outputs/feedback_opd_eval_v2'),
                          ('train-dir', 'outputs/mpp_feedback_opd_weight3_40k_2epoch_v1'),
                          ('eval-dir', 'outputs/feedback_opd_weight_eval_v1')):
        p.add_argument('--' + name, type=Path, default=ROOT / default)
    p.add_argument('--opd-weight', type=float, default=3.)
    p.add_argument('--action', choices=action_choices or ('check', 'status', 'eval_path', 'selected', 'report', 'report_eval'), default='status')
    p.add_argument('--model', default='opd3')
    p.add_argument('--split', choices=('action', 'scenery', 'eval'), default='action')
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--launch-world-size', type=int, choices=(2, 3, 4))
    p.add_argument('--launch-grad-accumulation', type=int)
    p.add_argument('--launch-global-batch-size', type=int)
    return p


def main():
    args = parser().parse_args()
    if args.action == 'status':
        status(args)
    elif args.action == 'eval_path':
        print(evaluation_path(args, args.model, args.split))
    elif args.action == 'selected':
        models, specs, configs = checkpoint_specs(args)
        dev = load_selection(args, models, specs, configs)
        if args.model not in ARMS:
            raise ValueError('select an arm: ce, opd1 or opd3')
        print(specs[dev['selected_checkpoints'][args.model]]['label'])
    elif args.action == 'check':
        configs, _ = checked_configs(args, require_candidate=False)
        saved = configs['opd3']
        world = args.launch_world_size or (saved or {}).get('world_size', 4)
        layout = weight_batch_layout(world)
        requested = dict(layout)
        if world == 4:
            requested['global_batch_size'] = 0
            requested.pop('global_batch_protocol')
        if (args.launch_grad_accumulation is not None and
                args.launch_grad_accumulation != layout['grad_accumulation']):
            raise ValueError('launch accumulation does not preserve real batch16')
        if (args.launch_global_batch_size is not None and
                args.launch_global_batch_size != requested['global_batch_size']):
            raise ValueError('launch global batch protocol differs')
        if saved and any(saved.get(k, 0) != v for k, v in requested.items()):
            raise ValueError('cannot change GPU layout in a started run; use a new output directory')
        if args.resume_from:
            config = (read_json(args.resume_from / 'metadata.json') or {}).get('identity', {})
            validate_weight_configs(configs['ce'], configs['opd1'], config, args.opd_weight)
            if configs['opd3'] is None:
                raise ValueError('resume requires the original candidate output directory')
        print(json.dumps({'check_passed': True, 'records': 40000, 'epochs': 2, 'optimizer_steps_total': 5000,
                          **requested, 'effective_global_batch_size': 16,
                          'ce_weight': 1., 'opd_weight': args.opd_weight,
                          'initial_dir': str(args.initial_dir), 'train_dir': str(args.train_dir),
                          'new_training_started': configs['opd3'] is not None}, indent=2), flush=True)
    else:
        official = args.action == 'report_eval'
        save_report(args, build_report(args, official), official)


if __name__ == '__main__':
    main()
