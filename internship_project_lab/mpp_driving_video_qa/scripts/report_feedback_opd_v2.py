#!/usr/bin/env python3
"""Fixed-step comparisons and dev-only checkpoint selection for both branches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from scripts.generate_answers import directory_fingerprint, write_json_atomic
from scripts.report_answer_opsd import read_json, line_count, paired_metrics, JUDGE_PROTOCOL
from scripts.report_feedback_opd import identities, validate_training, status as training_status
from scripts.report_full_qformer_scale import load_result
from scripts.report_preference import summarize, cluster_interval

BRANCHES = ('sft_control', 'feedback_opd')
LABELS = ('checkpoint-001000', 'checkpoint-002500', 'final')


def model_key(branch, label):
    return branch + '_' + label


def bundle_identity(folder):
    return {'adapter_identity': directory_fingerprint(folder / 'adapter'),
            'projector_identity': file_sha256(folder / 'projector.pt'),
            'qformer_adapter_sha256': file_sha256(folder / 'qformer_adapter.pt')}


def checkpoint_specs(args):
    configs = validate_training(args)
    if any(c['records'] != 40000 or c['world_size'] != 4 or c['grad_accumulation'] != 4 or
           c['epochs'] != 2 or c['total_steps'] != 5000 for c in configs):
        raise ValueError('v2 requires the agreed 40k QA, four GPUs, two epochs, 5000-step budget')
    if any(c.get('checkpoint_steps') != [1000, 2500] for c in configs):
        raise ValueError('use the fixed checkpoint opportunities in the v2 command')
    models = {'baseline': identities(args.initial_dir, True)}
    specs = {}
    for branch, folder, config in zip(BRANCHES, (args.control_dir, args.train_dir), configs):
        for label in LABELS:
            bundle = folder / label
            metadata = read_json(bundle / 'metadata.json') or {}
            step = 5000 if label == 'final' else int(label.split('-')[1])
            if metadata.get('identity') != config or metadata.get('optimizer_steps') != step:
                raise ValueError('checkpoint does not belong to this matched run: %s' % bundle)
            key = model_key(branch, label)
            models[key] = bundle_identity(bundle)
            specs[key] = {'branch': branch, 'label': label, 'step': step,
                          'bundle': str(bundle.resolve()), 'metadata_sha256': file_sha256(bundle / 'metadata.json')}
    return models, specs


def select_checkpoints(metrics, specs):
    """One equal dev selection opportunity per branch; ties retain the earlier step."""
    return {branch: max(sorted((key for key, value in specs.items() if value['branch'] == branch),
                              key=lambda key: specs[key]['step']),
                        key=lambda key: metrics[key]['lingo_judge_accuracy']) for branch in BRANCHES}


def pair(rows, a, b):
    return dict(paired_metrics(rows[a], rows[b]), **cluster_interval(rows[a], rows[b]))


def build_report(args, official=False):
    hashes, specs = checkpoint_specs(args)
    dev = read_json(args.eval_dir / 'report.json') if official else None
    if official:
        if (not dev or dev['models'] != hashes or dev['checkpoint_specs'] != specs or
                dev['split_sha256'] != {s: file_sha256(args.data_dir / ('dev_%s.jsonl' % s))
                                       for s in ('action', 'scenery')}):
            raise ValueError('matching development selection is required before official evaluation')
        # Recheck selection against the stored dev metrics, never official scores.
        if dev['selected_checkpoints'] != select_checkpoints(dev['dev_combined']['models'], specs):
            raise ValueError('development checkpoint selection changed')
        selected = dev['selected_checkpoints']
        keys = ['baseline'] + [selected[b] for b in BRANCHES]
        splits = ('eval',)
    else:
        keys, splits = list(hashes), ('action', 'scenery')
    pooled = {key: [] for key in keys}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'judge_protocol': JUDGE_PROTOCOL,
              'models': hashes, 'checkpoint_specs': specs, 'selection_split': 'dev', 'splits': {},
              'split_sha256': {}, 'interpretation': 'Five full frames, no reference/draft at student evaluation. '
                  'Matched update budgets, not matched GPU time. Development-selected intervals are exploratory; '
                  'the reused official 500 questions are not a new blind test.'}
    result['execution_runtime'] = {branch: read_json(folder / 'runtime_config.json')
                                   for branch, folder in zip(BRANCHES, (args.control_dir, args.train_dir))}
    result['execution_history_sha256'] = {branch: file_sha256(folder / 'runtime_history.jsonl')
                                         if (folder / 'runtime_history.jsonl').exists() else None
                                         for branch, folder in zip(BRANCHES, (args.control_dir, args.train_dir))}
    for split in splits:
        data = args.data_dir / ('eval.jsonl' if split == 'eval' else 'dev_%s.jsonl' % split)
        result['split_sha256'][split] = file_sha256(data)
        rows, protocols = {}, []
        for key in keys:
            rows[key], protocol = load_result(args.eval_dir / (key + '_' + split), data, hashes[key])
            pooled[key].extend(rows[key])
            protocols.append(protocol)
        if len(set(protocols)) != 1:
            raise ValueError('generation protocols differ')
        result['splits'][split] = {'models': {k: summarize(v) for k, v in rows.items()}}
    metrics = {k: summarize(v) for k, v in pooled.items()}
    if not official:
        selected = select_checkpoints(metrics, specs)
        same_step = {}
        for label in LABELS:
            a, b = (model_key(branch, label) for branch in BRANCHES)
            same_step[label] = pair(pooled, a, b)
        result['dev_combined'] = {'models': metrics, 'same_step_control_vs_opd': same_step,
            'baseline_comparisons': {k: pair(pooled, 'baseline', k) for k in keys if k != 'baseline'}}
    result['selected_checkpoints'] = selected
    chosen_rows = {'baseline': pooled['baseline'], **{b: pooled[selected[b]] for b in BRANCHES}}
    chosen_pairs = {a + '_vs_' + b: pair(chosen_rows, a, b) for a, b in
                    (('baseline', 'sft_control'), ('baseline', 'feedback_opd'), BRANCHES)}
    result['selected_comparison'] = {'models': {k: summarize(v) for k, v in chosen_rows.items()},
                                     'paired': chosen_pairs}
    result['opd_independent_gain_supported'] = (
        result['selected_comparison']['models']['feedback_opd']['lingo_judge_accuracy'] >
        result['selected_comparison']['models']['baseline']['lingo_judge_accuracy'] and
        chosen_pairs['sft_control_vs_feedback_opd']['net_change_95pct_interval'][0] > 0)
    result['selection_note'] = 'Each branch selects from the same steps 1000/2500/5000 by combined dev Judge; ties choose the earlier step.'
    if official:
        result['dev_report_sha256'] = file_sha256(args.eval_dir / 'report.json')
    return result


def save_report(args, result, official=False):
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    name = 'official_report' if official else 'report'
    write_json_atomic(args.eval_dir / (name + '.json'), result)
    lines = ['# Feedback OPD v2: matched 40k / 5000 updates', '', result['interpretation'], '',
             '| Split | Model | N | Judge | Token F1 |', '|---|---|---:|---:|---:|']
    blocks = list(result['splits'].items())
    if 'dev_combined' in result:
        blocks.append(('dev_combined', result['dev_combined']))
    for split, block in blocks:
        for key, value in block['models'].items():
            lines.append('| %s | %s | %d | %.2f%% | %.4f |' %
                         (split, key, value['records'], 100 * value['lingo_judge_accuracy'], value['token_f1']))
    pairs = list(result['selected_comparison']['paired'].items())
    if 'dev_combined' in result:
        pairs += [('same_step_' + k, v) for k, v in result['dev_combined']['same_step_control_vs_opd'].items()]
    for key, value in pairs:
        score, ci = value['lingo_judge_correct'], value['net_change_95pct_interval']
        lines.append('\n%s: fixed %d / regressed %d / net %+.2fpp; video-bootstrap 95%% CI [%+.2f, %+.2f]pp.' %
                     (key, score['fixed'], score['regressed'], 100 * score['net_change_mean'], 100 * ci[0], 100 * ci[1]))
    lines += ['', result['selection_note'], '', json.dumps(result['selected_checkpoints'], ensure_ascii=False)]
    (args.eval_dir / (name + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def selected_label(args, branch):
    hashes, specs = checkpoint_specs(args)
    report = read_json(args.eval_dir / 'report.json') or {}
    if report.get('models') != hashes or report.get('checkpoint_specs') != specs:
        raise ValueError('matching development report is required')
    selected = select_checkpoints(report['dev_combined']['models'], specs)
    if report['selected_checkpoints'] != selected:
        raise ValueError('selection changed')
    return specs[selected[branch]]['label']


def main():
    parser = argparse.ArgumentParser()
    for name, default in (('data-dir', 'datasets/processed/lingoqa_feedback_opd_40k_v2'),
                          ('initial-dir', 'outputs/mpp_lingoqa_qformer_all_1epoch_v1'),
                          ('probe-dir', 'outputs/feedback_teacher_compare_v2'),
                          ('train-dir', 'outputs/mpp_feedback_opd_40k_2epoch_v2'),
                          ('control-dir', 'outputs/mpp_feedback_sft_control_40k_2epoch_v2'),
                          ('eval-dir', 'outputs/feedback_opd_eval_v2')):
        parser.add_argument('--' + name, type=Path, default=ROOT / default)
    parser.add_argument('--action', choices=('status', 'report', 'report_eval', 'selected'), default='status')
    parser.add_argument('--branch', choices=BRANCHES, default='feedback_opd')
    args = parser.parse_args()
    if args.action == 'status':
        # Reuse PID-aware training status, add all fixed checkpoint evaluation jobs.
        import contextlib
        import io
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            training_status(args)
        result = json.loads(captured.getvalue())
        result['evaluation'] = {}
        keys = ['baseline'] + [model_key(b, label) for b in BRANCHES for label in LABELS]
        for key in keys:
            for split in ('action', 'scenery', 'eval'):
                folder = args.eval_dir / (key + '_' + split)
                data = args.data_dir / ('eval.jsonl' if split == 'eval' else 'dev_%s.jsonl' % split)
                result['evaluation'][key + '_' + split] = {
                    'generated': line_count(folder / 'predictions.jsonl'), 'total': line_count(data),
                    'scored': (folder / 'scored/summary.json').exists(),
                    'score_progress': read_json(folder / 'scored/score_progress.json')}
        print(json.dumps(result, indent=2))
    elif args.action == 'selected':
        print(selected_label(args, args.branch))
    else:
        official = args.action == 'report_eval'
        save_report(args, build_report(args, official), official)


if __name__ == '__main__':
    main()
