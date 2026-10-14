#!/usr/bin/env python3
"""Paired teacher conditions on cached TRAIN student answers; no optimization."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256
from mpp_video_qa.feedback_opd import (TEACHER_MODES, feedback_prompt, decode_completion,
                                     gate_from_scores, token_diagnostics)
from mpp_video_qa.opd import SharedFrozenTeacher, sample_completions, completion_logits
from scripts.feedback_opd_common import (add_runtime_arguments, configure_model, runtime_identity,
                                         training_dataset, build_student, Judge)
from scripts.generate_answers import write_json_atomic
from scripts.probe_feedback_opd import write_review


def select_cases(old_rows, originals, limit, seed):
    """Source-proportional, one wrong draw per question; original metadata intact."""
    rng = random.Random(seed)
    shuffled = list(old_rows)
    rng.shuffle(shuffled)
    by_source, used = {}, set()
    for row in shuffled:
        qid = row['question_id']
        if qid not in originals:
            continue
        original = originals[qid]
        fields = (('segment_id', 'segment_id'), ('question', 'question'),
                  ('references', 'answers'), ('frames', 'frames'), ('source_subset', 'source_subset'))
        if any(row[key] != original[source] for key, source in fields):
            raise ValueError('cached student answer differs from original training QA: %s' % qid)
        if (qid in used or not row['student_finished'] or row['student_judge']['correct'] or
                not math.isfinite(row['student_judge']['logit']) or not row['student_token_ids']):
            continue
        used.add(qid)
        by_source.setdefault(original['source_subset'], []).append(row)
    proportions = Counter(r['source_subset'] for r in originals.values())
    total = sum(proportions.values())
    quota = {s: int(limit * n / total) for s, n in proportions.items()}
    for s in sorted(quota, key=lambda s: (-(limit * proportions[s] / total - quota[s]), s))[:limit - sum(quota.values())]:
        quota[s] += 1
    selected = []
    for source, count in sorted(quota.items()):
        available = by_source.get(source, [])
        if len(available) < count:
            raise ValueError('not enough cached training errors for %s: %d/%d' % (source, len(available), count))
        # Mix old eligible and uncorrected cases, rather than selecting favorable cases only.
        groups = {}
        for row in available:
            groups.setdefault(row['gate_reason'], []).append(row)
        ordered = []
        for index in range(max(map(len, groups.values()))):
            for key in sorted(groups):
                if index < len(groups[key]):
                    ordered.append(groups[key][index])
        selected.extend(ordered[:count])
    rng.shuffle(selected)
    return selected


def comparison_report(rows, identities, expected, source_hash):
    result = {'protocol': 'feedback_teacher_comparison_v2', 'completed': len(rows) == expected,
              'draws': len(rows), 'total_draws': expected, 'identity_by_mode': identities,
              'source_probe_sha256': source_hash, 'modes': {},
              'visual_supervision_direction_verified': False,
              'interpretation': 'Paired cached training errors, identical student IDs in both conditions. '
                  'Fresh teacher correctness does not certify useful teacher-forced supervision. '
                  'Token probability movement is not an automatic factual error label. '
                  'This source-balanced diagnostic sample is not an unbiased error-rate estimate.'}
    for mode in TEACHER_MODES:
        values = [r['modes'][mode] for r in rows]
        detailed = [v for v in values if v.get('tokens')]
        eligible = [v for v in values if v['eligible']]
        result['modes'][mode] = {
            'gate_counts': dict(Counter(v['gate_reason'] for v in values)),
            'eligible_draws': len(eligible), 'eligible_rate': len(eligible) / max(1, len(values)),
            'token_diagnostic_draws': len(detailed),
            'eligible_mean_sampled_logp_increase_draws': sum(
                sum(t['feedback_minus_plain'] for t in v['tokens']) > 0
                for v in eligible if v.get('tokens')),
            'correction_repeats_wrong_draft': sum(
                str(v.get('teacher_correction', '')).strip().lower() == r['student_answer'].strip().lower()
                for r, v in zip(rows, values)),
            'token_jsd_mean': sum(t['jsd'] for v in detailed for t in v['tokens']) /
                max(1, sum(len(v['tokens']) for v in detailed))}
    result['paired_correction'] = {
        'reference_only_fixed': sum(not r['modes']['reference_draft']['eligible'] and
                                    r['modes']['reference_only']['eligible'] for r in rows),
        'reference_only_regressed': sum(r['modes']['reference_draft']['eligible'] and
                                        not r['modes']['reference_only']['eligible'] for r in rows)}
    return result


def main(args):
    configure_model(args, train=False)
    output, source = Path(args.output_dir), Path(args.source_probe_dir)
    output.mkdir(parents=True, exist_ok=True)
    old_config = json.loads((source / 'run_config.json').read_text())
    old_report = json.loads((source / 'probe_report.json').read_text())
    if not old_report.get('completed'):
        raise ValueError('the source training probe must be complete')
    identities = {}
    for mode in TEACHER_MODES:
        args.teacher_mode = mode
        identities[mode] = runtime_identity(args)
    # Reuse only the same original policy/frames/runtime, never a changed student snapshot.
    excluded = {'data_sha256', 'feedback_protocol', 'teacher_mode'}
    changes = [k for k, v in identities['reference_draft'].items()
               if k not in excluded and old_config.get(k) != v]
    if changes:
        raise ValueError('cached student probe runtime differs: %s' % changes)
    source_rows = load_jsonl(source / 'predictions.jsonl')
    if len(source_rows) != old_report['draws']:
        raise ValueError('source probe predictions are incomplete')
    dataset = training_dataset(args)
    originals = {r['question_id']: r for r in dataset.rows}
    selected = select_cases(source_rows, originals, args.cases, args.seed)
    source_hash = file_sha256(source / 'predictions.jsonl')
    selection = [(r['question_id'], r['sample_id'], r['student_token_ids']) for r in selected]
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    config = {'protocol': 'feedback_teacher_comparison_v2', 'identity_by_mode': identities,
              'source_probe_sha256': source_hash, 'selected_draws_sha256': selection_hash,
              'cases': args.cases, 'seed': args.seed}
    config_path = output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('comparison inputs changed; use a new output directory')
    write_json_atomic(config_path, config)
    if args.validate_only:
        plan = dict(config, validation_only=True, optimizer_updates=0,
                    sources=dict(Counter(r['source_subset'] for r in selected)),
                    questions=len({r['question_id'] for r in selected}),
                    old_gate_counts=dict(Counter(r['gate_reason'] for r in selected)))
        write_json_atomic(output / 'selection_validation.json', plan)
        print(json.dumps(plan, indent=2))
        return
    prediction_path = output / 'predictions.jsonl'
    rows = load_jsonl(prediction_path) if prediction_path.exists() else []
    expected = {(r['question_id'], r['sample_id']): r for r in selected}
    done = {(r['question_id'], r['sample_id']) for r in rows}
    if len(done) != len(rows) or not done <= set(expected):
        raise ValueError('existing comparison rows are duplicated or outside the selected sample')
    for row in rows:
        original = expected[(row['question_id'], row['sample_id'])]
        for key in ('segment_id', 'question', 'references', 'frames', 'student_answer', 'student_token_ids'):
            if row[key] != original[key]:
                raise ValueError('resumed comparison changes the cached student sequence')
    if len(rows) < len(selected):
        torch.set_num_threads(args.cpu_threads)
        device = torch.device('cuda', 0)
        def progress(stage, started=None, initial_count=0):
            value = {'status': stage, 'completed': len(rows), 'total': len(selected),
                     'rank0_pid': os.getpid(), 'updated_unix_seconds': time.time()}
            if started is not None:
                elapsed = time.time() - started
                value.update(elapsed_seconds=elapsed, estimated_remaining_hours=
                             (len(selected) - len(rows)) * elapsed / max(1, len(rows) - initial_count) / 3600)
            write_json_atomic(output / 'progress.json', value)
            print(json.dumps(value), flush=True)
        progress('loading_model')
        student = build_student(args, device, train=False)
        teacher = SharedFrozenTeacher(student, args)
        teacher.validate_initial_copy()
        judge = Judge(args)
        indices = {r['question_id']: i for i, r in enumerate(dataset.rows)}
        started, initial_count = time.time(), len(rows)
        with prediction_path.open('a', encoding='utf-8') as stream, torch.no_grad():
            for old in selected:
                if (old['question_id'], old['sample_id']) in done:
                    continue
                original = originals[old['question_id']]
                features = dataset[indices[old['question_id']]]['features'].unsqueeze(0).to(device)
                ids = torch.tensor(old['student_token_ids'], device=device, dtype=torch.long)
                if decode_completion(student, ids) != old['student_answer']:
                    raise ValueError('tokenizer/cached student text mismatch')
                student_logits = completion_logits(student, features, original['question'], ids)
                with teacher.active() as frozen:
                    plain_logits = completion_logits(frozen, features, original['question'], ids)
                row = {k: old[k] for k in ('question_id', 'segment_id', 'source_subset', 'sample_id',
                       'question', 'references', 'frames', 'student_answer', 'student_token_ids',
                       'student_finished', 'student_judge')}
                row['modes'] = {}
                for mode in TEACHER_MODES:
                    prompt = feedback_prompt(original, old['student_answer'], student.task_instruction, mode)
                    value = {'teacher_mode': mode, 'teacher_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                             'teacher_correction': None, 'teacher_judge': None, 'teacher_finished': False,
                             'eligible': False, 'gate_reason': 'teacher_context_too_long'}
                    try:
                        with teacher.active() as frozen:
                            correction, finished = sample_completions(frozen, features, original['question'],
                                args.max_new_tokens, prompt=prompt, do_sample=False)
                            text = decode_completion(frozen, correction)
                            feedback_logits = completion_logits(frozen, features, original['question'], ids, prompt)
                        score = judge.score(original, text)
                        reason = gate_from_scores(True, old['student_judge']['logit'], finished, score['logit'])
                        value.update(teacher_correction=text, teacher_judge=score, teacher_finished=finished,
                                     gate_reason=reason, eligible=reason == 'eligible',
                                     tokens=token_diagnostics(student_logits, plain_logits, feedback_logits,
                                                             ids, student.backbone.llm_tokenizer))
                        del feedback_logits
                    except ValueError as error:
                        if 'exceeds max_txt_len' not in str(error):
                            raise
                        value['context_error'] = str(error)
                    row['modes'][mode] = value
                del student_logits, plain_logits
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                rows.append(row)
                progress('comparing_teacher', started, initial_count)
    result = comparison_report(rows, identities, len(selected), source_hash)
    write_json_atomic(output / 'probe_report.json', result)
    for mode in TEACHER_MODES:
        write_review(output / ('review_%s.html' % mode),
                     [dict(r, **r['modes'][mode]) for r in rows], args.review_cases)
    (output / 'probe_report.md').write_text('# Paired teacher conditions\n\n' + result['interpretation'] +
        '\n\n```json\n' + json.dumps(result, indent=2, ensure_ascii=False) + '\n```\n', encoding='utf-8')
    write_json_atomic(output / 'progress.json', {'status': 'completed', 'completed': len(rows),
                      'total': len(selected), 'updated_unix_seconds': time.time()})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument('--source-probe-dir', default=str(ROOT / 'outputs/feedback_opd_probe_v1'))
    parser.add_argument('--cases', type=int, default=200)
    parser.add_argument('--review-cases', type=int, default=20)
    parser.add_argument('--validate-only', action='store_true', help='check reuse identities and sample selection without loading a GPU model')
    cli = parser.parse_args()
    if cli.cases < 1:
        parser.error('--cases must be positive')
    try:
        main(cli)
    except Exception:
        destination = Path(cli.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json_atomic(destination / 'progress.json', {'status': 'failed', 'rank0_pid': os.getpid(),
                          'updated_unix_seconds': time.time(), 'traceback': traceback.format_exc()})
        raise
