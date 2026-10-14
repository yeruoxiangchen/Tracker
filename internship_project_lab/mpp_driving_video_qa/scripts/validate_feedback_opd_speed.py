#!/usr/bin/env python3
"""Judge equivalence, actual Qwen loss/gradient checks, and bounded OPD timings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.feedback_opd import feedback_prompt
from mpp_video_qa.opd import SharedFrozenTeacher, completion_logits, distillation_loss
from mpp_video_qa.opd_runtime import PhaseTimer, TeacherGateCache
from scripts.feedback_opd_common import (add_runtime_arguments, configure_model, training_dataset,
                                        build_student, runtime_identity, Judge, current_rollout)
from scripts.evaluate_answers import load_judge, judge_score
from scripts.generate_answers import write_json_atomic
from scripts.train_sft import seed_all, rng_state, restore_rng_state, gpu_memory


def judge_cases(path, limit):
    cases = []
    for row in load_jsonl(path):
        base = dict(question=row['question'], answers=row['references'])
        cases.append((base, row['student_answer']))
        for value in row['modes'].values():
            if value.get('teacher_correction') is not None:
                cases.append((base, value['teacher_correction']))
        if len(cases) >= limit:
            break
    return cases[:limit]


def check_judge(args, device):
    tokenizer, cpu = load_judge(args.judge_model, torch.device('cpu'))
    import copy
    gpu = copy.deepcopy(cpu).to(device).float().eval()
    differences, times, flips, near, fallback_flips = [], defaultdict(list), 0, 0, 0
    cases = judge_cases(args.judge_validation_data, args.cases)
    details = []
    for index, (row, text) in enumerate(cases):
        values = {}
        for name, model in (('cpu', cpu), ('gpu', gpu)):
            if name == 'gpu':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            score, correct = judge_score(tokenizer, model, row['question'], row['answers'], text)
            if name == 'gpu':
                torch.cuda.synchronize(device)
            times[name].append(time.perf_counter() - start)
            values[name] = (score, correct)
        delta = abs(values['cpu'][0] - values['gpu'][0])
        differences.append(delta)
        flips += values['cpu'][1] != values['gpu'][1]
        fallback = abs(values['gpu'][0]) <= args.judge_boundary_margin
        near += fallback
        decision = values['cpu'][1] if fallback else values['gpu'][1]
        fallback_flips += decision != values['cpu'][1]
        details.append(dict(index=index, cpu_logit=values['cpu'][0], gpu_logit=values['gpu'][0],
                            delta=delta, boundary_fallback=fallback))
    result = dict(passed=bool(cases) and fallback_flips == 0 and max(differences) < args.judge_boundary_margin / 4,
                  judge_model=args.judge_model, scorer_sha256=file_sha256(ROOT/'scripts/evaluate_answers.py'),
                  gpu_name=torch.cuda.get_device_name(device), boundary_margin=args.judge_boundary_margin,
                  cases=len(cases), max_abs_logit_difference=max(differences, default=0), raw_decision_flips=flips,
                  boundary_cases=near, decision_flips_with_fallback=fallback_flips,
                  cpu_median_ms=1000*statistics.median(times['cpu']),
                  gpu_median_ms=1000*statistics.median(times['gpu']),
                  validation_data_sha256=file_sha256(args.judge_validation_data), details=details)
    write_json_atomic(Path(args.output_dir)/'judge.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'details'}, indent=2), flush=True)
    if not result['passed']:
        raise RuntimeError('GPU Judge equivalence failed')


def branch(name):
    return 'language' if 'llm_model.' in name else 'projector' if 'llm_proj.' in name else 'qformer'


def check_model(args, device):
    configure_model(args, train=True)
    identity = runtime_identity(args)
    dataset = training_dataset(args)
    print(json.dumps({'status': 'loading_model', 'validation_only': True}), flush=True)
    student = build_student(args, device, train=True)
    teacher = SharedFrozenTeacher(student, args)
    named = [(n,p) for n,p in student.named_parameters() if p.requires_grad]
    originals = {n:p.detach().cpu().clone() for n,p in named}
    validations = []
    for index in range(min(args.model_cases, len(dataset))):
        item = dataset[index]
        row, features = item['row'], item['features'].unsqueeze(0).to(device)
        ids = student.backbone.llm_tokenizer(item['answer']).input_ids + [student.backbone.llm_tokenizer.im_end_id]
        ids = torch.tensor(ids, device=device)
        prompt = feedback_prompt(row, '', student.task_instruction, 'reference_only')
        with teacher.active() as frozen:
            target = completion_logits(frozen, features, row['question'], ids, prompt, answer_only=True).detach()
        results = []
        state = rng_state()
        for fast in (False, True):
            restore_rng_state(state)
            student.zero_grad(set_to_none=True)
            logits = completion_logits(student, features, row['question'], ids, answer_only=fast)
            ce = student.answer_only_ce(features, [row['question']], [item['answer']]) if fast else student(features, [row['question']], [item['answer']]).loss
            jsd = distillation_loss(logits, target, 'jsd', 8)
            ce_value, jsd_value = float(ce.detach()), float(jsd.detach())
            logits_copy = logits.detach().float().cpu()
            (ce + jsd).backward()
            results.append((ce_value, jsd_value, logits_copy,
                            {n:p.grad.detach().cpu().clone() for n,p in named if p.grad is not None}))
            del ce, jsd, logits
        norms = defaultdict(lambda: [0., 0., 0., 0., 0.])
        if results[0][3].keys() != results[1][3].keys():
            raise RuntimeError('trainable gradient set differs')
        for name, old in results[0][3].items():
            new = results[1][3][name]
            value = norms[branch(name)]
            value[0] += float(old.double().square().sum())
            value[1] += float(new.double().square().sum())
            value[2] += float((old.double() * new.double()).sum())
            value[3] += float((old.double() - new.double()).square().sum())
            value[4] = max(value[4], float((old - new).abs().max()))
        gradients = {k: {'relative_l2_difference': (v[3]/max(v[0],1e-30))**.5,
                         'cosine': v[2]/max((v[0]*v[1])**.5, 1e-30), 'max_abs_difference': v[4]}
                     for k,v in norms.items()}
        ce_delta, jsd_delta = abs(results[0][0]-results[1][0]), abs(results[0][1]-results[1][1])
        valid = ce_delta <= .005 and jsd_delta <= .001 and all(
            v['relative_l2_difference'] <= .03 and v['cosine'] >= .9995 for v in gradients.values())
        validations.append(dict(question_id=row['question_id'], passed=valid, legacy_ce=results[0][0],
                                answer_only_ce=results[1][0], legacy_jsd=results[0][1], answer_only_jsd=results[1][1],
                                logits_max_abs_difference=float((results[0][2]-results[1][2]).abs().max()), gradients=gradients))
        student.zero_grad(set_to_none=True)
        del results, target, norms
    result = {'passed': bool(validations) and all(v['passed'] for v in validations),
              'identity': identity, 'cases': validations, 'gpu_memory': gpu_memory(device)}
    write_json_atomic(Path(args.output_dir)/'model.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise RuntimeError('answer-only logits/loss/gradient validation failed')
    if args.benchmark_updates:
        benchmark(args, device, student, teacher, named, originals, dataset, identity)


def benchmark(args, device, student, teacher, named, originals, dataset, identity):
    parameters = [p for _,p in named]
    measurements = {}
    args.judge_device = 'cpu'
    args.judge_equivalence_report = str(Path(args.output_dir)/'judge.json')
    for fast in (False, True):
        with torch.no_grad():
            for name,p in named:
                p.copy_(originals[name].to(p.device))
        seed_all(args.seed)
        args.runtime_judge_device = 'local' if fast and args.benchmark_gpu_judge else 'cpu'
        judge = Judge(args)
        timer = PhaseTimer(device, 1)
        cache = TeacherGateCache(Path(args.output_dir)/'benchmark_teacher_cache', dict(identity, run='benchmark')) if fast else None
        groups = [{'params': [p for n,p in named if branch(n)==k], 'lr':lr}
                  for k,lr in (('language',2.5e-6),('projector',2.5e-6),('qformer',5e-6))]
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
        torch.cuda.reset_peak_memory_stats(device)
        seconds, profiles = [], []
        hits, eligible = 0, 0
        for update in range(args.benchmark_updates + 2):
            optimizer.zero_grad(set_to_none=True)
            timer.begin(update)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for micro in range(args.benchmark_accumulation):
                # Same records and initial RNG for both paths; warmup uses the first two windows.
                item = dataset[(update*args.benchmark_accumulation + micro) % len(dataset)]
                row, features = item['row'], item['features'].unsqueeze(0).to(device)
                ids, prompt, diagnostic = current_rollout(student, teacher, judge, features, row,
                    args.max_new_tokens, 'reference_only', cache, timer)
                hits += diagnostic['teacher_cache_hit']
                if diagnostic['eligible']:
                    eligible += 1
                    with timer.phase('teacher_probability'):
                        with teacher.active() as frozen:
                            target = completion_logits(frozen, features, row['question'], ids, prompt, answer_only=fast).detach()
                    with timer.phase('student_jsd_forward'):
                        logits = completion_logits(student, features, row['question'], ids, answer_only=fast)
                        loss = distillation_loss(logits, target, 'jsd', 8)
                    with timer.phase('student_jsd_backward'):
                        (loss/args.benchmark_accumulation).backward()
                    del target, logits, loss
                with timer.phase('ce_forward'):
                    ce = student.answer_only_ce(features, [row['question']], [item['answer']]) if fast else student(features, [row['question']], [item['answer']]).loss
                with timer.phase('ce_backward'):
                    (ce/args.benchmark_accumulation).backward()
                del ce
            with timer.phase('optimizer'):
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
            profile = timer.finish()
            seconds.append(time.perf_counter()-started)
            profiles.append(profile)
            print(json.dumps({'benchmark': 'fast' if fast else 'legacy', 'update': update+1,
                              'total': args.benchmark_updates+2, 'seconds': seconds[-1]}), flush=True)
        measurements['fast' if fast else 'legacy'] = dict(
            median_seconds=statistics.median(seconds[2:]), mean_seconds=statistics.mean(seconds[2:]),
            timings=seconds, profiles=profiles, eligible=eligible, cache_hits=hits,
            gpu_memory=gpu_memory(device), judge_device=args.runtime_judge_device)
        del optimizer, judge
        student.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    result = {'completed': True, 'validation_only': True, 'world_size':1,
              'accumulation':args.benchmark_accumulation, 'measured_updates':args.benchmark_updates,
              'measurements': measurements,
              'speedup':measurements['legacy']['median_seconds']/measurements['fast']['median_seconds'],
              'interpretation':'Single-GPU bounded timing; excludes NCCL scaling and does not guarantee four-GPU throughput. No trained weights saved.'}
    write_json_atomic(Path(args.output_dir)/'benchmark.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='measurements'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument('--action', choices=('judge','model'), required=True)
    parser.add_argument('--judge-validation-data', default=str(ROOT/'outputs/feedback_teacher_compare_v2/predictions.jsonl'))
    parser.add_argument('--judge-boundary-margin', type=float, default=.05)
    parser.add_argument('--cases', type=int, default=128)
    parser.add_argument('--model-cases', type=int, default=2)
    parser.add_argument('--benchmark-updates', type=int, default=0)
    parser.add_argument('--benchmark-accumulation', type=int, default=4)
    parser.add_argument('--benchmark-gpu-judge', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda',0)
    torch.cuda.set_device(device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    check_judge(args,device) if args.action=='judge' else check_model(args,device)


if __name__=='__main__':
    main()
