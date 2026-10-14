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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"))
from mpp_video_qa.core import build_user_prompt, dump_jsonl, file_sha256, PROTOCOL
from mpp_video_qa.data import cache_file, CachedQADataset
from mpp_video_qa.feedback_opd import feedback_prompt, gate_reason, token_diagnostics, MATCHED_KEYS, validate_matched_configs
from mpp_video_qa.model import CachedVideoQA, save_projector, assistant_only_preprocess
from mpp_video_qa.opd import SharedFrozenTeacher, completion_logits, distillation_loss, disable_dropout
from mpp_video_qa.visual_bridge import VisualBridge
from scripts.feedback_opd_common import current_rollout
from scripts.prepare_feedback_opd import prepare
from scripts.probe_feedback_opd import report as probe_report
from scripts.report_feedback_opd import build_report, save_report, identities
from scripts.report_preference import summarize
from scripts.report_answer_opsd import JUDGE_PROTOCOL


def row(qid="train", source="action"):
    return {"question_id": qid, "segment_id": qid, "source_subset": source,
            "question": "Which way does the vehicle turn?", "answers": ["The vehicle turns left."],
            "frames": ["/%s/frame-%d.jpg" % (qid, i) for i in range(5)], "feedback_training_source": "train"}


class Tokenizer:
    im_start_id, im_end_id, eod_id, pad_token_id = 1, 2, 0, 0

    def __call__(self, text):
        import re
        pieces = re.split('(<\\|extra_0\\|>)', text)
        ids = [v for part in pieces for v in ([7] if part == '<|extra_0|>' else [8 + ord(c) % 110 for c in part])]
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return ' '.join('t%d' % int(i) for i in ids)


class FeedbackOPDTests(unittest.TestCase):
    def test_teacher_reference_isolated_and_gate_states(self):
        original = row()
        before = copy.deepcopy(original)
        prompt = feedback_prompt(original, "The vehicle turns right.")
        self.assertIn(original["answers"][0], prompt)
        self.assertIn("The vehicle turns right.", prompt)
        self.assertEqual(prompt.count('<ImageHere>'), 5)
        self.assertNotIn(original["answers"][0], build_user_prompt(original["question"], 5))
        self.assertEqual(original, before)
        with self.assertRaises(ValueError):
            feedback_prompt(dict(original, feedback_training_source="dev"), "wrong")
        self.assertEqual(gate_reason(False, False, True, True), "student_truncated")
        self.assertEqual(gate_reason(True, True), "student_correct_ce_only")
        self.assertEqual(gate_reason(True, False, True, False), "teacher_wrong_ce_only")
        self.assertEqual(gate_reason(True, False, True, True), "eligible")

    def test_runtime_does_not_call_teacher_for_correct_student(self):
        model = types.SimpleNamespace(task_instruction="instruction", backbone=types.SimpleNamespace(llm_tokenizer=Tokenizer()))
        judge = types.SimpleNamespace(score=lambda r, text: {"correct": True, "logit": 2.0})
        teacher = types.SimpleNamespace(active=lambda: self.fail("correct student should be CE-only"))
        with patch('scripts.feedback_opd_common.sample_completions', return_value=(torch.tensor([8, 2]), True)):
            ids, prompt, result = current_rollout(model, teacher, judge, torch.zeros(1), row(), 10)
        self.assertIsNone(prompt)
        self.assertFalse(result['eligible'])

    def test_dynamic_gate_refreshes_current_answers(self):
        import contextlib
        model = types.SimpleNamespace(task_instruction="instruction", backbone=types.SimpleNamespace(llm_tokenizer=Tokenizer()))
        teacher = types.SimpleNamespace(active=lambda: contextlib.nullcontext(model))
        judge = types.SimpleNamespace(score=lambda r, text: {"correct": 't9' in text, "logit": 2 if 't9' in text else -2})
        with patch('scripts.feedback_opd_common.sample_completions', side_effect=[
                (torch.tensor([8, 2]), True), (torch.tensor([9, 2]), True), (torch.tensor([9, 2]), True)]):
            first = current_rollout(model, teacher, judge, torch.zeros(1), row(), 10)[2]
            second = current_rollout(model, teacher, judge, torch.zeros(1), row(), 10)[2]
        self.assertTrue(first['eligible'])
        self.assertFalse(second['eligible'])
        self.assertEqual(second['gate_reason'], 'student_correct_ce_only')

    def test_token_diagnostics_original_sequence_and_not_visual_verification(self):
        s = torch.tensor([[2., 0., 1.], [1., 2., 0.]])
        t = torch.tensor([[0., 3., 1.], [1., 2., 0.]])
        tokens = token_diagnostics(s, s, t, torch.tensor([0, 1]), Tokenizer())
        self.assertEqual([t['token_id'] for t in tokens], [0, 1])
        self.assertLess(tokens[0]['feedback_minus_plain'], 0)
        result = probe_report([], {}, 2)
        self.assertFalse(result['visual_supervision_direction_verified'])
        self.assertFalse(result['completed'])

    def test_prepare_keeps_qa_and_rejects_held_out_frame_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cache = root / 'source', root / 'cache'
            train = [row('tr%d' % i, 'action' if i < 4 else 'scenery') for i in range(6)]
            dev, ev = [row('dev')], [row('eval')]
            for name, rows in (('train', train), ('dev', dev), ('eval', ev), ('dev_action', dev), ('dev_scenery', dev)):
                dump_jsonl(source / (name + '.jsonl'), rows)
            for r in train:
                target = cache_file(cache, r['segment_id'])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            args = types.SimpleNamespace(source_dir=source, output_dir=root/'out', train_size=6,
                                         probe_size=3, seed=44, max_qa_per_segment=2, cache_dir=cache)
            with patch('builtins.print'):
                prepare(args)
                prepare(args)
            selected = [json.loads(line) for line in (root/'out/train.jsonl').read_text().splitlines()]
            originals = {r['question_id']: r for r in train}
            self.assertTrue(all(r['answers'] == originals[r['question_id']]['answers'] and
                                r['question'] == originals[r['question_id']]['question'] for r in selected))
            dev[0]['frames'][0] = train[0]['frames'][0]
            dump_jsonl(source/'dev.jsonl', dev)
            with self.assertRaises(ValueError):
                prepare(args)

    def test_cache_checks_full_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = row()
            dump_jsonl(root/'data.jsonl', [original])
            target = cache_file(root/'cache', original['segment_id'])
            target.parent.mkdir(parents=True)
            payload = {'protocol': PROTOCOL, 'segment_id': original['segment_id'], 'feature_stage': 'eva',
                       'frame_paths': original['frames'], 'view': 'full', 'features': torch.zeros(5, 2, 1408)}
            torch.save(payload, target)
            dataset = CachedQADataset(root/'data.jsonl', root/'cache', 'eva', validate_full_frames=True)
            self.assertEqual(dataset[0]['features'].shape[0], 5)
            torch.save(dict(payload, view='full_crop'), target)
            with self.assertRaises(RuntimeError):
                dataset[0]

    def test_real_full_bridge_teacher_freeze_alignment_and_student_gradients(self):
        from transformers import BertConfig, GPT2Config, GPT2LMHeadModel
        from peft import LoraConfig, get_peft_model
        from lavis.models.minigpt4qwen_models.Qformer import BertLMHeadModel
        torch.manual_seed(17)
        config = BertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=64, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        config.encoder_width, config.add_cross_attention, config.cross_attention_freq, config.query_length = 16, True, 2, 32
        qformer = BertLMHeadModel(config)
        qformer.bert.embeddings.word_embeddings = qformer.bert.embeddings.position_embeddings = qformer.cls = None
        for layer in qformer.bert.encoder.layer:
            layer.output = layer.intermediate = None
        state = {'Qformer.'+k: v for k, v in qformer.state_dict().items()}
        state['query_tokens'] = torch.randn(1, 32, 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.save_pretrained(root/'ckpt/bert-base-uncased')
            torch.save({'format': 'mpp_visual_bridge_init_v1', 'vision_width':16, 'state_dict':state}, root/'init.pt')
            bridge = VisualBridge(root/'init.pt', root/'ckpt/bert-base-uncased', mode='full', trainable=True)
            bridge.save(root/'full.pt')
            llm = get_peft_model(GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16,
                                                         n_positions=2048, vocab_size=128)),
                                LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'], task_type='CAUSAL_LM'))
            for name, parameter in llm.named_parameters():
                if 'lora_B' in name:
                    parameter.data.fill_(.100001)
            llm.save_pretrained(root/'adapter')
            backbone = nn.Module()
            backbone.llm_model, backbone.llm_proj, backbone.visual_bridge = llm, nn.Linear(32, 16), bridge
            backbone.llm_tokenizer, backbone.max_txt_len = Tokenizer(), 1800
            backbone.replace_image_string, backbone.replace_image_token_id = '<|extra_0|>', 7
            backbone.preprocess = types.MethodType(assistant_only_preprocess, backbone)
            save_projector(backbone, root/'projector.pt')
            student = CachedVideoQA(backbone, feature_stage='eva')
            args = types.SimpleNamespace(teacher_adapter=str(root/'adapter'), teacher_projector=str(root/'projector.pt'),
                                         teacher_qformer_adapter=str(root/'full.pt'), bridge_init=str(root/'init.pt'),
                                         checkpoint_root=str(root), visual_checkpoint_root='', dtype='bf16',
                                         precision_protocol='fp32_trainables_v1')
            teacher = SharedFrozenTeacher(student, args)
            self.assertTrue(teacher.validate_initial_copy())
            disable_dropout(student)
            features = torch.randn(1, 5, 9, 16)
            ids = torch.tensor([10, 11, 2])
            student.eval()
            with torch.no_grad(), teacher.active() as frozen:
                plain = completion_logits(frozen, features, 'q', ids)
                targets = completion_logits(frozen, features, 'q', ids, feedback_prompt(row(), 'wrong'))
            torch.testing.assert_close(completion_logits(student, features, 'q', ids), plain, atol=0, rtol=0)
            self.assertEqual(teacher.model.backbone.visual_bridge.query_tokens.dtype, torch.float32)
            frozen_before = {n: p.detach().clone() for n, p in teacher.model.named_parameters()
                             if not n.startswith('backbone.llm_model.') or 'opd_teacher' in n}
            student.train()
            opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
            loss = distillation_loss(completion_logits(student, features, 'q', ids), targets)
            loss.backward()
            groups = {'language': '.lora_B.default.', 'projector': 'llm_proj.', 'qformer': 'visual_bridge.'}
            self.assertTrue(all(any(p.grad is not None and p.grad.abs().sum() > 0
                                   for n, p in student.named_parameters() if marker in n) for marker in groups.values()))
            self.assertTrue(all(p.grad is None for n, p in llm.named_parameters() if 'opd_teacher' in n))
            opt.step()
            with self.assertRaises(RuntimeError):
                teacher.validate_initial_copy()
            for n, p in teacher.model.named_parameters():
                if n in frozen_before:
                    torch.testing.assert_close(p, frozen_before[n], atol=0, rtol=0)

    def test_matched_control_rejects_different_ce_or_budget(self):
        control = dict.fromkeys(MATCHED_KEYS, 1)
        control['training_mode'] = 'sft_control'
        opd = dict(control, training_mode='feedback_opd')
        validate_matched_configs(control, opd)
        with self.assertRaises(ValueError):
            validate_matched_configs(control, dict(opd, ce_weight=.2))
        with self.assertRaises(ValueError):
            validate_matched_configs(control, dict(opd, total_steps=47))

    def test_eval_selection_ignores_official_and_checks_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = types.SimpleNamespace(data_dir=root/'data', initial_dir=root/'base',
                                         train_dir=root/'opd', control_dir=root/'control', eval_dir=root/'eval')
            for folder, original in ((args.initial_dir, True), (args.train_dir, False), (args.control_dir, False)):
                adapter = folder/('final_adapter' if original else 'final/adapter')
                adapter.mkdir(parents=True)
                (adapter/'config.json').write_text(str(folder))
                for name in ('final_projector.pt', 'final_qformer_adapter.pt') if original else ('final/projector.pt', 'final/qformer_adapter.pt'):
                    (folder/name).write_text(str(folder))
            dump_jsonl(args.data_dir/'train.jsonl', [row()])
            initial = identities(args.initial_dir, True)
            initialization = {'adapter':initial['adapter_identity'], 'projector':initial['projector_identity'],
                              'qformer':initial['qformer_adapter_sha256']}
            common = dict.fromkeys(MATCHED_KEYS, 1)
            common.update(initialization=initialization, data_sha256=file_sha256(args.data_dir/'train.jsonl'))
            for mode, folder in (('sft_control', args.control_dir), ('feedback_opd', args.train_dir)):
                (folder/'run_config.json').write_text(json.dumps(dict(common, training_mode=mode)))
                (folder/'summary.json').write_text(json.dumps({'completed': True, 'optimizer_steps':1}))
            for split in ('action', 'scenery', 'eval'):
                originals = [row(split+'1'), row(split+'2')]
                data = args.data_dir/('eval.jsonl' if split=='eval' else 'dev_'+split+'.jsonl')
                dump_jsonl(data, originals)
                for model, folder, original in (('baseline', args.initial_dir, True), ('sft_control', args.control_dir, False),
                                                ('feedback_opd', args.train_dir, False)):
                    destination = args.eval_dir/(model+'_'+split)
                    destination.mkdir(parents=True)
                    values = [True, model=='feedback_opd' and split=='action']
                    predictions = [dict(question_id=r['question_id'], segment_id=r['segment_id'], question=r['question'],
                                        references=r['answers'], prediction='a', lingo_judge_correct=value,
                                        normalized_exact=0., token_f1=.5) for r, value in zip(originals, values)]
                    dump_jsonl(destination/'predictions.jsonl', predictions)
                    dump_jsonl(destination/'scored/scored_predictions.jsonl', predictions)
                    (destination/'scored/summary.json').write_text(json.dumps(dict(summarize(predictions), judge_protocol=JUDGE_PROTOCOL)))
                    protocol = {'protocol':PROTOCOL, 'cache_dir':'same', 'feature_stage':'eva', 'task_instruction_version':'legacy_v1',
                                'task_instruction':'same', 'precision_protocol':'fp32_trainables_v1', 'dtype':'bf16',
                                'max_new_tokens':96, 'max_records':0, 'batch_size':1, 'bridge_init_sha256':'same',
                                'data_sha256':file_sha256(data)}
                    (destination/'run_manifest.json').write_text(json.dumps(dict(protocol, **identities(folder, original))))
            dev = build_report(args)
            self.assertEqual(dev['selected_model'], 'feedback_opd')
            with patch('builtins.print'):
                save_report(args, dev)
            official = build_report(args, official=True)
            self.assertEqual(official['selected_model'], 'feedback_opd')
            self.assertEqual(set(official['splits']['eval']['models']), {'baseline', 'feedback_opd'})
            (args.train_dir/'final/projector.pt').write_text('changed')
            with self.assertRaises(ValueError):
                build_report(args, official=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
