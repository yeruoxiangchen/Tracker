# MPP Driving Video QA

An independent, application-oriented video SFT project built on the published
MPP-Qwen-Next checkpoint.  The task is free-form question answering over short
driving clips.  It does not use the temporal-grounding heads, candidate pools,
or verifier adapters from the sibling experiments.

The code keeps model assets external: pass the upstream MPP-LLaVA checkout and
checkpoint directory explicitly.  Generated data, caches, adapters, and reports
remain under this project.

## Pipeline

October 14: [question-input Q-Former SFT](docs/question_qformer_sft_20261014.md).
Restore BERT text modules while inheriting the complete trained visual bridge, projector
and language QKV LoRA. The original question guides per-frame feature extraction; Qwen
still receives the original question and 32 visual tokens per frame. Reuse EVA patches and
the unchanged 40k QA / two-epoch / four-GPU / 5,000-update CE recipe. Full bridge checkpoints
include question metadata and automatically restore the correct inference path.
Twelve new CPU tests and fourteen historical regressions passed. All 44,820 QA contracts
and 28,099 EVA caches were verified. A real single-GPU two-update / accumulation-four smoke
verified text gradients, all branch updates, question-dependent visual tokens, complete
save/reload and free generation at 18.84 GiB peak allocated. Formal training and new accuracy
evaluation have not started. Run `bash scripts/run_question_qformer_sft.sh train`; use
`status`, `eval_dev`, and `eval_selected` as documented.

Repository snapshot, October 14: [retained SFT, résumé experiments and checkpoint guide](docs/resume_checkpoint_guide_20261014.md).
The retained CE model scores 60.46% on dev and 55.4% on the repeatedly reused official 500.
The résumé describes the actual full-SFT parent (60.00% / 54.2%) followed by weight-1
feedback OPD (60.23% / 54.8%); OPD has not demonstrated an advantage over matched CE.
Compact original configurations, summaries, reports and checkpoint locations are checked
in under `docs/artifacts/20261014/`; large model/data assets remain external.
`scripts/setup_upstream.sh` recreates the pinned MPP-LLaVA revision with the archived local
loading/generation patch. All historical experiment source, including untrained branches,
is preserved. The weight-3 run is also complete: dev selected step 2500 (60.39%), official
54.2%, without demonstrated gain over CE. Expanded language LoRA completed with dev
60.42% and official 54.4%. Earlier implementation-time status below is historical.

October 11: [reference-only OPD weight comparison](docs/feedback_opd_weight_20261011.md).
Add CE + 3×JSD on the same 40k QA / two-epoch / batch16 / 5,000-update recipe.
Training now supports two, three or four GPUs from `CUDA_VISIBLE_DEVICES`.
Two GPUs accumulate eight slots; three use six slots with 6+5+5 real QA and
globally normalized losses (no repeated examples or batch18).
Read the completed CE and weight-1 runs without rewriting their artifacts; compare all
three arms at steps 1,000/2,500/5,000 and select checkpoints only on development data.
The new student can also replay the same 300 train cases and eight effective batches,
reusing the old diagnostics. Run `bash scripts/run_feedback_opd_weight.sh train` and
monitor with `status`; evaluation and background commands are in the document.
Historical weight-1 execution switched to the fast version at step 1,000 and this is
reported explicitly, including replica-count/RNG differences. The first four-GPU
attempt failed while loading onto an occupied GPU, with zero updates. Retry errors
are archived; no new accuracy or full-model two/three-GPU smoke is claimed.

October 10: [fixed-trajectory feedback OPD diagnostics](docs/feedback_opd_diagnostics_20261010.md).
Compare parent SFT, matched CE and OPD with the same frozen reference-only teacher,
300 cached training answers and eight complete historical effective batches. Record
token distributions, independent free answers and CE/JSD gradient geometry without
optimizer updates. Run `bash scripts/run_feedback_opd_diagnostics.sh run` on one GPU,
or `run_parallel` with three distinct visible GPUs; monitor with `status`.
All three arms completed on October 10 at 12:32 UTC. OPD reduced fixed-prefix JSD in
186/200 cached train errors relative to CE; free answers improved from 83/200 to 89/200.
The replayed JSD gradient was about 0.10 of CE with positive cosines in all eight
effective batches. These are selected training diagnostics, not new held-out gains.

October 9 implementation: [expanded language LoRA, matched pure-CE SFT](docs/language_lora_sft_20261009.md).
Following the decision to pause the region-evidence route, retain the original five-frame
input and expand rank-16 Qwen LoRA from fused QKV to attention output and all three MLP
linear layers (160 modules). Reuse the same full-SFT parent and unchanged 40k/two-epoch,
5,000-update recipe as the historical CE control; do not add teacher or OPD supervision.
FP32 QKV inheritance, zero-initialized new increments, saved-config loading/resume and
dev-only checkpoint selection are implemented. Twenty-two targeted tests and the actual
parent/data audit passed. Formal training has not started; there is no new accuracy result.
Run `bash scripts/run_language_lora_sft.sh train`; four-GPU background, monitoring and
evaluation commands are in the document. Historical CE official accuracy remains 55.4%.

October 9 follow-up: [visual failure audit and matched-input controls](docs/evidence_case_audit_20261009.md)
completed on 24 fixed, outcome-stratified dev cases. In 12 of 16 selected regressions,
duplicated full frames already caused regression; true crops independently helped two
attribute cases and hurt four cases versus duplicates. This is not population accuracy.
Across the original dev run, 795 selected plans added only whole-frame copies (net −38).
Both evidence selection and downstream view handling need work; some Judge flips are
not semantic regressions. Keep the five-frame baseline; no new training was started.

Completed October 8; analyzed October 9: **external VLM evidence search degraded dev
Judge from 60.39% to 55.95%** on 4,320 questions (291 fixes, 483 regressions; net −192).
The segment-bootstrap interval for the change is [−6.02, −3.07] percentage points.
Of 3,378 questions with selected crops, 286 improved and 479 regressed; the 434 failed
searches fell back to the original frames. This does not establish useful automatic
evidence for the frozen MPP. The official 500 remains unrun; no new training was performed.
Retain the original-five-frame baseline pending diagnosis of region correctness and
MPP view adaptation. See [the full analysis](docs/evidence_search_results_20261009.md),
[raw report](outputs/vlm_evidence_search_v1/report_dev.md), and timeline events 54–55.

Implementation: [October 8 external VLM evidence search → frozen MPP QA](docs/evidence_search_20261008.md).
The external model reads the complete question and five chronological frames, requests
native-pixel crops, observes the returned images, and selects only previously observed
regions. MPP retains all original frames and receives selected images, not external
answers or reasoning. Default evaluation is frozen MPP full vs evidence on dev4320,
then the reused official500 after explicit dev-policy freezing. No new training or
external direct-answer benchmark is launched. Independent per-GPU search workers,
resume, compact live status, offline HTML, matched Judge reporting and cost traces
are implemented; 38 targeted tests passed. The 8B weights were downloaded with resumed
HTTP transfer; the full dev workflow subsequently completed, with 7,755 unique crop
features and both MPP arms scored. Measured search peak allocated memory was 17.36 GiB.
Use `bash scripts/run_evidence_search.sh dev`; download/background/monitoring commands
and configuration boundaries are in the document.

Completed October 8: the previous heuristic frozen-input experiment **degraded** dev
Judge: full 60.30%, duplicate 56.88%, detector+rule ROI 53.52% (238 fixes, 531 regressions
vs full). It does not validate automatic ROI gains. ROI SFT was not started; the old
report's missing training epochs do not mean its frozen comparisons are incomplete.
See [the original report](outputs/auto_region_eval_v1/report.md) and timeline event52.

Previous implementation: [October 7 automatic candidate region views and SFT](docs/auto_region_views_20261007.md).
Image-only Grounding-DINO proposals are routed using the original question, then native
crops are added alongside all five chronological full frames. Original QA values stay
unchanged. Selected EVA crops are deduplicated separately; full-frame caches are reused.
The new entrypoint supplies frozen full/duplicate/ROI comparisons, four-GPU SFT,
dev-only checkpoint selection, offline previews, and real short-clip inference.
Detection proposals are candidates, not verified target identities. Preprocessing and
frozen evaluation have now completed with the negative result above; ROI training has not started.
Start with `bash scripts/run_auto_region_sft.sh install_detector`; see the document for
the complete commands. Existing central-crop/OPD reproduction paths remain unchanged.

Prior diagnosis: [October 7 development failure validation](docs/dev_failure_validation_20261007.md).
Reviewed 80 shared failures and 30 Judge-positive controls with their original five frames.
Fixed-weight temporal controls did not show a robust improvement. On a separate 28-case
matched-input comparison, full frames plus reviewer-selected ROIs fixed five real object
errors and caused one Judge regression versus duplicated full frames. This is a selected,
manual-localization diagnostic, not an automatic system or a full-development accuracy gain.
Offline review pages embed original previews, frozen crops, predictions, scores and notes:
[failures](outputs/dev_failure_validation_v1/review_failure.html),
[controls](outputs/dev_failure_validation_v1/review_control.html).
No weights or reference labels were changed; these development cases/crops are not training data.

Completed experiment: [2026-10-06 feedback OPD v2: 40k QA, 5,000 updates](docs/feedback_opd_v2_20261006.md).
The 40k immutable original-QA pool is prepared and covers all 27,599 training clips
(25,800 Action / 14,200 Scenery). A paired comparison reuses 200 cached train errors to
compare reference-plus-draft and reference-only teacher conditions on identical student tokens.
The matched SFT and OPD branches each run two epochs / 5,000 updates, with dev evaluation
at steps 1,000, 2,500 and 5,000. Each branch chooses its own dev checkpoint before the same
official evaluation. The 200-case teacher comparison completed on October 6 at 04:20 UTC:
reference-plus-draft accepted 103/200 corrections, reference-only 190/200 (90 fixes, 3 regressions).
These are training-side, reference-conditioned diagnostics, not student accuracy gains.
The SFT control completed 5,000 updates (80,000 QA visits; 7,922.75 training seconds).
OPD completed 5,000 updates on October 6 at 22:41 UTC after stateful fast resume from step
1,000. Productive fast-run time for the remaining 4,000 updates was 10h45m. Both branches
selected their final checkpoint on 4,320 dev questions: parent / SFT / OPD scored
60.00% / 60.46% / 60.23%. OPD versus matched SFT fixed 80 and regressed 90 questions,
with a paired interval crossing zero. On the reused official 500, they scored
54.2% / 55.4% / 54.8%, also without demonstrated independent OPD gain.
See the [dev report](outputs/feedback_opd_eval_v2/report.md),
[official report](outputs/feedback_opd_eval_v2/official_report.md), and timeline events 48–50.
The [four-GPU execution optimization](docs/opd_speed_20261006.md) remains available;
runtime acceleration should not be confused with an accuracy improvement.

Previous implementation: [2026-10-05 failure-feedback conditioned OPD](docs/feedback_opd_20261005.md).
It starts from the complete 406,212-QA SFT snapshot, uses a frozen same-model teacher with
training-only references and current student drafts, and updates the full Q-Former, projector,
and language LoRA. Every example receives the same reference CE as the matched SFT control;
only finished student errors with Judge-accepted teacher corrections receive additional JSD.
The completed 2k-question / 4k-answer training probe corrected 1,364 of 1,852 student errors
(73.65%), but teacher-forced distributions sometimes strengthened wrong continuations.
That correction rate is not a held-out student score or proof of beneficial OPD supervision.
The previous 10k/625-update training entry remains available without overwriting its artifacts.

Completed: [2026-10-05 teacher prompt and native-video comparison](docs/teacher_adaptation_20261005.md).
The completed 600-question dev comparison gives MPP 58.83% and the zero-shot 4B teacher 42.33%
(82 fixes, 181 regressions). New variants compare complete driving instructions and native five-frame
video input, reusing the same probe and historical answers. B/C finished with 40.33% / 38.83%,
below the 58.83% MPP student. They are not promoted to the new token-OPD teacher.

Previous implementation: [2026-10-05 external teacher comparison and offline answer distillation](docs/teacher_distillation_20261005.md).
It starts from the full-data 54.2% model, compares a local Qwen3-VL-4B candidate teacher on 600 dev
questions, and prepares 10k training candidates for filtered teacher targets plus reference replay.
Matched original-answer SFT and distillation share data, initialization and training budget.
Start with `CUDA_VISIBLE_DEVICES=0 bash scripts/run_teacher_distillation.sh teacher_compare`.
The first teacher comparison completed and failed the global-advantage criterion; both training runs
remain unstarted. This is offline sequence distillation, not token-level OPD.

Completed: [2026-10-04 full Q-Former continuation](docs/full_qformer_continue_20261004.md).
The additional 25,389-update stage completed on October 5. Dev selected checkpoint-024000:
61.78% versus parent 60.05%; official 52.2% versus parent 54.2% (fixed 15, regressed 25).
The official paired interval crosses zero; no demonstrated official gain. The 54.2% parent remains
the reference for the new experiment. See the [continuation report](outputs/full_qformer_continue_eval_v1/official_report.md).

Completed: [2026-10-04 full-data Q-Former scaling experiment](docs/full_qformer_scale_20261004.md).
All 406,212 original training questions were used for one epoch (25,389 updates, 10h25m training).
The current best full-data model scores 60.05% on dev and 54.2% on official 500, versus
57.22%/51.0% for the previous 100k model. See the [dev report](outputs/full_qformer_scale_eval_v1/report.md)
and [official report](outputs/full_qformer_scale_eval_v1/official_report.md).

Completed: [2026-10-03 full Q-Former + projector + language LoRA SFT](docs/full_qformer_sft_20261003.md).
The new 100k original-QA subset covers all 27,599 training clips, uses frozen EVA caches,
and fine-tunes the complete existing visual bridge for two epochs. Training and evaluation
share FP32 trainable-weight loading. The selected epoch-02 scores 57.22% on dev and 51.0% on
official 500, versus same-protocol joint 53.89%/47.2%. This checkpoint is retained as a historical comparison.

Previous implementation: [2026-10-02 factual preference DPO and matched SFT](docs/preference_dpo_20261002.md).
It reuses the original five-frame Q-Former cache and joint LoRA/projector initialization,
caches fixed reference log-probabilities, and trains DPO with the same CE weights as its
SFT control. Conservative reference-based text rules selected 120 preference pairs plus
40 replay examples; these are not presented as manually verified visual labels.
Reference scoring and two-step four-GPU DPO/control smoke checks passed. Resume runs
successfully but is not bitwise identical to uninterrupted training. Formal runs and dev evaluation
are complete: joint 54.19%, matched SFT 53.73%, DPO 53.61% on 4320 questions; no demonstrated gain.

The previous [answer-conditioned OPD](docs/answer_opsd_20261001.md) comparison is complete:
on 4320 development questions, joint / continued SFT / OPD scored 54.05% / 53.66% / 53.22%.
That OPD experiment retained joint. See the
[OPD event archive](docs/answer_opsd_events_20261002.md) and
[complete chronological record](docs/project_timeline.md).

Previous visual adaptation/crop-teacher workflow:
[2026-10-01 visual SFT and teacher comparison](docs/visual_sft_opd_20261001.md).
The fixed central-crop teacher did not improve the selected 80-case diagnostic, so it is
not used by the new OPD entrypoint.

Previous continuation workflow: [2026-10-01 targeted SFT and diagnosis](docs/action_targeted_sft_20261001.md).

Previous targeted-SFT results: [2026-10-01 experiment archive](docs/action_targeted_results_20261001.md).
Targeted continuation regressed on Action dev; the full-data joint LoRA/projector model remains the selected baseline.

1. Convert LingoQA parquet/JSON annotations into a stable JSONL contract.
2. Cache the frozen MPP visual output once per segment.
3. Train a Qwen attention LoRA with assistant-only language loss.
4. Generate answers with the published MPP checkpoint and the adapted model.
5. Compare normalized exact/F1 scores and, when installed, Lingo-Judge.

See [docs/implementation.md](docs/implementation.md) for data assumptions,
commands, and evaluation protocol.

For the chronological project history—including data preparation, cache
optimization, failed launches, code fixes, and validated smoke results—start
with [docs/project_timeline.md](docs/project_timeline.md).

The expanded Action + Scenery run is driven by `scripts/run_full_v2.sh`. Its
versioned dataset is `datasets/processed/lingoqa_full_v2`; the manifest fixes the
old Action dev videos and a separate Scenery dev split before any training.
Download interruptions, range-resume handling, the full-v2 data audit, and its
first cache startup failure are recorded in
[docs/data_download_and_full_v2_bringup_20260928.md](docs/data_download_and_full_v2_bringup_20260928.md).

Full-v2 training, evaluation, and the completed 48-video action diagnostics are
documented in [docs/full_v2_action_diagnosis_20260929.md](docs/full_v2_action_diagnosis_20260929.md).
The offline visual review is at `outputs/full_v2_action_diagnosis_v1/review.html`;
it embeds all frames and needs no server. These are development diagnostics,
not additional training labels or an independent test benchmark.

For an auditable inventory of the completed full-v2 run, checkpoints, result
files, code changes, hashes, and validation status, see
[docs/experiment_code_result_archive_20260930.md](docs/experiment_code_result_archive_20260930.md).
