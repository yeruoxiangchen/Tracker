# Feedback OPD v2: matched 40k / 5000 updates

Five full frames, no reference/draft at student evaluation. Matched update budgets, not matched GPU time. Development-selected intervals are exploratory; the reused official 500 questions are not a new blind test.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| action | baseline | 3120 | 58.37% | 0.6726 |
| action | sft_control_checkpoint-001000 | 3120 | 57.15% | 0.6692 |
| action | sft_control_checkpoint-002500 | 3120 | 58.27% | 0.6689 |
| action | sft_control_final | 3120 | 58.75% | 0.6768 |
| action | feedback_opd_checkpoint-001000 | 3120 | 57.63% | 0.6670 |
| action | feedback_opd_checkpoint-002500 | 3120 | 58.30% | 0.6672 |
| action | feedback_opd_final | 3120 | 58.33% | 0.6726 |
| scenery | baseline | 1200 | 64.25% | 0.5913 |
| scenery | sft_control_checkpoint-001000 | 1200 | 64.75% | 0.5923 |
| scenery | sft_control_checkpoint-002500 | 1200 | 65.08% | 0.5871 |
| scenery | sft_control_final | 1200 | 64.92% | 0.5906 |
| scenery | feedback_opd_checkpoint-001000 | 1200 | 64.25% | 0.5941 |
| scenery | feedback_opd_checkpoint-002500 | 1200 | 64.50% | 0.5888 |
| scenery | feedback_opd_final | 1200 | 65.17% | 0.5925 |
| dev_combined | baseline | 4320 | 60.00% | 0.6500 |
| dev_combined | sft_control_checkpoint-001000 | 4320 | 59.26% | 0.6479 |
| dev_combined | sft_control_checkpoint-002500 | 4320 | 60.16% | 0.6461 |
| dev_combined | sft_control_final | 4320 | 60.46% | 0.6528 |
| dev_combined | feedback_opd_checkpoint-001000 | 4320 | 59.47% | 0.6468 |
| dev_combined | feedback_opd_checkpoint-002500 | 4320 | 60.02% | 0.6454 |
| dev_combined | feedback_opd_final | 4320 | 60.23% | 0.6503 |

baseline_vs_sft_control: fixed 195 / regressed 175 / net +0.46pp; video-bootstrap 95% CI [-0.48, +1.41]pp.

baseline_vs_feedback_opd: fixed 194 / regressed 184 / net +0.23pp; video-bootstrap 95% CI [-0.72, +1.17]pp.

sft_control_vs_feedback_opd: fixed 80 / regressed 90 / net -0.23pp; video-bootstrap 95% CI [-0.82, +0.38]pp.

same_step_checkpoint-001000: fixed 105 / regressed 96 / net +0.21pp; video-bootstrap 95% CI [-0.48, +0.89]pp.

same_step_checkpoint-002500: fixed 105 / regressed 111 / net -0.14pp; video-bootstrap 95% CI [-0.83, +0.56]pp.

same_step_final: fixed 80 / regressed 90 / net -0.23pp; video-bootstrap 95% CI [-0.82, +0.38]pp.

Each branch selects from the same steps 1000/2500/5000 by combined dev Judge; ties choose the earlier step.

{"sft_control": "sft_control_final", "feedback_opd": "feedback_opd_final"}
