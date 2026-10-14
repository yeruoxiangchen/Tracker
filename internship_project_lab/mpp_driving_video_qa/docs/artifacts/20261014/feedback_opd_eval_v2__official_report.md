# Feedback OPD v2: matched 40k / 5000 updates

Five full frames, no reference/draft at student evaluation. Matched update budgets, not matched GPU time. Development-selected intervals are exploratory; the reused official 500 questions are not a new blind test.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| eval | baseline | 500 | 54.20% | 0.3196 |
| eval | sft_control_final | 500 | 55.40% | 0.3169 |
| eval | feedback_opd_final | 500 | 54.80% | 0.3180 |

baseline_vs_sft_control: fixed 23 / regressed 17 / net +1.20pp; video-bootstrap 95% CI [-1.20, +3.60]pp.

baseline_vs_feedback_opd: fixed 23 / regressed 20 / net +0.60pp; video-bootstrap 95% CI [-1.80, +3.20]pp.

sft_control_vs_feedback_opd: fixed 10 / regressed 13 / net -0.60pp; video-bootstrap 95% CI [-2.40, +1.40]pp.

Each branch selects from the same steps 1000/2500/5000 by combined dev Judge; ties choose the earlier step.

{"sft_control": "sft_control_final", "feedback_opd": "feedback_opd_final"}
