# Expanded language LoRA SFT

Same full-SFT parent, original QA, five frames and 5000-update CE budget. Only language LoRA coverage expands; rank remains 16. Historical QKV predictions are reused. The repeatedly used official 500 is not a new blind test.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| eval | parent | 500 | 54.20% | 0.3196 |
| eval | qkv_final | 500 | 55.40% | 0.3169 |
| eval | decoder_final | 500 | 54.40% | 0.3197 |

parent_vs_qkv: fixed 23 / regressed 17 / net +1.20pp; video-bootstrap 95% CI [-1.20, +3.60]pp.

parent_vs_decoder: fixed 21 / regressed 20 / net +0.20pp; video-bootstrap 95% CI [-2.00, +2.60]pp.

qkv_vs_decoder: fixed 5 / regressed 10 / net -1.00pp; video-bootstrap 95% CI [-2.60, +0.40]pp.

Select each branch from steps 1000/2500/5000 by combined dev Judge; ties choose the earlier step.

Development-selected model: `qkv_final`.
