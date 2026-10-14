# Expanded language LoRA SFT

Same full-SFT parent, original QA, five frames and 5000-update CE budget. Only language LoRA coverage expands; rank remains 16. Historical QKV predictions are reused. The repeatedly used official 500 is not a new blind test.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| action | parent | 3120 | 58.37% | 0.6726 |
| action | qkv_checkpoint-001000 | 3120 | 57.15% | 0.6692 |
| action | qkv_checkpoint-002500 | 3120 | 58.27% | 0.6689 |
| action | qkv_final | 3120 | 58.75% | 0.6768 |
| action | decoder_checkpoint-001000 | 3120 | 57.15% | 0.6688 |
| action | decoder_checkpoint-002500 | 3120 | 58.65% | 0.6703 |
| action | decoder_final | 3120 | 58.40% | 0.6738 |
| scenery | parent | 1200 | 64.25% | 0.5913 |
| scenery | qkv_checkpoint-001000 | 1200 | 64.75% | 0.5923 |
| scenery | qkv_checkpoint-002500 | 1200 | 65.08% | 0.5871 |
| scenery | qkv_final | 1200 | 64.92% | 0.5906 |
| scenery | decoder_checkpoint-001000 | 1200 | 65.17% | 0.5940 |
| scenery | decoder_checkpoint-002500 | 1200 | 64.00% | 0.5870 |
| scenery | decoder_final | 1200 | 65.67% | 0.5906 |
| dev_combined | parent | 4320 | 60.00% | 0.6500 |
| dev_combined | qkv_checkpoint-001000 | 4320 | 59.26% | 0.6479 |
| dev_combined | qkv_checkpoint-002500 | 4320 | 60.16% | 0.6461 |
| dev_combined | qkv_final | 4320 | 60.46% | 0.6528 |
| dev_combined | decoder_checkpoint-001000 | 4320 | 59.38% | 0.6480 |
| dev_combined | decoder_checkpoint-002500 | 4320 | 60.14% | 0.6472 |
| dev_combined | decoder_final | 4320 | 60.42% | 0.6507 |

parent_vs_qkv: fixed 195 / regressed 175 / net +0.46pp; video-bootstrap 95% CI [-0.48, +1.41]pp.

parent_vs_decoder: fixed 201 / regressed 183 / net +0.42pp; video-bootstrap 95% CI [-0.55, +1.42]pp.

qkv_vs_decoder: fixed 67 / regressed 69 / net -0.05pp; video-bootstrap 95% CI [-0.63, +0.56]pp.

same_step_checkpoint-001000: fixed 70 / regressed 65 / net +0.12pp; video-bootstrap 95% CI [-0.42, +0.68]pp.

same_step_checkpoint-002500: fixed 81 / regressed 82 / net -0.02pp; video-bootstrap 95% CI [-0.63, +0.62]pp.

same_step_final: fixed 67 / regressed 69 / net -0.05pp; video-bootstrap 95% CI [-0.63, +0.56]pp.

Select each branch from steps 1000/2500/5000 by combined dev Judge; ties choose the earlier step.

Development-selected model: `qkv_final`.
