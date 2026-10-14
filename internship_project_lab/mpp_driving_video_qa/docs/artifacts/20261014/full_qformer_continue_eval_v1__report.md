# Full Q-Former continuation comparison

One additional SFT stage from the full-data parent, with a fresh optimizer and lower LR. The dataset and model architecture are retained. Bootstrap intervals are exploratory after checkpoint selection.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| action | baseline | 3120 | 58.59% | 0.6728 |
| action | checkpoint-006000 | 3120 | 57.37% | 0.6668 |
| action | checkpoint-012000 | 3120 | 58.85% | 0.6754 |
| action | checkpoint-018000 | 3120 | 60.32% | 0.6750 |
| action | checkpoint-024000 | 3120 | 60.13% | 0.6801 |
| action | final | 3120 | 60.10% | 0.6799 |
| scenery | baseline | 1200 | 63.83% | 0.5910 |
| scenery | checkpoint-006000 | 1200 | 64.00% | 0.5959 |
| scenery | checkpoint-012000 | 1200 | 65.42% | 0.5913 |
| scenery | checkpoint-018000 | 1200 | 65.50% | 0.5924 |
| scenery | checkpoint-024000 | 1200 | 66.08% | 0.5921 |
| scenery | final | 1200 | 65.33% | 0.5918 |
| dev_combined | baseline | 4320 | 60.05% | 0.6501 |
| dev_combined | checkpoint-006000 | 4320 | 59.21% | 0.6471 |
| dev_combined | checkpoint-012000 | 4320 | 60.67% | 0.6521 |
| dev_combined | checkpoint-018000 | 4320 | 61.76% | 0.6520 |
| dev_combined | checkpoint-024000 | 4320 | 61.78% | 0.6556 |
| dev_combined | final | 4320 | 61.55% | 0.6554 |

action / baseline_vs_checkpoint-006000: fixed 172, regressed 210, net -1.22 pp; video-bootstrap 95% CI [-2.56, +0.19] pp.

action / baseline_vs_checkpoint-012000: fixed 173, regressed 165, net +0.26 pp; video-bootstrap 95% CI [-1.01, +1.51] pp.

action / baseline_vs_checkpoint-018000: fixed 197, regressed 143, net +1.73 pp; video-bootstrap 95% CI [+0.26, +3.14] pp.

action / baseline_vs_checkpoint-024000: fixed 173, regressed 125, net +1.54 pp; video-bootstrap 95% CI [+0.19, +2.92] pp.

action / baseline_vs_final: fixed 174, regressed 127, net +1.51 pp; video-bootstrap 95% CI [+0.16, +2.91] pp.

scenery / baseline_vs_checkpoint-006000: fixed 48, regressed 46, net +0.17 pp; video-bootstrap 95% CI [-1.42, +1.67] pp.

scenery / baseline_vs_checkpoint-012000: fixed 54, regressed 35, net +1.58 pp; video-bootstrap 95% CI [+0.08, +3.08] pp.

scenery / baseline_vs_checkpoint-018000: fixed 56, regressed 36, net +1.67 pp; video-bootstrap 95% CI [-0.17, +3.42] pp.

scenery / baseline_vs_checkpoint-024000: fixed 57, regressed 30, net +2.25 pp; video-bootstrap 95% CI [+0.75, +3.83] pp.

scenery / baseline_vs_final: fixed 51, regressed 33, net +1.50 pp; video-bootstrap 95% CI [-0.08, +3.00] pp.

dev_combined / baseline_vs_checkpoint-006000: fixed 220, regressed 256, net -0.83 pp; video-bootstrap 95% CI [-1.85, +0.30] pp.

dev_combined / baseline_vs_checkpoint-012000: fixed 227, regressed 200, net +0.62 pp; video-bootstrap 95% CI [-0.44, +1.63] pp.

dev_combined / baseline_vs_checkpoint-018000: fixed 253, regressed 179, net +1.71 pp; video-bootstrap 95% CI [+0.48, +2.90] pp.

dev_combined / baseline_vs_checkpoint-024000: fixed 230, regressed 155, net +1.74 pp; video-bootstrap 95% CI [+0.62, +2.83] pp.

dev_combined / baseline_vs_final: fixed 225, regressed 160, net +1.50 pp; video-bootstrap 95% CI [+0.39, +2.64] pp.

Selected using development data: `checkpoint-024000`.

Dev combined Judge point estimate selects the candidate for official evaluation; ties retain baseline/earlier checkpoint. Retain baseline as main model if paired CI crosses zero. Official results are never used to choose a checkpoint.

Recommended main model after paired-interval check: `checkpoint-024000`.
