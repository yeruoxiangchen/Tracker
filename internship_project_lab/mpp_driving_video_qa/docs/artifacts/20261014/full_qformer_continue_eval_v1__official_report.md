# Full Q-Former continuation comparison

One additional SFT stage from the full-data parent, with a fresh optimizer and lower LR. The dataset and model architecture are retained. Bootstrap intervals are exploratory after checkpoint selection.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| eval | baseline | 500 | 54.20% | 0.3200 |
| eval | checkpoint-024000 | 500 | 52.20% | 0.3176 |

eval / baseline_vs_checkpoint-024000: fixed 15, regressed 25, net -2.00 pp; video-bootstrap 95% CI [-4.20, +0.20] pp.

Selected using development data: `checkpoint-024000`.
