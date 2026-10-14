# Full Q-Former training-scale comparison

Scale changes both unique QA coverage and total updates; this does not isolate either causal contribution.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| eval | joint | 500 | 47.20% | 0.3134 |
| eval | qformer100k | 500 | 51.00% | 0.3235 |
| eval | full | 500 | 54.20% | 0.3200 |

eval / joint_vs_qformer100k: fixed 45, regressed 26, net +3.80 pp; video-bootstrap 95% CI [+0.40, +7.20] pp.

eval / joint_vs_full: fixed 55, regressed 20, net +7.00 pp; video-bootstrap 95% CI [+3.40, +10.80] pp.

eval / qformer100k_vs_full: fixed 36, regressed 20, net +3.20 pp; video-bootstrap 95% CI [+0.40, +6.00] pp.

Selected using development data: `full`.
