# Full Q-Former training-scale comparison

Scale changes both unique QA coverage and total updates; this does not isolate either causal contribution.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| action | joint | 3120 | 51.31% | 0.6423 |
| action | qformer100k | 3120 | 54.49% | 0.6585 |
| action | full | 3120 | 58.59% | 0.6728 |
| scenery | joint | 1200 | 60.58% | 0.5825 |
| scenery | qformer100k | 1200 | 64.33% | 0.5882 |
| scenery | full | 1200 | 63.83% | 0.5910 |
| dev_combined | joint | 4320 | 53.89% | 0.6257 |
| dev_combined | qformer100k | 4320 | 57.22% | 0.6390 |
| dev_combined | full | 4320 | 60.05% | 0.6501 |

action / joint_vs_qformer100k: fixed 296, regressed 197, net +3.17 pp; video-bootstrap 95% CI [+1.49, +4.93] pp.

action / joint_vs_full: fixed 413, regressed 186, net +7.28 pp; video-bootstrap 95% CI [+5.04, +9.41] pp.

action / qformer100k_vs_full: fixed 267, regressed 139, net +4.10 pp; video-bootstrap 95% CI [+2.65, +5.67] pp.

scenery / joint_vs_qformer100k: fixed 91, regressed 46, net +3.75 pp; video-bootstrap 95% CI [+1.83, +5.75] pp.

scenery / joint_vs_full: fixed 94, regressed 55, net +3.25 pp; video-bootstrap 95% CI [+1.25, +5.25] pp.

scenery / qformer100k_vs_full: fixed 53, regressed 59, net -0.50 pp; video-bootstrap 95% CI [-2.33, +1.25] pp.

dev_combined / joint_vs_qformer100k: fixed 387, regressed 243, net +3.33 pp; video-bootstrap 95% CI [+2.09, +4.67] pp.

dev_combined / joint_vs_full: fixed 507, regressed 241, net +6.16 pp; video-bootstrap 95% CI [+4.63, +7.79] pp.

dev_combined / qformer100k_vs_full: fixed 320, regressed 198, net +2.82 pp; video-bootstrap 95% CI [+1.65, +4.07] pp.

Selected using development data: `full`.
