# OPD weight comparison

Original five frames and question, no reference/draft in student evaluation. 40k QA, two epochs, 5000 updates; CE vs OPD weight 1 vs weight 3. Historical scores are read without rewriting them. The reused official 500 is not a new blind test.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| action | parent | 3120 | 58.37% | 0.6726 |
| action | ce_checkpoint-001000 | 3120 | 57.15% | 0.6692 |
| action | ce_checkpoint-002500 | 3120 | 58.27% | 0.6689 |
| action | ce_final | 3120 | 58.75% | 0.6768 |
| action | opd1_checkpoint-001000 | 3120 | 57.63% | 0.6670 |
| action | opd1_checkpoint-002500 | 3120 | 58.30% | 0.6672 |
| action | opd1_final | 3120 | 58.33% | 0.6726 |
| action | opd3_checkpoint-001000 | 3120 | 57.18% | 0.6623 |
| action | opd3_checkpoint-002500 | 3120 | 58.88% | 0.6602 |
| action | opd3_final | 3120 | 58.21% | 0.6663 |
| scenery | parent | 1200 | 64.25% | 0.5913 |
| scenery | ce_checkpoint-001000 | 1200 | 64.75% | 0.5923 |
| scenery | ce_checkpoint-002500 | 1200 | 65.08% | 0.5871 |
| scenery | ce_final | 1200 | 64.92% | 0.5906 |
| scenery | opd1_checkpoint-001000 | 1200 | 64.25% | 0.5941 |
| scenery | opd1_checkpoint-002500 | 1200 | 64.50% | 0.5888 |
| scenery | opd1_final | 1200 | 65.17% | 0.5925 |
| scenery | opd3_checkpoint-001000 | 1200 | 63.75% | 0.5886 |
| scenery | opd3_checkpoint-002500 | 1200 | 64.33% | 0.5863 |
| scenery | opd3_final | 1200 | 64.67% | 0.5903 |
| dev_combined | parent | 4320 | 60.00% | 0.6500 |
| dev_combined | ce_checkpoint-001000 | 4320 | 59.26% | 0.6479 |
| dev_combined | ce_checkpoint-002500 | 4320 | 60.16% | 0.6461 |
| dev_combined | ce_final | 4320 | 60.46% | 0.6528 |
| dev_combined | opd1_checkpoint-001000 | 4320 | 59.47% | 0.6468 |
| dev_combined | opd1_checkpoint-002500 | 4320 | 60.02% | 0.6454 |
| dev_combined | opd1_final | 4320 | 60.23% | 0.6503 |
| dev_combined | opd3_checkpoint-001000 | 4320 | 59.00% | 0.6418 |
| dev_combined | opd3_checkpoint-002500 | 4320 | 60.39% | 0.6397 |
| dev_combined | opd3_final | 4320 | 60.00% | 0.6452 |

parent_vs_ce: fixed 195 / regressed 175 / net +0.46pp; segment-bootstrap 95% CI [-0.48, +1.41]pp.

parent_vs_opd1: fixed 194 / regressed 184 / net +0.23pp; segment-bootstrap 95% CI [-0.72, +1.17]pp.

parent_vs_opd3: fixed 239 / regressed 222 / net +0.39pp; segment-bootstrap 95% CI [-0.69, +1.54]pp.

ce_vs_opd1: fixed 80 / regressed 90 / net -0.23pp; segment-bootstrap 95% CI [-0.82, +0.38]pp.

ce_vs_opd3: fixed 223 / regressed 226 / net -0.07pp; segment-bootstrap 95% CI [-1.17, +0.97]pp.

opd1_vs_opd3: fixed 204 / regressed 197 / net +0.16pp; segment-bootstrap 95% CI [-0.84, +1.12]pp.

checkpoint-001000/ce_vs_opd1: fixed 105 / regressed 96 / net +0.21pp; segment-bootstrap 95% CI [-0.48, +0.89]pp.

checkpoint-001000/ce_vs_opd3: fixed 173 / regressed 184 / net -0.25pp; segment-bootstrap 95% CI [-1.05, +0.56]pp.

checkpoint-001000/opd1_vs_opd3: fixed 133 / regressed 153 / net -0.46pp; segment-bootstrap 95% CI [-1.22, +0.35]pp.

checkpoint-002500/ce_vs_opd1: fixed 105 / regressed 111 / net -0.14pp; segment-bootstrap 95% CI [-0.83, +0.56]pp.

checkpoint-002500/ce_vs_opd3: fixed 200 / regressed 190 / net +0.23pp; segment-bootstrap 95% CI [-0.73, +1.23]pp.

checkpoint-002500/opd1_vs_opd3: fixed 152 / regressed 136 / net +0.37pp; segment-bootstrap 95% CI [-0.37, +1.17]pp.

final/ce_vs_opd1: fixed 80 / regressed 90 / net -0.23pp; segment-bootstrap 95% CI [-0.82, +0.38]pp.

final/ce_vs_opd3: fixed 154 / regressed 174 / net -0.46pp; segment-bootstrap 95% CI [-1.31, +0.44]pp.

final/opd1_vs_opd3: fixed 127 / regressed 137 / net -0.23pp; segment-bootstrap 95% CI [-1.03, +0.54]pp.

Each arm selects its own best combined-dev Judge among steps 1000/2500/5000; ties keep earlier steps. Intervals after dev selection are exploratory.

{"ce": "ce_final", "opd1": "opd1_final", "opd3": "opd3_checkpoint-002500"}

Historical weight-1 OPD switched to the fast implementation at step 1000. Matched objective/data/budget does not establish identical execution or random trajectories. Changing replica count changes rank-specific RNG streams, sampled answers and possibly gates, even with the same shuffled global QA batches and effective batch16. Use a same-version weight-1 rerun to confirm a small apparent weight-3 advantage.

Same execution recorded from step zero: False.
