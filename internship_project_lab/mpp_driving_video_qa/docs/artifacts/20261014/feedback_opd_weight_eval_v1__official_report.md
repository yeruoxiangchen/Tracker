# OPD weight comparison

Original five frames and question, no reference/draft in student evaluation. 40k QA, two epochs, 5000 updates; CE vs OPD weight 1 vs weight 3. Historical scores are read without rewriting them. The reused official 500 is not a new blind test.

| Split | Model | N | Judge | Token F1 |
|---|---|---:|---:|---:|
| eval | parent | 500 | 54.20% | 0.3196 |
| eval | ce_final | 500 | 55.40% | 0.3169 |
| eval | opd1_final | 500 | 54.80% | 0.3180 |
| eval | opd3_checkpoint-002500 | 500 | 54.20% | 0.3170 |

parent_vs_ce: fixed 23 / regressed 17 / net +1.20pp; segment-bootstrap 95% CI [-1.20, +3.60]pp.

parent_vs_opd1: fixed 23 / regressed 20 / net +0.60pp; segment-bootstrap 95% CI [-1.80, +3.20]pp.

parent_vs_opd3: fixed 23 / regressed 23 / net +0.00pp; segment-bootstrap 95% CI [-2.20, +2.40]pp.

ce_vs_opd1: fixed 10 / regressed 13 / net -0.60pp; segment-bootstrap 95% CI [-2.40, +1.40]pp.

ce_vs_opd3: fixed 17 / regressed 23 / net -1.20pp; segment-bootstrap 95% CI [-3.80, +1.40]pp.

opd1_vs_opd3: fixed 17 / regressed 20 / net -0.60pp; segment-bootstrap 95% CI [-3.00, +1.80]pp.

Each arm selects its own best combined-dev Judge among steps 1000/2500/5000; ties keep earlier steps. Intervals after dev selection are exploratory.

{"ce": "ce_final", "opd1": "opd1_final", "opd3": "opd3_checkpoint-002500"}

Historical weight-1 OPD switched to the fast implementation at step 1000. Matched objective/data/budget does not establish identical execution or random trajectories. Changing replica count changes rank-specific RNG streams, sampled answers and possibly gates, even with the same shuffled global QA batches and effective batch16. Use a same-version weight-1 rerun to confirm a small apparent weight-3 advantage.

Same execution recorded from step zero: False.
