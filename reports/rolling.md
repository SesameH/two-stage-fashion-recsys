# Rolling evaluation

One model, scored on consecutive weeks without retraining.

Only the row for `2020-09-16` is a valid generalisation estimate. It is the last
week in the dataset, so it is the only week the model can be scored on without having
been fitted on later data — which is why there is exactly one.

Every earlier row is contaminated, in one of two ways: a week inside the training
range is straightforwardly in-sample, and a week *before* the training range was
scored by a model fitted on information from after it. Neither simulates production.
They are reported because the size of the in-sample gap is itself diagnostic: a large
gap means overfitting, and watching it shrink across model versions is how the
multi-week training decision was justified (see README).

| week       |   customers |   MAP@12 |      B2 |   lift |
|:-----------|------------:|---------:|--------:|-------:|
| 2020-09-02 |       75822 |  0.03479 | 0.02245 |  0.549 |
| 2020-09-09 |       72019 |  0.03574 | 0.02487 |  0.437 |
| 2020-09-16 |       68984 |  0.03296 | 0.02557 |  0.289 |

In-sample gap for this model: 0.03574 / 0.03296 = **+8.4%**.

Week-to-week variation for one fixed model: 0.03479 on 2020-09-02 against 0.03296 on
2020-09-16 is 5.6%, which is the threshold any improvement has to clear to mean anything.

## Candidate budget

Sampled 8,000 customers from the 2020-09-16 validation week.

|   budget |   recall@budget |   MAP@12 |
|---------:|----------------:|---------:|
|      100 |         0.16202 |  0.03209 |
|      300 |         0.27706 |  0.03233 |
|      500 |         0.33263 |  0.03232 |

<!-- manual: measurements below are hand-written and preserved by the generator -->

## Across model versions

The in-sample gap, measured the same way each time. These rows come from earlier runs of this
script against the models in `models/`; only the last one is the deployed model.

| model | in-sample gap |
|---|---|
| 1 training week, 55 features | +54.8% |
| 5 training weeks, 55 features | +17.8% |
| 5 training weeks, 63 features | +9.2% |
| **5 training weeks, 59 features (deployed)** | **+8.4%** |

Adding features moved MAP@12 by little and narrowed the gap each time; removing the four
retrieval features the serving snapshot could not supply narrowed it again while costing 1.7% of
the headline. Judging a feature set by its headline metric alone would have missed both.

