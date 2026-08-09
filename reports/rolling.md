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
| 2020-09-09 |       72019 |  0.03663 | 0.02487 |  0.473 |
| 2020-09-16 |       68984 |  0.03354 | 0.02557 |  0.312 |

In-sample gap for this model: 0.03663 / 0.03354 = **+9.2%**.

Across model versions, measured the same way:

| model | in-sample gap |
|---|---|
| 1 training week, 55 features | +54.8% |
| 5 training weeks, 55 features | +17.8% |
| 5 training weeks, 63 features | **+9.2%** |

The added features moved MAP@12 by only +1.2% — inside the noise floor — yet narrowed the
generalisation gap again. Judging a feature set by its headline metric alone would have missed
that.

## Candidate budget

Sampled 8,000 customers from the 2020-09-16 validation week.

|   budget |   recall@300 |   MAP@12 |
|---------:|-------------:|---------:|
|      300 |      0.27706 |  0.03343 |
