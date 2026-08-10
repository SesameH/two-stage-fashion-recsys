# Tree count

Trained on ['2020-08-12', '2020-08-19', '2020-08-26', '2020-09-02'], tree count selected on `2020-09-09`, reported on the validation week elsewhere. 15,000 customers, `bagging_by_query=False`.

The selection week is disjoint from both the training weeks and the validation week, so
choosing a tree count here does not tune the number the README quotes.

|   trees |   MAP@12 |   vs_best |
|--------:|---------:|----------:|
|     100 |  0.03185 |   -0.0127 |
|     250 |  0.0322  |   -0.0019 |
|     500 |  0.03218 |   -0.0025 |
|     750 |  0.03226 |    0      |
|    1000 |  0.03205 |   -0.0065 |
|    1250 |  0.03206 |   -0.0062 |
|    1500 |  0.03211 |   -0.0046 |

<!-- manual: measurements below are hand-written and preserved by the generator -->

## Run-to-run variance, which turned out to be the real finding

The same protocol was run twice. The tables disagree about the argmax:

| trees | 100 | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|---|
| run 1 | 0.03203 | 0.03232 | **0.03234** | 0.03220 | 0.03201 | 0.03210 | 0.03198 |
| run 2 (above) | 0.03185 | 0.03220 | 0.03218 | **0.03226** | 0.03205 | 0.03206 | 0.03211 |

Same seed, same data, same parameters. LightGBM's `deterministic` parameter defaults to `false`,
and with multithreaded training the histogram build order is not fixed, so two fits of the same
configuration are not bit-identical. The spread here is ~0.5%.

That is the same size as the gaps between tree counts, which settles the question more firmly than
either table alone: anywhere from 250 to 1500 trees is indistinguishable, the argmax is not stable
enough to be worth chasing, and 500 stays because it is in the plateau and cheaper than 750.

Also worth stating plainly: 0.5% run-to-run variance on the metric means no single-run comparison
below ~1% is evidence of anything. The 5.6% week-to-week noise floor already implied that; this
puts a floor under it from a second, independent direction.

## bagging_by_query

Same protocol, `--bagging-by-query`: 0.03206 at 500 trees against 0.03234 for the default in the
same run, −0.9%. LightGBM's row-level bagging ignores query boundaries, so under `lambdarank` each
tree sees a truncated candidate list per customer rather than fewer whole customers; sampling whole
queries is the theoretically cleaner choice and measures no better than noise.

