# Resume block — every number below is measured, with the command that reproduces it

```latex
\project{Two-Stage Fashion Recommendation System}{Python, LightGBM, DuckDB, FastAPI, Docker}
\begin{itemize}
  \item Built a two-stage recommender (retrieval + ranking) over 31.8M H\&M transactions,
        converting 3.5\,GB of CSV to 308\,MB of partitioned Parquet in 5.5\,s and computing all
        features in DuckDB SQL under a strict \texttt{as\_of} time split.
  \item Combined six retrieval strategies (deep popularity, age-bucketed popularity, exact
        repurchase, product-variant, category, item-kNN) fused by reciprocal rank fusion,
        reaching recall@300 of 0.274 over 300 candidates per customer; leave-one-out ablation
        showed one strategy lowered union recall and was demoted to a feature-only signal.
  \item Trained a LightGBM LambdaRank model on 63 engineered features and 5.6M labelled
        candidate rows, achieving MAP@12 of 0.0335 versus 0.0256 for a repurchase-plus-
        bestseller baseline (+31\%), on the competition's own metric definition.
  \item Served the model behind FastAPI in a 1.66\,GB container at p50 34\,ms / p99 39\,ms for
        300 candidates; switching the feature snapshot from in-memory tables to Parquet views
        cut resident memory 3.1\,GB $\to$ 834\,MB and cold start 55\,s $\to$ 7\,s with no change
        to p99 or accuracy.
  \item Enforced training-serving consistency with a single shared feature-assembly path and an
        automated offline/online parity test, plus a leakage test that rebuilds features against
        a truncated transaction log and asserts equality.
\end{itemize}
```

## Reproducing each claim

| claim | command |
|---|---|
| 31.8M rows, 3.5 GB -> 308 MB, 5.5 s | `make data` |
| baselines 0.00875 / 0.02211 / 0.02557 / 0.00942 | `make baselines` |
| recall@300 0.274, leave-one-out ablation | `make recall` |
| MAP@12 0.0335, 63 features, 5.6M rows | `make train` |
| segment and long-tail failure modes | `make analyze` |
| in-sample gap +9.2%, budget sweep | `make rolling` |
| p50/p99, compression curve, memory modes | `make bench` |
| leakage + offline/online parity | `make test` |

## Lines deliberately NOT claimed

- **No leaderboard rank.** The metric definition matches the competition's (buyers only), and
  0.0335 sits above a published silver-medal solution's 0.02996 — but validation is on the last
  week of the public dataset, not the competition's held-out week. Same scale, different week.
- **No "model compression reduced latency".** It was measured and rejected: cutting to 20% of
  the trees cost 13% of MAP@12 and returned 4% of p50. The memory/latency win came from the
  snapshot, not the model.
- **No QPS figure.** Latency was measured request-at-a-time, not under concurrent load.
- **No deep learning, NLP or CV.** Error analysis quantifies the gap this leaves: articles with
  no prior sales are 4.3% of ground truth and are hit 0.00% of the time.
