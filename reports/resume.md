# Resume block — every number below is measured, with the command that reproduces it

```latex
\project{Two-Stage Fashion Recommendation System}{Python, LightGBM, DuckDB, FastAPI, Docker}
\begin{itemize}
  \item Built a two-stage recommender (retrieval + ranking) over 31.8M H\&M transactions,
        converting 3.5\,GB of CSV to 308\,MB of partitioned Parquet in 5.5\,s and computing all
        features in DuckDB SQL under a strict \texttt{as\_of} time split.
  \item Combined five retrieval strategies (deep popularity, age-bucketed popularity, exact
        repurchase, product-variant, category) fused by reciprocal rank fusion, reaching recall@300
        of 0.274 over 300 candidates per customer; a sixth was built, shown by leave-one-out
        ablation to lower union recall, and then cut outright when a parity check found the serving
        snapshot could not supply its features, leaving them constant in production.
  \item Trained a LightGBM LambdaRank model on 59 engineered features and 5.6M labelled
        candidate rows, achieving MAP@12 of 0.0330 versus 0.0256 for a repurchase-plus-
        bestseller baseline (+29\%), on the competition's own metric definition.
  \item Served the model behind FastAPI in a container at p50 34\,ms / p99 39\,ms in-process for
        300 candidates, deployed on Cloud Run at p50 147\,ms; measured two snapshot modes on both
        and found the local winner (Parquet views: 3.1\,GB $\to$ 834\,MB resident, 55\,s $\to$
        7\,s cold start) doubles median latency in production, so the deployment ships the other
        one.
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
| MAP@12 0.0330, 59 features, 5.6M rows | `make train` |
| segment and long-tail failure modes | `make analyze` |
| in-sample gap +8.4%, noise floor 5.6%, budget sweep | `make rolling` |
| p50/p99, offline/online parity, compression curve | `make bench` |
| leakage + offline/online parity | `make test` |

## Lines deliberately NOT claimed

- **No leaderboard rank.** The metric definition matches the competition's (buyers only), and
  0.0330 sits above a published silver-medal solution's 0.02996 — but validation is on the last
  week of the public dataset, not the competition's held-out week. Same scale, different week.
- **No "model compression reduced latency".** It was measured and rejected: cutting to 20% of the
  trees returns 4.7% of p50, which is what a request spends outside the model anyway. (The first
  version of that measurement, on 400 customers, put the accuracy cost at 13%; at 3,000 it is
  1.0%. The verdict held, its stated reason did not.)
- **No QPS figure.** Latency was measured request-at-a-time, not under concurrent load.
- **No claim that the offline and online paths are identical.** They agree on every shared
  feature block and on retrieval fusion, which is tested; what the tests initially did not cover
  was a feature the snapshot could not supply at all, and that turned out to be four of them.
  `/metrics` now publishes `unservable_features` so the gap is visible rather than assumed away.
- **No deep learning, NLP or CV.** Error analysis quantifies the gap this leaves: articles with
  no prior sales are 4.3% of ground truth and are hit 0.00% of the time.
