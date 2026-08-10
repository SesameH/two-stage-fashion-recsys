# Serving

Sample: 3000 customers from the validation week. Model: 500 trees, 59 features.

## Latency by candidate budget (in-process)

|   n_candidates |   requests |   avg_candidates |   p50_ms |   p95_ms |   p99_ms |   mean_ms |   max_ms |
|---------------:|-----------:|-----------------:|---------:|---------:|---------:|----------:|---------:|
|             50 |       3000 |             50   |    31.7  |    35.1  |    42.52 |     32.24 |   165.05 |
|            100 |       3000 |            100   |    32.17 |    35.72 |    42.88 |     32.69 |   116.96 |
|            300 |       3000 |            300   |    33.75 |    36.43 |    43.21 |     34.24 |   138.66 |
|            500 |       3000 |            445.4 |    34.54 |    36.66 |    41.8  |     34.8  |    81.65 |

## Offline/online parity

The offline pipeline and the request path scored on the same customers. They read different storage by design, so an identical top 12 is the assertion worth making.

|                     |      value |
|:--------------------|-----------:|
| customers           | 3000       |
| MAP@12 offline      |    0.02992 |
| MAP@12 online       |    0.02992 |
| mean top-12 overlap |    1       |
| identical top 12    |    1       |

## Compression trade-off

Tree count truncated at predict time via `num_iteration`; MAP@12 recomputed on the same customers. Candidate budget fixed at the default.

|   trees |   pct_of_full |   MAP@12 |   p50_ms |   p95_ms |   p99_ms |   mean_ms |   max_ms |
|--------:|--------------:|---------:|---------:|---------:|---------:|----------:|---------:|
|     500 |           100 |  0.02992 |    33.71 |    36.35 |    44.74 |     34.15 |   162.19 |
|     250 |            50 |  0.02982 |    33.01 |    34.9  |    38.67 |     33.21 |    96.41 |
|     100 |            20 |  0.02961 |    32.12 |    34.04 |    37.52 |     32.36 |    89.17 |
|      50 |            10 |  0.02982 |    32.2  |    36.58 |    52.73 |     33.03 |   122.35 |
|      25 |             5 |  0.0286  |    31.72 |    33.44 |    35.02 |     31.88 |   106.07 |

<!-- manual: measurements below are hand-written and preserved by the generator -->

## Containerised, and the memory/latency tradeoff

Image 499 MB (`python:3.12-slim` + 7 serving dependencies + the 244 MB snapshot + the 3.4 MB
model). Measured inside the container, 240 requests each:

| snapshot mode | resident memory | p50 | p95 | p99 |
|---|---|---|---|---|
| `memory` — every table materialised, customer_key indexes | 3.1 GiB | 39.1 ms | 49.7 ms | 96.0 ms |
| `parquet` — Parquet views, row-group pruning, no indexes | **834 MiB** | 63.4 ms | 88.1 ms | **101.8 ms** |

3.8x less memory for 1.6x the median — and p99 is unchanged. `category_candidates` alone is
22.8M rows, which is what makes the in-memory copy expensive.

On this evidence alone `parquet` is the right default: tail latency is what a user experiences
and it did not move, while the median is 24 ms slower on a machine a quarter of the size. That
conclusion does not survive contact with Cloud Run — see below, which is why the deployment ships
`memory` instead.

It is also a cleaner comparison than the tree-count curve above. Pruning the model to a fifth of
its trees returns 4.7% of p50 for 1.0% of MAP@12; changing how the snapshot is opened returns a
74% memory reduction for 0% of MAP@12. Neither is a large win, but only one of them touches
accuracy at all.

For reference, in-process (no HTTP, no container) is p50 34.4 ms / p99 39.1 ms, so the
container and HTTP stack account for the rest.

## Cold start

Scale to zero is what keeps this in the few-dollars-a-month range, and the cost is a cold start on
the first click. Measured in a local container, since that is where both modes can be compared:

| | parquet mode | memory mode |
|---|---|---|
| container start to first HTTP 200 | **7 s** | ~55 s |
| first request after boot | 357 ms | — |
| second request | 162 ms | — |

The 8x difference has the same cause as the memory difference: `memory` mode materialises 22.8M
rows of `category_candidates` before it can answer anything, `parquet` mode opens a view.

An earlier version of this section read the 7 s figure as licence to scale to zero, and said that
"had cold start stayed at 55 s the honest choice would have been an always-on machine". The
deployment then shipped `memory` — the 55 s branch — because the latency comparison inverted in
production, and the sentence stayed. Both halves could not be true at once, so the deployed cold
start was measured instead of inferred from either:

| after 17 minutes idle, `memory` mode on Cloud Run | |
|---|---|
| first request (instance cold) | **22.5 s** |
| second request | 0.20 s |
| `/api/demo/explain` warm | 0.81 s |

22.5 s, against 55 s for the same mode in a local container. Cloud Run's 2 vCPU builds the
in-memory tables faster than the laptop container did, so the local figure was pessimistic — the
third time in this project that a development-machine measurement failed to transfer, and the only
one that transferred in the project's favour.

It is still 22.5 s, which is not "a few seconds", and the README says so plainly rather than
hedging. The alternatives are both worse for a portfolio link: `min-instances 1` removes the wait
and costs roughly $25/month for an idle 4 GiB instance, and `parquet` mode starts in a fraction of
the time but doubles the median for every warm request thereafter. Waiting once beats paying
monthly or being slower forever.

## Deployed on Cloud Run — where the local conclusion inverted

Live at `https://hm-recsys-vpllq7symq-an.a.run.app` (2 vCPU, scale to zero, asia-northeast1).
Both snapshot modes were re-measured against the deployed service, because the tradeoff chosen
on a laptop turned out not to transfer:

| snapshot mode | local container p50 | Cloud Run p50 | Cloud Run p95 |
|---|---|---|---|
| `parquet` | 63 ms | 294 ms | 336 ms |
| `memory` | 39 ms | **147 ms** | 210 ms |

Locally, parquet mode costs 62% of the median to save 74% of resident memory — clearly worth
it. On Cloud Run it costs 100%. The cause is the filesystem: Cloud Run's container filesystem is
a network-backed overlay, not local NVMe, so a mode built around reading files per request pays
a much higher price there. The deployment therefore runs `memory` mode on a 4 GiB machine, which
scale-to-zero makes nearly free.

End-to-end from a client in Taiwan is p50 375 ms against a server-side 147 ms; the remaining
~228 ms is TLS and round trips to Tokyo, not the recommender.

The general point is the same one this project keeps running into: a number measured on the
development machine is a hypothesis about production, not a result. This one happened to invert.
