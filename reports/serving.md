# Serving

Sample: 400 customers from the validation week. Model: 500 trees, 63 features.

## Latency by candidate budget (in-process)

|   n_candidates |   requests |   avg_candidates |   p50_ms |   p95_ms |   p99_ms |   mean_ms |   max_ms |
|---------------:|-----------:|-----------------:|---------:|---------:|---------:|----------:|---------:|
|             50 |        400 |               50 |    32.47 |    36.17 |    52.63 |     33.34 |   109.55 |
|            100 |        400 |              100 |    32.6  |    35.87 |    48.82 |     33.21 |    74.12 |
|            300 |        400 |              300 |    34.41 |    36.91 |    39.05 |     34.63 |    42.34 |
|            500 |        400 |              448 |    35.56 |    38.26 |    44.73 |     35.96 |   128.31 |

## Compression trade-off

Tree count truncated at predict time via `num_iteration`; MAP@12 recomputed on the same customers. Candidate budget fixed at the default.

|   trees |   pct_of_full |   MAP@12 |   p50_ms |   p95_ms |   p99_ms |   mean_ms |   max_ms |
|--------:|--------------:|---------:|---------:|---------:|---------:|----------:|---------:|
|     500 |           100 |  0.02223 |    34.8  |    39.8  |    60.53 |     35.66 |    85.52 |
|     250 |            50 |  0.02174 |    33.63 |    35.79 |    38.04 |     34.02 |   107.08 |
|     100 |            20 |  0.02094 |    32.91 |    36.95 |    52.44 |     33.91 |   140.43 |
|      50 |            10 |  0.02031 |    32.4  |    35.02 |    45.12 |     32.82 |    61.74 |
|      25 |             5 |  0.01942 |    32.17 |    35.43 |    48.47 |     32.96 |   134.91 |

## Containerised, and the memory/latency tradeoff

Image 1.66 GB (`python:3.12-slim` + 7 serving dependencies + the 320 MB snapshot + the 3.4 MB
model). Measured inside the container, 240 requests each:

| snapshot mode | resident memory | p50 | p95 | p99 |
|---|---|---|---|---|
| `memory` — every table materialised, customer_key indexes | 3.1 GiB | 39.1 ms | 49.7 ms | 96.0 ms |
| `parquet` — Parquet views, row-group pruning, no indexes | **834 MiB** | 63.4 ms | 88.1 ms | **101.8 ms** |

3.8x less memory for 1.6x the median — and p99 is unchanged. `category_candidates` alone is
22.8M rows, which is what makes the in-memory copy expensive.

Deployment uses `parquet` (`HM_SERVE_MODE`, see fly.toml). Tail latency is what a user
experiences and it did not move; the median is 24 ms slower on a machine a quarter of the size.
This is also the more honest comparison than the tree-count curve above: pruning the model
traded 13% of MAP@12 for 4% of p50, while changing how the snapshot is opened traded 0% of
MAP@12 for a 74% memory reduction.

For reference, in-process (no HTTP, no container) is p50 34.4 ms / p99 39.1 ms, so the
container and HTTP stack account for the rest.

## Cold start

`min_machines_running = 0` on Fly means the machine sleeps when idle, which is what keeps this
in the few-dollars-a-month range. The cost is a cold start on the first click, so it was
measured rather than assumed:

| | parquet mode | memory mode |
|---|---|---|
| container start to first HTTP 200 | **7 s** | ~55 s |
| first request after boot | 357 ms | — |
| second request | 162 ms | — |

The 8x difference has the same cause as the memory difference: `memory` mode materialises 22.8M
rows of `category_candidates` before it can answer anything, `parquet` mode opens a view. Adding
Fly's machine boot, a cold click lands around 10-15 s and every request after that is warm.

That is what makes scale-to-zero acceptable here. Had cold start stayed at 55 s the honest
choice would have been to pay for an always-on machine, because nobody clicking a portfolio link
waits a minute.
