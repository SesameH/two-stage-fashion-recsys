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
