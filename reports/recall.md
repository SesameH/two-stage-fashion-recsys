# Retrieval layer

- Validation week starting `2020-09-16`, 68,984 customers
- All strategies read only `t_dat < 2020-09-16`
- Union recall@300: **0.27372**

| strategy                              |   recall@300 |   coverage |   avg_candidates |   sec |
|:--------------------------------------|-------------:|-----------:|-----------------:|------:|
| R1 repurchase                         |      0.03821 |     0.7323 |              7.7 |   0.2 |
| R2 popularity (age)                   |      0.19376 |     1      |            199.9 |   4.6 |
| R2b popularity (global)               |      0.23119 |     1      |            300   |   1.3 |
| R3 item-kNN                           |      0.06606 |     0.7303 |             92.4 |   1.8 |
| R5 category                           |      0.04492 |     0.7323 |             36   |   0.5 |
| R6 product variant                    |      0.06396 |     0.7311 |             30   |   0.5 |
| R4 ALS                                |      0.04885 |     0.8887 |             44.4 |   9.2 |
| UNION (all)                           |      0.27372 |     1      |            300   |   0.6 |
| union without R1 repurchase           |      0.27367 |     1      |            300   |   0   |
| union without R2 popularity (age)     |      0.2547  |     1      |            300   |   0   |
| union without R2b popularity (global) |      0.25309 |     1      |            279.4 |   0   |
| union without R3 item-kNN             |      0.27664 |     1      |            300   |   0   |
| union without R5 category             |      0.27291 |     1      |            300   |   0   |
| union without R6 product variant      |      0.2717  |     1      |            300   |   0   |
| union without R4 ALS                  |      0.27242 |     1      |            300   |   0   |

## Recall vs candidate budget (union)

| budget | 12 | 50 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|
| recall | 0.05067 | 0.11726 | 0.16701 | 0.22733 | 0.27372 | 0.33641 |

The `union without X` rows are a leave-one-out ablation: the drop from the UNION row
is the unique contribution of strategy X, which is the only honest way to justify
keeping five strategies instead of one.
