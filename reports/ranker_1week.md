# Ranking stage

Train week `2020-09-09`, validation week `2020-09-16`.

```json
{
  "MAP@12": 0.03258,
  "recall@12": 0.06933,
  "NDCG@12": 0.04815,
  "coverage": 1.0,
  "n_features": 55,
  "train_rows": 1133391,
  "elapsed_sec": 334.7
}
```

## Feature importance (gain, top 30)

| feature                  |      gain |
|:-------------------------|----------:|
| score_r6_product_variant | 114645    |
| days_since_last          |  39227.5  |
| ca_days_since_last       |  37620.2  |
| rank_r6_product_variant  |  34113.8  |
| rrf_score                |  26283    |
| n_sold_1d                |  24702.5  |
| age                      |  22910.1  |
| rank_r2_popularity       |  22604.5  |
| price_ratio              |  22345.8  |
| garment_group_name       |  19888.8  |
| trend_ratio              |  18591.7  |
| art_avg_price            |  17770.1  |
| cust_avg_price           |  17240.5  |
| online_ratio             |  15601.5  |
| cg_days_since_last       |  15211.4  |
| days_since_first_sale    |  13864.1  |
| cust_std_price           |  13533.7  |
| score_r3_item_knn        |  13338.4  |
| rank_r3_item_knn         |  13095.3  |
| n_sold                   |  11585.4  |
| days_since_first         |  11487.3  |
| score_r2_popularity      |  10686.9  |
| max_discount             |  10514.4  |
| rank_r1_repurchase       |   9901.83 |
| cust_max_price           |   9893.22 |
| rank_r2b_global          |   9166.04 |
| cg_share                 |   8901.28 |
| n_sold_30d               |   8727.31 |
| n_buyers                 |   8648.28 |
| n_sources                |   8408.48 |
