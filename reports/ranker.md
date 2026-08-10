# Ranking stage

Train weeks `['2020-08-12', '2020-08-19', '2020-08-26', '2020-09-02', '2020-09-09']`, validation week `2020-09-16`.

```json
{
  "MAP@12": 0.03296,
  "recall@12": 0.06883,
  "NDCG@12": 0.04834,
  "coverage": 1.0,
  "n_features": 59,
  "train_rows": 5620587,
  "train_weeks": [
    "2020-08-12",
    "2020-08-19",
    "2020-08-26",
    "2020-09-02",
    "2020-09-09"
  ],
  "neg_ratio": 20,
  "elapsed_sec": 765.9
}
```

## Feature importance (gain, top 30)

| feature                  |     gain |
|:-------------------------|---------:|
| score_r6_product_variant | 399617   |
| rank_r6_product_variant  | 210414   |
| cpc_days_since_last      | 147059   |
| rrf_score                | 141218   |
| garment_group_name       | 128047   |
| n_sold_1d                | 116866   |
| days_since_last          | 114200   |
| rank_r2_popularity       | 107613   |
| ca_days_since_last       |  96926.1 |
| trend_ratio              |  53662.3 |
| age                      |  51385.3 |
| price_ratio              |  45494.2 |
| days_since_first_sale    |  42247   |
| product_group_name       |  42185.4 |
| rank_r1_repurchase       |  37538.3 |
| online_ratio             |  35525.3 |
| art_avg_price            |  33293.5 |
| n_buyers                 |  31002.6 |
| max_discount             |  27453.5 |
| price_z                  |  26531   |
| n_sold_30d               |  25420.1 |
| rank_r2b_global          |  25002.3 |
| score_r2b_global         |  24650.8 |
| price_pctile             |  22981.2 |
| cust_avg_price           |  22747.3 |
| n_sold                   |  22679.1 |
| department_no            |  21825.5 |
| section_no               |  21443   |
| cg_days_since_last       |  20479.6 |
| score_r2_popularity      |  18833.6 |
