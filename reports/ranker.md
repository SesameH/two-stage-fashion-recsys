# Ranking stage

Train weeks `['2020-08-12', '2020-08-19', '2020-08-26', '2020-09-02', '2020-09-09']`, validation week `2020-09-16`.

```json
{
  "MAP@12": 0.03354,
  "recall@12": 0.07112,
  "NDCG@12": 0.04944,
  "coverage": 1.0,
  "n_features": 63,
  "train_rows": 5620587,
  "train_weeks": [
    "2020-08-12",
    "2020-08-19",
    "2020-08-26",
    "2020-09-02",
    "2020-09-09"
  ],
  "neg_ratio": 20,
  "elapsed_sec": 837.8
}
```

## Feature importance (gain, top 30)

| feature                  |     gain |
|:-------------------------|---------:|
| score_r6_product_variant | 394694   |
| rank_r6_product_variant  | 196214   |
| garment_group_name       | 145372   |
| cpc_days_since_last      | 127962   |
| n_sold_1d                | 123817   |
| rrf_score                | 123531   |
| days_since_last          | 117679   |
| rank_r2_popularity       | 102713   |
| ca_days_since_last       |  98574.5 |
| rank_r3_item_knn         |  54820.2 |
| age                      |  50731.4 |
| trend_ratio              |  49146.2 |
| days_since_first_sale    |  42623   |
| price_ratio              |  42080.6 |
| online_ratio             |  37913.2 |
| rank_r1_repurchase       |  37076.1 |
| score_r3_item_knn        |  33243.9 |
| art_avg_price            |  32616.8 |
| product_group_name       |  31459   |
| n_purchases_7d           |  30626.1 |
| n_sources                |  30352   |
| max_discount             |  29600.6 |
| n_buyers                 |  27629.3 |
| n_sold                   |  27137.1 |
| score_r2b_global         |  25692.4 |
| n_sold_30d               |  23815.6 |
| rank_r2b_global          |  23780.7 |
| price_z                  |  23511.5 |
| price_pctile             |  22421.4 |
| cust_avg_price           |  20808.5 |
