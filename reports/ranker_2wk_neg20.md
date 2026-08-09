# Ranking stage

Train weeks `['2020-09-02', '2020-09-09']`, validation week `2020-09-16`.

```json
{
  "MAP@12": 0.03303,
  "recall@12": 0.07049,
  "NDCG@12": 0.04868,
  "coverage": 1.0,
  "n_features": 55,
  "train_rows": 2362899,
  "train_weeks": [
    "2020-09-02",
    "2020-09-09"
  ],
  "neg_ratio": 20,
  "elapsed_sec": 460.6
}
```

## Feature importance (gain, top 30)

| feature                  |     gain |
|:-------------------------|---------:|
| score_r6_product_variant | 217098   |
| ca_days_since_last       |  73659.1 |
| days_since_last          |  60236.3 |
| rank_r6_product_variant  |  58448.8 |
| rrf_score                |  54584.6 |
| rank_r2_popularity       |  53685   |
| garment_group_name       |  43238.1 |
| n_sold_1d                |  41290.2 |
| price_ratio              |  29285.3 |
| age                      |  28706.3 |
| days_since_first_sale    |  28150.9 |
| trend_ratio              |  24726.7 |
| rank_r3_item_knn         |  23367.3 |
| art_avg_price            |  21657.9 |
| online_ratio             |  21286.9 |
| score_r3_item_knn        |  20839.2 |
| n_sources                |  20793.5 |
| cust_avg_price           |  18400.1 |
| n_sold                   |  17354.3 |
| max_discount             |  16866.9 |
| cg_days_since_last       |  16554.3 |
| rank_r1_repurchase       |  15918   |
| n_buyers                 |  15491.9 |
| n_sold_30d               |  15229.6 |
| cust_std_price           |  13767.2 |
| n_purchases_7d           |  13724.5 |
| score_r2_popularity      |  13302.4 |
| cust_max_price           |  12465.6 |
| rank_r2b_global          |  12104.8 |
| days_since_first         |  11385.6 |
