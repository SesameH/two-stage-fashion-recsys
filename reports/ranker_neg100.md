# Ranking stage

Train weeks `['2020-09-02', '2020-09-09']`, validation week `2020-09-16`.

```json
{
  "MAP@12": 0.03272,
  "recall@12": 0.06897,
  "NDCG@12": 0.04795,
  "coverage": 1.0,
  "n_features": 55,
  "train_rows": 11364419,
  "train_weeks": [
    "2020-09-02",
    "2020-09-09"
  ],
  "neg_ratio": 100,
  "elapsed_sec": 894.6
}
```

## Feature importance (gain, top 30)

| feature                  |     gain |
|:-------------------------|---------:|
| score_r6_product_variant | 194632   |
| ca_days_since_last       | 162394   |
| days_since_last          | 111487   |
| garment_group_name       | 101600   |
| rank_r6_product_variant  |  72324.4 |
| rrf_score                |  56967.2 |
| age                      |  49911.4 |
| days_since_first_sale    |  39631   |
| price_ratio              |  38332.1 |
| rank_r2_popularity       |  37681.9 |
| n_sold_1d                |  35558.7 |
| n_sold                   |  34858.8 |
| score_r3_item_knn        |  33933.3 |
| art_avg_price            |  33725.3 |
| section_no               |  32733.1 |
| score_r2_popularity      |  32117   |
| trend_ratio              |  31422.1 |
| cust_avg_price           |  28210.7 |
| online_ratio             |  26606.2 |
| rank_r3_item_knn         |  26135.9 |
| rank_r1_repurchase       |  25491.9 |
| n_sold_30d               |  24309.4 |
| cg_days_since_last       |  24193.9 |
| max_discount             |  22326.6 |
| n_purchases_7d           |  22227.6 |
| cust_max_price           |  20413.6 |
| n_buyers                 |  19907.1 |
| cust_std_price           |  19202.6 |
| score_r2b_global         |  19038.8 |
| rank_r2b_global          |  18687.2 |
