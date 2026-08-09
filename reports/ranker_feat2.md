# Ranking stage

Train weeks `['2020-09-02', '2020-09-09']`, validation week `2020-09-16`.

```json
{
  "MAP@12": 0.03342,
  "recall@12": 0.07088,
  "NDCG@12": 0.04906,
  "coverage": 1.0,
  "n_features": 63,
  "train_rows": 2362899,
  "train_weeks": [
    "2020-09-02",
    "2020-09-09"
  ],
  "neg_ratio": 20,
  "elapsed_sec": 822.3
}
```

## Feature importance (gain, top 30)

| feature                  |     gain |
|:-------------------------|---------:|
| score_r6_product_variant | 180317   |
| rank_r6_product_variant  |  69235.5 |
| cpc_days_since_last      |  60774.4 |
| days_since_last          |  59350.9 |
| rrf_score                |  56379.9 |
| rank_r2_popularity       |  53783   |
| n_sold_1d                |  44268   |
| garment_group_name       |  43123.1 |
| ca_days_since_last       |  42062.6 |
| age                      |  27371   |
| days_since_first_sale    |  23920.7 |
| price_ratio              |  23639.3 |
| trend_ratio              |  21430.3 |
| rank_r3_item_knn         |  20738   |
| online_ratio             |  19942.8 |
| price_z                  |  18767.7 |
| n_sold                   |  18067.1 |
| score_r3_item_knn        |  17212   |
| max_discount             |  16539.6 |
| n_sources                |  16232.3 |
| n_sold_30d               |  16127.1 |
| cust_avg_price           |  15988   |
| n_buyers                 |  15581.6 |
| art_avg_price            |  14572.9 |
| rank_r1_repurchase       |  14468.9 |
| n_purchases_7d           |  14394.7 |
| cg_days_since_last       |  13595.6 |
| cust_std_price           |  12019.4 |
| cust_max_price           |  11918   |
| score_r2_popularity      |  11451.3 |
