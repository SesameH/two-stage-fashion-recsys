# Error analysis

Validation week `2020-09-16`, 68,984 customers.

## By prior activity (recency)

| recency_segment   |   customers |   map_model |   map_b2 |   lift |
|:------------------|------------:|------------:|---------:|-------:|
| 0 never purchased |        5572 |     0.01002 |  0.00788 |  0.271 |
| 1 active <=7d     |       12670 |     0.10156 |  0.08367 |  0.214 |
| 2 active 8-30d    |       19416 |     0.0283  |  0.02152 |  0.315 |
| 3 active 31-90d   |       18431 |     0.01381 |  0.00774 |  0.783 |
| 4 dormant >90d    |       12895 |     0.00987 |  0.00768 |  0.285 |

## By purchase frequency

| frequency_segment   |   customers |   map_model |   map_b2 |   lift |
|:--------------------|------------:|------------:|---------:|-------:|
| 0 none              |        5572 |     0.01002 |  0.00788 |  0.271 |
| 1 low (1-5)         |        5210 |     0.03924 |  0.0327  |  0.2   |
| 2 mid (6-20)        |       13323 |     0.03396 |  0.02633 |  0.29  |
| 3 high (21-60)      |       22322 |     0.03277 |  0.02528 |  0.297 |
| 4 very high (60+)   |       22557 |     0.03677 |  0.02812 |  0.307 |

## By age

| age_segment   |   customers |   map_model |   map_b2 |   lift |
|:--------------|------------:|------------:|---------:|-------:|
| 25-34         |       21311 |     0.03178 |  0.02512 |  0.265 |
| 35-49         |       12999 |     0.03345 |  0.02733 |  0.224 |
| 50+           |       15686 |     0.03681 |  0.02759 |  0.334 |
| <25           |       18696 |     0.03069 |  0.02304 |  0.332 |
| unknown       |         292 |     0.03689 |  0.03247 |  0.136 |

## By item popularity decile

Deciles are over trailing-30-day sales; decile 0 means the article had not sold at all before the cutoff.

|   decile |   purchases |   hit_rate |   share_of_truth |
|---------:|------------:|-----------:|-----------------:|
|        0 |        9279 |     0      |           0.0434 |
|        1 |        1276 |     0.0016 |           0.006  |
|        2 |        1012 |     0.002  |           0.0047 |
|        3 |        1556 |     0.009  |           0.0073 |
|        4 |        2170 |     0.0101 |           0.0102 |
|        5 |        3551 |     0.0104 |           0.0166 |
|        6 |        5880 |     0.0146 |           0.0275 |
|        7 |        8944 |     0.0171 |           0.0418 |
|        8 |       14802 |     0.0235 |           0.0693 |
|        9 |       32470 |     0.0276 |           0.1519 |
|       10 |      132788 |     0.0736 |           0.6213 |

## Diversity

```json
{
  "distinct_items_recommended": 14542,
  "catalogue_coverage": 0.1378,
  "category_entropy_bits": 1.743,
  "max_category_entropy_bits": 3.807,
  "most_recommended_item_share": 0.0718
}
```
