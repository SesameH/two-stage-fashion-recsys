# Error analysis

Validation week `2020-09-16`, 68,984 customers.

## By prior activity (recency)

| recency_segment   |   customers |   map_model |   map_b2 |   lift |
|:------------------|------------:|------------:|---------:|-------:|
| 0 never purchased |        5572 |     0.01025 |  0.00788 |  0.301 |
| 1 active <=7d     |       12670 |     0.10062 |  0.08367 |  0.202 |
| 2 active 8-30d    |       19416 |     0.02738 |  0.02152 |  0.272 |
| 3 active 31-90d   |       18431 |     0.01344 |  0.00774 |  0.735 |
| 4 dormant >90d    |       12895 |     0.01059 |  0.00768 |  0.379 |

## By purchase frequency

| frequency_segment   |   customers |   map_model |   map_b2 |   lift |
|:--------------------|------------:|------------:|---------:|-------:|
| 0 none              |        5572 |     0.01025 |  0.00788 |  0.301 |
| 1 low (1-5)         |        5210 |     0.03978 |  0.0327  |  0.216 |
| 2 mid (6-20)        |       13323 |     0.03421 |  0.02633 |  0.299 |
| 3 high (21-60)      |       22322 |     0.03221 |  0.02528 |  0.274 |
| 4 very high (60+)   |       22557 |     0.03585 |  0.02812 |  0.275 |

## By age

| age_segment   |   customers |   map_model |   map_b2 |   lift |
|:--------------|------------:|------------:|---------:|-------:|
| 25-34         |       21311 |     0.03125 |  0.02512 |  0.244 |
| 35-49         |       12999 |     0.03301 |  0.02733 |  0.208 |
| 50+           |       15686 |     0.03598 |  0.02759 |  0.304 |
| <25           |       18696 |     0.0309  |  0.02304 |  0.341 |
| unknown       |         292 |     0.03603 |  0.03247 |  0.11  |

## By item popularity decile

Deciles are over trailing-30-day sales; decile 0 means the article had not sold at all before the cutoff.

|   decile |   purchases |   hit_rate |   share_of_truth |
|---------:|------------:|-----------:|-----------------:|
|        0 |        9279 |     0      |           0.0434 |
|        1 |        1044 |     0.0057 |           0.0049 |
|        2 |        1247 |     0      |           0.0058 |
|        3 |        1739 |     0.0075 |           0.0081 |
|        4 |        2132 |     0.0117 |           0.01   |
|        5 |        3395 |     0.0103 |           0.0159 |
|        6 |        5775 |     0.014  |           0.027  |
|        7 |        9047 |     0.0169 |           0.0423 |
|        8 |       14863 |     0.0227 |           0.0695 |
|        9 |       32238 |     0.0272 |           0.1508 |
|       10 |      132969 |     0.074  |           0.6221 |

## Diversity

```json
{
  "distinct_items_recommended": 13969,
  "catalogue_coverage": 0.1324,
  "category_entropy_bits": 1.851,
  "max_category_entropy_bits": 3.7,
  "most_recommended_item_share": 0.0517
}
```
