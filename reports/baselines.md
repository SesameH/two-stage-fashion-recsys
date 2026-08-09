# Baseline ladder

- Validation week: `2020-09-16` to `2020-09-23`
- Evaluation population: 68,984 customers with >=1 purchase in that week
- All features cut off strictly before `2020-09-16`

| baseline             |   MAP@12 |   recall@12 |   coverage |   sec |
|:---------------------|---------:|------------:|-----------:|------:|
| B0 bestsellers (7d)  |  0.00875 |     0.0255  |     1      |   0   |
| B1 repurchase (90d)  |  0.02211 |     0.03647 |     0.7323 |   0.3 |
| B2 repurchase + fill |  0.02557 |     0.05201 |     1      |   0.1 |

`coverage` is the share of evaluated customers the baseline returns any prediction for.
