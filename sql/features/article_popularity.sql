-- Item-side popularity and trend features.
--
-- Parameters (positional, in order):
--   $1 as_of_date   exclusive upper bound
--   $2 lookback_days
--
-- trend_ratio is deliberately smoothed: a brand-new item with 3 sales in the last week and
-- 0 before it would otherwise divide by zero and dominate the ranker.
SELECT
    article_id,
    COUNT(*)                                                            AS n_sold,
    COUNT(DISTINCT customer_key)                                        AS n_buyers,
    AVG(price)                                                          AS avg_price,
    MIN(price)                                                          AS min_price,
    MAX(price)                                                          AS max_price,
    1.0 - MIN(price) / NULLIF(MAX(price), 0)                            AS max_discount,
    COUNT(*) FILTER (WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 1 DAY)  AS n_sold_1d,
    COUNT(*) FILTER (WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 7 DAY)  AS n_sold_7d,
    COUNT(*) FILTER (WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 30 DAY) AS n_sold_30d,
    DATE_DIFF('day', MIN(t_dat), CAST($1 AS DATE))                      AS days_since_first_sale,
    DATE_DIFF('day', MAX(t_dat), CAST($1 AS DATE))                      AS days_since_last_sale,
    PERCENT_RANK() OVER (ORDER BY AVG(price))                           AS price_pctile,
    (COUNT(*) FILTER (WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 7 DAY) + 1.0)
        / (COUNT(*) FILTER (
              WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 14 DAY
                AND t_dat <  CAST($1 AS DATE) - INTERVAL 7 DAY
          ) + 1.0)                                                      AS trend_ratio
FROM transactions
WHERE t_dat < CAST($1 AS DATE)
  AND t_dat >= CAST($1 AS DATE) - CAST($2 AS INTEGER) * INTERVAL 1 DAY
GROUP BY article_id
