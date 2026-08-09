-- Customer-side behavioural features.
--
-- Parameters (positional, in order):
--   $1 as_of_date   exclusive upper bound; nothing at or after this date may be read
--   $2 as_of_date   again, for the DATE_DIFF anchor
--   $3 lookback_days
--
-- The `t_dat < $1` predicate is the leakage guard for this entire file. Any change here
-- must be mirrored in tests/test_no_leakage.py.
SELECT
    customer_key,
    COUNT(*)                                          AS n_purchases,
    COUNT(DISTINCT article_id)                        AS n_unique_items,
    COUNT(DISTINCT t_dat)                             AS n_active_days,
    AVG(price)                                        AS avg_price,
    MAX(price)                                        AS max_price,
    STDDEV_POP(price)                                 AS std_price,
    MAX(t_dat)                                        AS last_purchase_date,
    DATE_DIFF('day', MAX(t_dat), CAST($2 AS DATE))    AS days_since_last,
    DATE_DIFF('day', MIN(t_dat), CAST($2 AS DATE))    AS days_since_first,
    COUNT(*) FILTER (WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 7 DAY)   AS n_purchases_7d,
    COUNT(*) FILTER (WHERE t_dat >= CAST($1 AS DATE) - INTERVAL 30 DAY)  AS n_purchases_30d,
    AVG(CAST(sales_channel_id AS FLOAT) - 1)          AS online_ratio
FROM transactions
WHERE t_dat < CAST($1 AS DATE)
  AND t_dat >= CAST($1 AS DATE) - CAST($3 AS INTEGER) * INTERVAL 1 DAY
GROUP BY customer_key
