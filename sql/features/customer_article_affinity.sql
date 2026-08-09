-- (customer, article) interaction features — the strongest block in the ranker, because
-- fashion repurchase is a dominant signal.
--
-- Parameters (positional, in order):
--   $1 as_of_date   exclusive upper bound
--   $2 as_of_date   again, for the DATE_DIFF anchor
--   $3 lookback_days
SELECT
    t.customer_key,
    t.article_id,
    COUNT(*)                                            AS ca_n_purchases,
    DATE_DIFF('day', MAX(t.t_dat), CAST($2 AS DATE))    AS ca_days_since_last,
    MIN(t.price)                                        AS ca_min_price_paid,
    -- how much of this customer's spend went to this item's product group
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY t.customer_key) AS ca_share_of_purchases
FROM transactions t
WHERE t.t_dat < CAST($1 AS DATE)
  AND t.t_dat >= CAST($1 AS DATE) - CAST($3 AS INTEGER) * INTERVAL 1 DAY
GROUP BY t.customer_key, t.article_id
