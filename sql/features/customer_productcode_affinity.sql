-- (customer, product_code) affinity.
--
-- Parameters (positional, in order):
--   $1 as_of_date   exclusive upper bound
--   $2 as_of_date   again, for the DATE_DIFF anchor
--   $3 lookback_days
--
-- Why product_code rather than article_id: colour and size variants of one garment carry
-- different article_ids, so article-level affinity misses a customer rebuying the same shirt
-- in another colour. The R6 retrieval strategy built on this observation produced the single
-- strongest feature in the ranker; this is the same signal at feature granularity.
SELECT
    t.customer_key,
    a.product_code,
    COUNT(*)                                            AS cpc_n_purchases,
    COUNT(DISTINCT t.article_id)                        AS cpc_n_variants,
    DATE_DIFF('day', MAX(t.t_dat), CAST($2 AS DATE))    AS cpc_days_since_last,
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY t.customer_key) AS cpc_share
FROM transactions t
JOIN eval_customers e USING (customer_key)
JOIN articles a ON a.article_id = t.article_id
WHERE t.t_dat < CAST($1 AS DATE)
  AND t.t_dat >= CAST($1 AS DATE) - CAST($3 AS INTEGER) * INTERVAL 1 DAY
GROUP BY 1, 2
