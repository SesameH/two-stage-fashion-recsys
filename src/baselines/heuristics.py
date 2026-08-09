"""Rule-based baselines B0-B2.

Every function takes `as_of` and reads only `t_dat < as_of`. B2 is the baseline the
two-stage model has to beat; repurchase is an extremely strong signal in fashion.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb

from src.config import K


def top_articles(
    con: duckdb.DuckDBPyConnection, as_of: date, days: int = 7, k: int = K
) -> list[int]:
    """B0: global bestsellers over the `days` before `as_of`."""
    rows = con.execute(
        """
        SELECT article_id
        FROM transactions
        WHERE t_dat < CAST(? AS DATE) AND t_dat >= CAST(? AS DATE) - CAST(? AS INTEGER) * INTERVAL 1 DAY
        GROUP BY article_id
        ORDER BY count(*) DESC, article_id
        LIMIT ?
        """,
        [as_of, as_of, days, k],
    ).fetchall()
    return [r[0] for r in rows]


def top_articles_by_age_bucket(
    con: duckdb.DuckDBPyConnection, as_of: date, days: int = 7, k: int = K
) -> dict[int, list[int]]:
    """Bestsellers within 10-year age buckets. Used as a cold-start filler and by R2."""
    rows = con.execute(
        """
        WITH tx AS (
            SELECT t.article_id,
                   CAST(floor(coalesce(c.age, 30) / 10) AS INTEGER) AS age_bucket
            FROM transactions t
            JOIN customers c USING (customer_key)
            WHERE t.t_dat < CAST(? AS DATE)
              AND t.t_dat >= CAST(? AS DATE) - CAST(? AS INTEGER) * INTERVAL 1 DAY
        ),
        ranked AS (
            SELECT age_bucket, article_id,
                   row_number() OVER (PARTITION BY age_bucket ORDER BY count(*) DESC, article_id) AS rn
            FROM tx GROUP BY age_bucket, article_id
        )
        SELECT age_bucket, article_id FROM ranked WHERE rn <= ? ORDER BY age_bucket, rn
        """,
        [as_of, as_of, days, k],
    ).fetchall()

    out: dict[int, list[int]] = {}
    for bucket, article in rows:
        out.setdefault(bucket, []).append(article)
    return out


def repurchase(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    lookback_days: int = 90,
    k: int = K,
) -> dict[int, list[int]]:
    """B1: the customer's own recent articles, most recently bought first.

    Ties broken by purchase count then article id so the output is deterministic.
    Requires `register_customers` to have been called.
    """
    rows = con.execute(
        """
        WITH hist AS (
            SELECT t.customer_key, t.article_id,
                   max(t.t_dat) AS last_dat,
                   count(*)     AS n
            FROM transactions t
            JOIN eval_customers e USING (customer_key)
            WHERE t.t_dat < CAST(? AS DATE)
              AND t.t_dat >= CAST(? AS DATE) - CAST(? AS INTEGER) * INTERVAL 1 DAY
            GROUP BY t.customer_key, t.article_id
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY customer_key ORDER BY last_dat DESC, n DESC, article_id
            ) AS rn
            FROM hist
        )
        SELECT customer_key, article_id FROM ranked WHERE rn <= ? ORDER BY customer_key, rn
        """,
        [as_of, as_of, lookback_days, k],
    ).fetchall()

    out: dict[int, list[int]] = {}
    for customer, article in rows:
        out.setdefault(customer, []).append(article)
    return out


def pad(
    preds: dict[int, list[int]],
    customers: list[int],
    filler: list[int],
    k: int = K,
) -> dict[int, list[int]]:
    """B2: top up every customer to k slots with `filler`, skipping duplicates.

    Customers absent from `preds` (no purchase history in the window) get pure filler,
    which is what makes B2 a genuine cold-start-safe baseline rather than B1's partial one.
    """
    out = {}
    for c in customers:
        items = list(preds.get(c, []))[:k]
        if len(items) < k:
            seen = set(items)
            items.extend(f for f in filler if f not in seen)
        out[c] = items[:k]
    return out


def last_week(as_of: date) -> date:
    return as_of - timedelta(days=7)
