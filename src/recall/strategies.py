"""Retrieval strategies R1-R5.

All of them read only `t_dat < as_of` and all of them expect the temp table `eval_customers`
to already exist (see `src.baselines.heuristics.register_customers`).

They return the long format documented in `src.recall.base`.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from src.recall.base import normalize

# --- R1: repurchase ---------------------------------------------------------

R1_SQL = """
WITH hist AS (
    SELECT t.customer_key, t.article_id,
           max(t.t_dat) AS last_dat,
           count(*)     AS n
    FROM transactions t
    JOIN eval_customers e USING (customer_key)
    WHERE t.t_dat < CAST($as_of AS DATE)
      AND t.t_dat >= CAST($as_of AS DATE) - CAST($lookback AS INTEGER) * INTERVAL 1 DAY
    GROUP BY 1, 2
),
ranked AS (
    SELECT customer_key, article_id,
           n AS score,
           row_number() OVER (
               PARTITION BY customer_key ORDER BY last_dat DESC, n DESC, article_id
           ) AS rank
    FROM hist
)
SELECT customer_key, article_id, rank, CAST(score AS DOUBLE) AS score
FROM ranked WHERE rank <= $n
"""


def r1_repurchase(
    con: duckdb.DuckDBPyConnection, as_of: date, n: int = 30, lookback_days: int = 90
) -> pd.DataFrame:
    """Articles the customer has already bought, most recent first."""
    df = con.execute(R1_SQL, {"as_of": as_of, "lookback": lookback_days, "n": n}).df()
    return normalize(df, "r1_repurchase")


# --- R2: popularity ---------------------------------------------------------

# The popularity *tables* are customer-independent, so they are also exactly what the serving
# layer precomputes. Defining them once here means the online ranking of bestsellers cannot
# drift from the offline one.

AGE_BUCKET = "CAST(floor(coalesce(age, 30) / 10) AS INTEGER)"

AGE_POPULARITY_SQL = f"""
SELECT age_bucket, article_id, score, rank FROM (
    SELECT {AGE_BUCKET.replace('age', 'c.age')} AS age_bucket,
           t.article_id,
           CAST(count(*) AS DOUBLE) AS score,
           row_number() OVER (
               PARTITION BY {AGE_BUCKET.replace('age', 'c.age')}
               ORDER BY count(*) DESC, t.article_id
           ) AS rank
    FROM transactions t
    JOIN customers c USING (customer_key)
    WHERE t.t_dat < CAST($as_of AS DATE)
      AND t.t_dat >= CAST($as_of AS DATE) - CAST($days AS INTEGER) * INTERVAL 1 DAY
    GROUP BY 1, 2
) WHERE rank <= $n
"""

GLOBAL_POPULARITY_SQL = """
SELECT article_id, score, rank FROM (
    SELECT article_id, CAST(count(*) AS DOUBLE) AS score,
           row_number() OVER (ORDER BY count(*) DESC, article_id) AS rank
    FROM transactions
    WHERE t_dat < CAST($as_of AS DATE)
      AND t_dat >= CAST($as_of AS DATE) - CAST($days AS INTEGER) * INTERVAL 1 DAY
    GROUP BY 1
) WHERE rank <= $n
"""


def age_popularity_table(
    con: duckdb.DuckDBPyConnection, as_of: date, n: int = 200, days: int = 7
) -> pd.DataFrame:
    """Bestsellers per 10-year age bucket. Customer-independent."""
    return con.execute(AGE_POPULARITY_SQL, {"as_of": as_of, "days": days, "n": n}).df()


def global_popularity_table(
    con: duckdb.DuckDBPyConnection, as_of: date, n: int = 400, days: int = 7
) -> pd.DataFrame:
    """Global bestsellers. Customer-independent."""
    return con.execute(GLOBAL_POPULARITY_SQL, {"as_of": as_of, "days": days, "n": n}).df()


def r2_popularity(
    con: duckdb.DuckDBPyConnection, as_of: date, n: int = 30, days: int = 7
) -> pd.DataFrame:
    """Bestsellers within the customer's 10-year age bucket. Carries the cold-start load."""
    table = age_popularity_table(con, as_of, n, days)
    con.register("age_pop_tmp", table)
    try:
        df = con.execute(
            f"""
            SELECT e.customer_key, p.article_id, p.rank, p.score
            FROM eval_customers e
            JOIN customers c USING (customer_key)
            JOIN age_pop_tmp p ON p.age_bucket = {AGE_BUCKET.replace('age', 'c.age')}
            """
        ).df()
    finally:
        con.unregister("age_pop_tmp")
    return normalize(df, "r2_popularity")


def r2b_global_popularity(
    con: duckdb.DuckDBPyConnection, as_of: date, n: int = 400, days: int = 7
) -> pd.DataFrame:
    """Deep global bestsellers, identical for every customer.

    Not a personalisation strategy at all, and it is the single highest-recall source in the
    stack: 93.7% of validation-week purchases are of articles inside the trailing-7-day
    top-20000, so depth beats cleverness at the candidate-generation stage. Personalisation
    still adds on top (see reports/recall.md) — but as re-ranking signal, not as reach.
    """
    con.register("glob_pop_tmp", global_popularity_table(con, as_of, n, days))
    try:
        df = con.execute(
            "SELECT e.customer_key, t.article_id, t.rank, t.score "
            "FROM eval_customers e CROSS JOIN glob_pop_tmp t"
        ).df()
    finally:
        con.unregister("glob_pop_tmp")
    return normalize(df, "r2b_global")


# --- R3: item-kNN over co-purchase ------------------------------------------

R3_SQL = """
WITH basket AS (
    SELECT customer_key, t_dat, article_id
    FROM transactions
    WHERE t_dat < CAST($as_of AS DATE)
      AND t_dat >= CAST($as_of AS DATE) - CAST($lookback AS INTEGER) * INTERVAL 1 DAY
),
item_pop AS (
    SELECT article_id, count(*) AS n FROM basket GROUP BY 1
),
pairs AS (
    -- Same customer, same day = one basket. a.article_id < b.article_id gives each unordered
    -- pair once; the neighbour table below re-expands it into both directions.
    SELECT a.article_id AS i, b.article_id AS j, count(*) AS co
    FROM basket a
    JOIN basket b
      ON a.customer_key = b.customer_key
     AND a.t_dat = b.t_dat
     AND a.article_id < b.article_id
    GROUP BY 1, 2
    HAVING count(*) >= $min_co
),
sim AS (
    SELECT p.i, p.j, p.co / sqrt(CAST(pi.n AS DOUBLE) * pj.n) AS s
    FROM pairs p JOIN item_pop pi ON pi.article_id = p.i
                 JOIN item_pop pj ON pj.article_id = p.j
),
neighbours AS (
    SELECT i AS item, j AS neighbour, s FROM sim
    UNION ALL
    SELECT j AS item, i AS neighbour, s FROM sim
),
top_neighbours AS (
    SELECT item, neighbour, s FROM (
        SELECT *, row_number() OVER (PARTITION BY item ORDER BY s DESC, neighbour) AS rn
        FROM neighbours
    ) WHERE rn <= $n_neighbours
),
seeds AS (
    -- The customer's most recent distinct articles act as query items.
    SELECT customer_key, article_id, rn AS seed_rank FROM (
        SELECT t.customer_key, t.article_id,
               row_number() OVER (
                   PARTITION BY t.customer_key ORDER BY max(t.t_dat) DESC, t.article_id
               ) AS rn
        FROM transactions t
        JOIN eval_customers e USING (customer_key)
        WHERE t.t_dat < CAST($as_of AS DATE)
          AND t.t_dat >= CAST($as_of AS DATE) - CAST($seed_lookback AS INTEGER) * INTERVAL 1 DAY
        GROUP BY t.customer_key, t.article_id
    ) WHERE rn <= $n_seeds
),
scored AS (
    -- Decay by seed recency so a purchase from last week outweighs one from two months ago.
    SELECT s.customer_key, tn.neighbour AS article_id,
           sum(tn.s / s.seed_rank) AS score
    FROM seeds s
    JOIN top_neighbours tn ON tn.item = s.article_id
    GROUP BY 1, 2
)
SELECT customer_key, article_id, rank, score FROM (
    SELECT *, row_number() OVER (PARTITION BY customer_key ORDER BY score DESC, article_id) AS rank
    FROM scored
) WHERE rank <= $n
"""


def r3_item_knn(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    n: int = 50,
    lookback_days: int = 90,
    seed_lookback_days: int = 90,
    n_seeds: int = 20,
    n_neighbours: int = 40,
    min_co: int = 3,
) -> pd.DataFrame:
    """Co-purchase item-kNN with cosine normalisation.

    `min_co` matters more than it looks: without it the pair table is dominated by
    once-observed coincidences, which are both noise and the bulk of the memory cost.
    """
    df = con.execute(
        R3_SQL,
        {
            "as_of": as_of,
            "lookback": lookback_days,
            "seed_lookback": seed_lookback_days,
            "n_seeds": n_seeds,
            "n_neighbours": n_neighbours,
            "min_co": min_co,
            "n": n,
        },
    ).df()
    return normalize(df, "r3_item_knn")


# --- R4: ALS ----------------------------------------------------------------


def r4_als(model, customers: list[int], n: int = 50) -> pd.DataFrame:
    """Matrix-factorisation candidates. `model` comes from `src.baselines.als.fit`."""
    preds = model.recommend(customers, k=n)
    rows = [
        (c, a, i + 1, float(n - i))
        for c, articles in preds.items()
        for i, a in enumerate(articles)
    ]
    df = pd.DataFrame(rows, columns=["customer_key", "article_id", "rank", "score"])
    return normalize(df, "r4_als")


# --- R5: bestsellers inside the customer's preferred categories --------------

R5_SQL = """
WITH prefs AS (
    -- The customer's top product groups by purchase count.
    SELECT customer_key, product_group_name, n FROM (
        SELECT t.customer_key, a.product_group_name, count(*) AS n,
               row_number() OVER (
                   PARTITION BY t.customer_key ORDER BY count(*) DESC, a.product_group_name
               ) AS rn
        FROM transactions t
        JOIN eval_customers e USING (customer_key)
        JOIN articles a ON a.article_id = t.article_id
        WHERE t.t_dat < CAST($as_of AS DATE)
          AND t.t_dat >= CAST($as_of AS DATE) - CAST($lookback AS INTEGER) * INTERVAL 1 DAY
        GROUP BY 1, 2
    ) WHERE rn <= $n_groups
),
group_top AS (
    SELECT product_group_name, article_id, n, rn FROM (
        SELECT a.product_group_name, t.article_id, count(*) AS n,
               row_number() OVER (
                   PARTITION BY a.product_group_name ORDER BY count(*) DESC, t.article_id
               ) AS rn
        FROM transactions t
        JOIN articles a ON a.article_id = t.article_id
        WHERE t.t_dat < CAST($as_of AS DATE)
          AND t.t_dat >= CAST($as_of AS DATE) - CAST($days AS INTEGER) * INTERVAL 1 DAY
        GROUP BY 1, 2
    ) WHERE rn <= $n_per_group
)
SELECT customer_key, article_id, rank, score FROM (
    SELECT p.customer_key, g.article_id,
           CAST(g.n AS DOUBLE) * p.n AS score,
           row_number() OVER (
               PARTITION BY p.customer_key ORDER BY CAST(g.n AS DOUBLE) * p.n DESC, g.article_id
           ) AS rank
    FROM prefs p JOIN group_top g USING (product_group_name)
) WHERE rank <= $n
"""


def r5_category(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    n: int = 30,
    lookback_days: int = 90,
    days: int = 7,
    n_groups: int = 3,
    n_per_group: int = 20,
) -> pd.DataFrame:
    """Recent bestsellers restricted to product groups the customer actually shops."""
    df = con.execute(
        R5_SQL,
        {
            "as_of": as_of,
            "lookback": lookback_days,
            "days": days,
            "n_groups": n_groups,
            "n_per_group": n_per_group,
            "n": n,
        },
    ).df()
    return normalize(df, "r5_category")


# --- R6: same product, different colour or size ------------------------------

R6_SQL = """
WITH owned AS (
    -- product_code is the first 7 digits of article_id: same garment, different
    -- colourway or size. Buying "the same thing again" usually means a *different*
    -- article_id, which is invisible to exact-article repurchase (R1).
    SELECT t.customer_key, a.product_code, max(t.t_dat) AS last_dat, count(*) AS n
    FROM transactions t
    JOIN eval_customers e USING (customer_key)
    JOIN articles a ON a.article_id = t.article_id
    WHERE t.t_dat < CAST($as_of AS DATE)
      AND t.t_dat >= CAST($as_of AS DATE) - CAST($lookback AS INTEGER) * INTERVAL 1 DAY
    GROUP BY 1, 2
),
variants AS (
    SELECT a.product_code, t.article_id, count(*) AS n_sold
    FROM transactions t
    JOIN articles a ON a.article_id = t.article_id
    WHERE t.t_dat < CAST($as_of AS DATE)
      AND t.t_dat >= CAST($as_of AS DATE) - CAST($days AS INTEGER) * INTERVAL 1 DAY
    GROUP BY 1, 2
)
SELECT customer_key, article_id, rank, score FROM (
    SELECT o.customer_key, v.article_id,
           CAST(v.n_sold AS DOUBLE) * o.n
             / (1 + DATE_DIFF('day', o.last_dat, CAST($as_of AS DATE))) AS score,
           row_number() OVER (
               PARTITION BY o.customer_key
               ORDER BY CAST(v.n_sold AS DOUBLE) * o.n
                        / (1 + DATE_DIFF('day', o.last_dat, CAST($as_of AS DATE))) DESC,
                        v.article_id
           ) AS rank
    FROM owned o JOIN variants v USING (product_code)
) WHERE rank <= $n
"""


def r6_product_variant(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    n: int = 50,
    lookback_days: int = 90,
    days: int = 14,
) -> pd.DataFrame:
    """Currently-selling variants of products the customer already owns.

    Ranked by variant sales volume weighted by how recently and how often the customer
    bought that product, so a garment bought last week outranks one bought in June.
    """
    df = con.execute(
        R6_SQL, {"as_of": as_of, "lookback": lookback_days, "days": days, "n": n}
    ).df()
    return normalize(df, "r6_product_variant")


def r3_last_knn(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    n: int = 100,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Item-kNN seeded by the customer's single most recent article.

    R3 aggregates neighbours over the customer's whole recent history, which averages away
    the strongest piece of context there is: what they bought last. Same similarity table,
    one seed, no recency decay needed because there is nothing to decay against.
    """
    df = con.execute(
        R3_SQL,
        {
            "as_of": as_of,
            "lookback": lookback_days,
            "seed_lookback": lookback_days,
            "n_seeds": 1,
            "n_neighbours": 60,
            "min_co": 3,
            "n": n,
        },
    ).df()
    return normalize(df, "r3_last_knn")
