"""THE feature builder. Offline training and online serving both assemble features here.

The two callers cannot share raw inputs: offline has the 31M-row transaction log, online has
precomputed Parquet tables and a millisecond budget. So the module separates the two things
that can skew independently:

  sources    WHERE the four feature blocks come from. Two implementations, by necessity.
  assemble   HOW they become a feature row: the joins, the derived columns, the null policy,
             the column order, the categorical dtypes. Exactly one implementation.

Skew almost never comes from the aggregation itself — it comes from a join key, a fillna, or
a column order that drifted on one side only. Those all live in `assemble`, which is shared.
The `sources` implementations are then verified against each other by tests/test_parity.py.

Every offline query reads only `t_dat < as_of`, inherited from sql/features/*.sql.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import pandas as pd

from src.config import FEATURE_LOOKBACK_DAYS
from src.data.db import load_sql

CATEGORICAL = [
    "product_group_name",
    "index_group_name",
    "garment_group_name",
    "club_member_status",
    "fashion_news_frequency",
]

NON_FEATURES = ["customer_key", "article_id", "label"]

CUSTOMER_GROUP_SQL = """
SELECT t.customer_key, a.product_group_name,
       count(*) AS cg_n_purchases,
       count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY t.customer_key) AS cg_share,
       DATE_DIFF('day', max(t.t_dat), CAST($as_of AS DATE)) AS cg_days_since_last
FROM transactions t
JOIN eval_customers e USING (customer_key)
JOIN articles a ON a.article_id = t.article_id
WHERE t.t_dat < CAST($as_of AS DATE)
  AND t.t_dat >= CAST($as_of AS DATE) - CAST($lookback AS INTEGER) * INTERVAL 1 DAY
GROUP BY 1, 2
"""

# Same aggregations but over every customer, for the serving snapshot. Offline the blocks are
# restricted to `eval_customers`; without that restriction they are computed for all 1.4M
# customers on every chunk, which is both wasted work and enough memory to be OOM-killed.
CUSTOMER_GROUP_ALL_SQL = CUSTOMER_GROUP_SQL.replace(
    "JOIN eval_customers e USING (customer_key)\n", ""
)


def all_customers_sql(name: str) -> str:
    """A feature query with its `eval_customers` restriction removed."""
    return load_sql(name).replace("JOIN eval_customers e USING (customer_key)\n", "")


# Retrieval columns for strategies that did not propose a candidate. Offline and online must
# encode absence the same way: 9999 keeps rank monotone (smaller = better) rather than making
# "not retrieved" look like the best possible rank, and 0.0 is a genuine absence of score.
MISSING_RANK = 9999
MISSING_SCORE = 0.0


@dataclass
class FeatureSources:
    """The four feature blocks, however they were obtained.

    Each field is either a DataFrame or the name of a table already present on the
    connection. Serving passes table names so that a 42k-row article block is joined inside
    DuckDB instead of being marshalled into pandas on every request.
    """

    customer: pd.DataFrame | str  # one row per customer
    article: pd.DataFrame | str  # one row per article
    affinity: pd.DataFrame | str  # one row per (customer, article)
    category: pd.DataFrame | str  # one row per (customer, product_group_name)
    product_code: pd.DataFrame | str  # one row per (customer, product_code)


def offline_sources(
    con: duckdb.DuckDBPyConnection, as_of: date, lookback_days: int = FEATURE_LOOKBACK_DAYS
) -> FeatureSources:
    """Compute the blocks from the transaction log. Requires the `eval_customers` temp table."""
    return FeatureSources(
        customer=con.execute(load_sql("customer_recency"), [as_of, as_of, lookback_days]).df(),
        article=con.execute(load_sql("article_popularity"), [as_of, lookback_days]).df(),
        affinity=con.execute(
            load_sql("customer_article_affinity"), [as_of, as_of, lookback_days]
        ).df(),
        category=con.execute(
            CUSTOMER_GROUP_SQL, {"as_of": as_of, "lookback": lookback_days}
        ).df(),
        product_code=con.execute(
            load_sql("customer_productcode_affinity"), [as_of, as_of, lookback_days]
        ).df(),
    )


def serving_sources(con: duckdb.DuckDBPyConnection, customer_key: int) -> FeatureSources:
    """Point at the precomputed serving tables. No data is copied into Python here.

    The customer-keyed blocks are filtered by a view rather than materialised, so DuckDB can
    use the customer_key index; the article block is shared across all requests and needs no
    filtering at all.
    """
    for name, table in (
        ("cust_f", "customer_features"),
        ("aff_f", "affinity"),
        ("cgrp_f", "customer_category"),
        ("cpc_f", "customer_productcode"),
    ):
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW {name}_v AS "
            f"SELECT * FROM {table} WHERE customer_key = {int(customer_key)}"
        )
    return FeatureSources(
        customer="cust_f_v",
        article="article_features",
        affinity="aff_f_v",
        category="cgrp_f_v",
        product_code="cpc_f_v",
    )


ASSEMBLE_SQL = """
SELECT c.*,
       cu.n_purchases, cu.n_unique_items, cu.n_active_days,
       cu.avg_price AS cust_avg_price, cu.max_price AS cust_max_price,
       cu.std_price AS cust_std_price,
       cu.days_since_last, cu.days_since_first,
       cu.n_purchases_7d, cu.n_purchases_30d, cu.online_ratio,
       cm.age, cm.fn, cm.active, cm.club_member_status, cm.fashion_news_frequency,
       ar.n_sold, ar.n_buyers,
       ar.avg_price AS art_avg_price, ar.max_discount,
       ar.n_sold_1d, ar.n_sold_7d, ar.n_sold_30d,
       ar.days_since_first_sale, ar.days_since_last_sale, ar.trend_ratio,
       af.ca_n_purchases, af.ca_days_since_last, af.ca_share_of_purchases,
       cg.cg_n_purchases, cg.cg_share, cg.cg_days_since_last,
       ar.avg_price / NULLIF(cu.avg_price, 0) AS price_ratio,
       ar.price_pctile,
       -- How unusual is this item's price for this customer, in their own spread? A ratio
       -- alone cannot say whether 1.5x is normal for a volatile shopper or extreme for a
       -- consistent one.
       (ar.avg_price - cu.avg_price) / NULLIF(cu.std_price, 0) AS price_z,
       cpc.cpc_n_purchases, cpc.cpc_n_variants, cpc.cpc_days_since_last, cpc.cpc_share,
       a.product_group_name, a.index_group_name, a.garment_group_name,
       a.product_type_no, a.section_no, a.department_no,
       a.perceived_colour_value_id, a.graphical_appearance_no
FROM cand_f c
LEFT JOIN {cust_f} cu ON cu.customer_key = c.customer_key
LEFT JOIN customers cm ON cm.customer_key = c.customer_key
LEFT JOIN {art_f} ar ON ar.article_id = c.article_id
LEFT JOIN {aff_f} af ON af.customer_key = c.customer_key AND af.article_id = c.article_id
LEFT JOIN articles a ON a.article_id = c.article_id
LEFT JOIN {cgrp_f} cg ON cg.customer_key = c.customer_key
                     AND cg.product_group_name = a.product_group_name
LEFT JOIN {cpc_f} cpc ON cpc.customer_key = c.customer_key
                     AND cpc.product_code = a.product_code
ORDER BY c.customer_key, c.article_id
"""

# "No rows matched" genuinely means zero for counts and shares. Everything else stays NaN,
# which LightGBM handles natively and which carries the information that we know nothing.
FILL_ZERO = [
    "ca_n_purchases", "cg_n_purchases", "cg_share", "ca_share_of_purchases",
    "cpc_n_purchases", "cpc_n_variants", "cpc_share",
]


def assemble(
    con: duckdb.DuckDBPyConnection, candidates: pd.DataFrame, sources: FeatureSources
) -> pd.DataFrame:
    """Join the blocks onto candidates. The one code path that defines a feature row."""
    registered = []
    con.register("cand_f", candidates)
    registered.append("cand_f")

    names = {}
    for alias, block in (
        ("cust_f", sources.customer),
        ("art_f", sources.article),
        ("aff_f", sources.affinity),
        ("cgrp_f", sources.category),
        ("cpc_f", sources.product_code),
    ):
        if isinstance(block, str):
            names[alias] = block
        else:
            con.register(alias, block)
            registered.append(alias)
            names[alias] = alias

    sql = ASSEMBLE_SQL.format(**names)
    try:
        out = con.execute(sql).df()
    finally:
        for name in registered:
            con.unregister(name)

    for col in FILL_ZERO:
        out[col] = out[col].fillna(0.0)
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    return out


def build_features(
    con: duckdb.DuckDBPyConnection,
    candidates: pd.DataFrame,
    as_of: date,
    lookback_days: int = FEATURE_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Offline entry point: compute sources from the transaction log, then assemble."""
    return assemble(con, candidates, offline_sources(con, as_of, lookback_days))


def build_features_serving(
    con: duckdb.DuckDBPyConnection, candidates: pd.DataFrame, customer_key: int
) -> pd.DataFrame:
    """Online entry point: read precomputed sources, then assemble through the same path."""
    return assemble(con, candidates, serving_sources(con, customer_key))


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature column names, in a stable order. Offline and online must agree exactly."""
    return [c for c in df.columns if c not in NON_FEATURES]
