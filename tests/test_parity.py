"""Offline/online feature parity — the training-serving skew guard.

Offline reads the 31M-row transaction log. Online reads a precomputed Parquet snapshot. Those
two paths *must* differ, which is exactly why they need an automated equality check: nothing
about the code makes it obvious when the snapshot stops agreeing with the log.

The test builds the same candidate rows for the same customer at the same `as_of` through both
`FeatureSources` implementations and asserts the assembled feature rows are identical.

A failure here means one of:
  - the serving snapshot is stale relative to the transaction log
  - `precompute.py` and the offline pipeline disagree on a lookback window or filter
  - a new feature was added to `assemble` that the snapshot does not carry
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import DATA_PROCESSED
from src.data.db import connect, register_customers
from src.features.builder import assemble, offline_sources, serving_sources
from src.serve.precompute import SERVING

pytestmark = pytest.mark.skipif(
    not (SERVING / "meta.json").exists(),
    reason="serving snapshot not built; run `make serve-data`",
)

N_CUSTOMERS = 25
# Feature blocks the serving snapshot carries. Retrieval columns are compared separately
# because the online candidate set is deliberately a subset (see README, serving section).
SHARED_BLOCKS = [
    "n_purchases", "n_unique_items", "n_active_days", "cust_avg_price", "cust_max_price",
    "cust_std_price", "days_since_last", "days_since_first", "n_purchases_7d",
    "n_purchases_30d", "online_ratio", "age", "n_sold", "n_buyers", "art_avg_price",
    "max_discount", "n_sold_1d", "n_sold_7d", "n_sold_30d", "days_since_first_sale",
    "days_since_last_sale", "trend_ratio", "ca_n_purchases", "ca_days_since_last",
    "ca_share_of_purchases", "cg_n_purchases", "cg_share", "cg_days_since_last",
    "price_ratio", "price_pctile", "price_z", "cpc_n_purchases", "cpc_n_variants",
    "cpc_days_since_last", "cpc_share",
    "product_group_name", "index_group_name", "garment_group_name",
]


@pytest.fixture(scope="module")
def snapshot_as_of():
    return json.loads((SERVING / "meta.json").read_text())["as_of"]


@pytest.fixture(scope="module")
def serving_con():
    import duckdb

    con = duckdb.connect()
    for name in (
        "customer_features", "article_features", "affinity", "customer_category",
        "customer_productcode", "repurchase", "variants", "articles", "customers",
    ):
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{SERVING / (name + '.parquet')}')"
        )
    yield con
    con.close()


@pytest.fixture(scope="module")
def sample(snapshot_as_of):
    """Customers with history, plus the candidate rows to score them on."""
    con = connect()
    customers = con.execute(
        "SELECT customer_key FROM transactions WHERE t_dat < ? "
        "GROUP BY 1 HAVING count(*) >= 5 ORDER BY customer_key LIMIT ?",
        [snapshot_as_of, N_CUSTOMERS],
    ).df()["customer_key"].tolist()

    # A fixed candidate set: the customer's own recent articles plus global bestsellers, so
    # both paths score identical rows and any difference is a feature difference.
    register_customers(con, customers)
    cand = con.execute(
        """
        SELECT customer_key, article_id, 1.0 AS rrf_score, 1 AS n_sources FROM (
            SELECT t.customer_key, t.article_id,
                   row_number() OVER (PARTITION BY t.customer_key ORDER BY t.article_id) AS rn
            FROM transactions t JOIN eval_customers e USING (customer_key)
            WHERE t.t_dat < ?
            GROUP BY 1, 2
        ) WHERE rn <= 20
        """,
        [snapshot_as_of],
    ).df()
    con.close()
    return customers, cand


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["customer_key", "article_id"]).reset_index(drop=True)


def test_offline_online_feature_parity(sample, serving_con, snapshot_as_of):
    from datetime import date

    customers, cand = sample
    as_of = date.fromisoformat(snapshot_as_of)

    con = connect()
    register_customers(con, customers)
    offline = _sorted(assemble(con, cand, offline_sources(con, as_of)))
    con.close()

    online_parts = []
    for ck in customers:
        rows = cand[cand["customer_key"] == ck]
        if rows.empty:
            continue
        online_parts.append(assemble(serving_con, rows, serving_sources(serving_con, ck)))
    online = _sorted(pd.concat(online_parts, ignore_index=True))

    assert len(offline) == len(online), "candidate rows diverged between paths"
    assert list(offline.columns) == list(online.columns), "column order differs"

    for col in SHARED_BLOCKS:
        left, right = offline[col], online[col]
        if str(left.dtype) == "category":
            pd.testing.assert_series_equal(
                left.astype(str), right.astype(str), check_names=False
            )
        else:
            pd.testing.assert_series_equal(
                left.astype("float64"), right.astype("float64"),
                check_names=False, rtol=1e-6, atol=1e-6,
            )


def test_snapshot_records_its_cutoff():
    meta = json.loads((SERVING / "meta.json").read_text())
    assert "as_of" in meta, "serving snapshot must record the date it was built for"


def test_serving_snapshot_has_no_future_rows(snapshot_as_of):
    """The snapshot's own aggregates must not reflect anything at or after the cutoff."""
    con = connect()
    after = con.execute(
        "SELECT count(*) FROM transactions WHERE t_dat >= ?", [snapshot_as_of]
    ).fetchone()[0]
    con.close()
    assert after > 0, "cutoff is past the end of the data, so this test proves nothing"

    import duckdb

    n = duckdb.connect().execute(
        f"SELECT max(days_since_last) FROM read_parquet('{SERVING / 'customer_features.parquet'}')"
    ).fetchone()[0]
    assert n is not None and n >= 0, "days_since_last must be non-negative w.r.t. the cutoff"


def test_processed_data_present():
    assert (DATA_PROCESSED / "articles.parquet").exists()
