"""End-to-end smoke test of the ingest + query path on a tiny synthetic dataset.

Runs without the Kaggle download, so CI can catch a broken SQL contract in ~1 second.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from src.data import ingest
from src.data.db import load_sql
from src.eval.split import eval_customers, ground_truth

CUSTOMERS = [
    "0" * 63 + "1",
    "0" * 63 + "2",
    "f" * 64,
]

TRANSACTIONS = [
    # (t_dat, customer index, article_id, price, channel)
    ("2020-09-01", 0, 100, 0.05, 2),
    ("2020-09-02", 0, 100, 0.04, 2),  # repurchase of the same article
    ("2020-09-03", 0, 200, 0.10, 1),
    ("2020-09-10", 1, 100, 0.05, 2),
    ("2020-09-17", 0, 300, 0.20, 2),  # inside the val week
    ("2020-09-18", 2, 100, 0.06, 1),  # inside the val week
]


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("processed")

    (raw / "customers.csv").write_text(
        "customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code\n"
        + "".join(f"{c},1.0,1.0,ACTIVE,Regularly,{25 + i},abc\n" for i, c in enumerate(CUSTOMERS))
    )
    (raw / "articles.csv").write_text(
        "article_id,product_type_name,product_group_name,index_group_name\n"
        "100,Trousers,Garment Lower body,Ladieswear\n"
        "200,Sweater,Garment Upper body,Ladieswear\n"
        "300,Dress,Garment Full body,Ladieswear\n"
    )
    (raw / "transactions_train.csv").write_text(
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        + "".join(f"{d},{CUSTOMERS[c]},{a},{p},{ch}\n" for d, c, a, p, ch in TRANSACTIONS)
    )

    con = duckdb.connect()
    ingest.convert_articles(con, raw, out)
    ingest.convert_customers(con, raw, out)
    ingest.convert_transactions(con, raw, out)
    ingest.verify(con, out)
    con.close()
    return out


@pytest.fixture
def con(processed):
    from src.data.db import connect

    c = connect(processed)
    yield c
    c.close()


def test_row_count_and_types(con):
    rows = con.execute("SELECT count(*) FROM transactions").fetchone()[0]
    assert rows == len(TRANSACTIONS)

    types = dict(
        con.execute(
            "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM transactions)"
        ).fetchall()
    )
    assert types["article_id"] == "INTEGER"
    assert types["customer_key"] == "BIGINT"
    assert types["price"] == "FLOAT"
    assert types["t_dat"] == "DATE"


def test_customer_key_is_stable_and_distinct(con):
    keys = con.execute(
        "SELECT count(DISTINCT customer_key) FROM customers"
    ).fetchone()[0]
    assert keys == len(CUSTOMERS)


def test_ground_truth_covers_only_the_target_week(con):
    gt = ground_truth(con, date(2020, 9, 16)).set_index("customer_key")["articles"]
    assert len(gt) == 2  # customers 0 and 2 bought in the val week; customer 1 did not
    assert sorted(eval_customers(con, date(2020, 9, 16))) == sorted(gt.index.tolist())
    all_articles = {a for arts in gt for a in arts}
    assert all_articles == {300, 100}


def test_customer_features_respect_as_of_date(con):
    """The 2020-09-17 purchase must be invisible to features built as of 2020-09-16."""
    sql = load_sql("customer_recency")
    as_of = date(2020, 9, 16)
    df = con.execute(sql, [as_of, as_of, 90]).df().set_index("customer_key")

    key0 = con.execute("SELECT customer_key FROM customers ORDER BY age LIMIT 1").fetchone()[0]
    row = df.loc[key0]
    assert row["n_purchases"] == 3  # the 09-17 row is excluded
    assert row["n_unique_items"] == 2
    assert row["days_since_last"] == 13  # 09-03 -> 09-16


def test_article_features_respect_as_of_date(con):
    sql = load_sql("article_popularity")
    df = con.execute(sql, [date(2020, 9, 16), 90]).df().set_index("article_id")
    assert 300 not in df.index  # article 300 only sells inside the val week
    assert df.loc[100, "n_sold"] == 3
    assert df.loc[100, "n_sold_7d"] == 1  # only the 09-10 purchase is within 7 days


def test_affinity_features_respect_as_of_date(con):
    sql = load_sql("customer_article_affinity")
    as_of = date(2020, 9, 16)
    df = con.execute(sql, [as_of, as_of, 90]).df()
    assert set(df["article_id"]) == {100, 200}
    repurchase = df[df["article_id"] == 100]["ca_n_purchases"].max()
    assert repurchase == 2
