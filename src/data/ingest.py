"""CSV -> Parquet conversion with type compression, driven entirely by DuckDB.

Run: python -m src.data.ingest

Type decisions:
  customer_id  64-char hex string -> int64 key (last 15 hex chars = 60 bits, always positive).
               15 rather than 16 chars because 16 hex digits overflow a signed BIGINT.
               60 bits over 1.4M ids makes a collision ~1e-6 likely, and `verify()` asserts
               the mapping is bijective anyway. The full string is kept in a side mapping
               table for the serving layer.
  article_id   int64 -> int32 (max observed ~9.6e8, fits comfortably)
  price        float64 -> float32
  t_dat        varchar -> DATE

Transactions are written Hive-partitioned by t_year/t_month rather than by raw t_dat:
734 daily partitions produce files far below the Parquet row-group size and slow every
scan down. Month partitions keep predicate pushdown on the time split while staying
sanely sized.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from src.config import DATA_PROCESSED, DATA_RAW

CUSTOMER_KEY_EXPR = "CAST('0x' || substr(customer_id, -15) AS BIGINT)"


def _log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


def convert_transactions(con: duckdb.DuckDBPyConnection, raw: Path, out: Path) -> None:
    src = raw / "transactions_train.csv"
    dst = out / "transactions"
    _log(f"transactions: {src} -> {dst}")
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(t_dat AS DATE)                       AS t_dat,
                {CUSTOMER_KEY_EXPR}                       AS customer_key,
                CAST(article_id AS INTEGER)               AS article_id,
                CAST(price AS FLOAT)                      AS price,
                CAST(sales_channel_id AS TINYINT)         AS sales_channel_id,
                CAST(year(CAST(t_dat AS DATE)) AS SMALLINT)  AS t_year,
                CAST(month(CAST(t_dat AS DATE)) AS TINYINT)  AS t_month
            FROM read_csv_auto('{src}', header=true)
        )
        TO '{dst}'
        (FORMAT PARQUET, PARTITION_BY (t_year, t_month),
         COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1)
        """
    )


def convert_customers(con: duckdb.DuckDBPyConnection, raw: Path, out: Path) -> None:
    src = raw / "customers.csv"
    _log(f"customers: {src}")
    con.execute(
        f"""
        COPY (
            SELECT
                {CUSTOMER_KEY_EXPR}                  AS customer_key,
                CAST(age AS FLOAT)                   AS age,
                CAST(FN AS FLOAT)                    AS fn,
                CAST(Active AS FLOAT)                AS active,
                club_member_status,
                fashion_news_frequency,
                postal_code
            FROM read_csv_auto('{src}', header=true)
        )
        TO '{out / "customers.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    # The serving layer receives raw customer_id strings; it needs the reverse lookup.
    con.execute(
        f"""
        COPY (
            SELECT customer_id, {CUSTOMER_KEY_EXPR} AS customer_key
            FROM read_csv_auto('{src}', header=true)
        )
        TO '{out / "customer_id_map.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def convert_articles(con: duckdb.DuckDBPyConnection, raw: Path, out: Path) -> None:
    src = raw / "articles.csv"
    _log(f"articles: {src}")
    con.execute(
        f"""
        COPY (
            SELECT * REPLACE (CAST(article_id AS INTEGER) AS article_id)
            FROM read_csv_auto('{src}', header=true)
        )
        TO '{out / "articles.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def verify(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    """Sanity checks that would otherwise surface as silently wrong metrics later."""
    tx = f"read_parquet('{out / 'transactions' / '**' / '*.parquet'}', hive_partitioning=true)"

    n_rows, n_cust, n_art, d_min, d_max = con.execute(
        f"SELECT count(*), count(DISTINCT customer_key), count(DISTINCT article_id),"
        f" min(t_dat), max(t_dat) FROM {tx}"
    ).fetchone()
    _log(f"rows={n_rows:,} customers={n_cust:,} articles={n_art:,} dates={d_min}..{d_max}")

    # A hash collision would silently merge two customers' histories.
    raw_cust, mapped_cust = con.execute(
        f"SELECT count(DISTINCT customer_id), count(DISTINCT customer_key)"
        f" FROM read_parquet('{out / 'customer_id_map.parquet'}')"
    ).fetchone()
    if raw_cust != mapped_cust:
        raise RuntimeError(
            f"customer_key collision: {raw_cust:,} ids collapsed to {mapped_cust:,} keys"
        )
    _log(f"customer_key bijective over {raw_cust:,} ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DATA_RAW)
    parser.add_argument("--out", type=Path, default=DATA_PROCESSED)
    args = parser.parse_args()

    missing = [
        f.name
        for f in (
            args.raw / "transactions_train.csv",
            args.raw / "customers.csv",
            args.raw / "articles.csv",
        )
        if not f.exists()
    ]
    if missing:
        raise SystemExit(
            f"missing raw files in {args.raw}: {', '.join(missing)}\n"
            "Download with: make download  (needs Kaggle API credentials)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA enable_progress_bar")

    t0 = time.perf_counter()
    convert_articles(con, args.raw, args.out)
    convert_customers(con, args.raw, args.out)
    convert_transactions(con, args.raw, args.out)
    verify(con, args.out)
    _log(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
