"""DuckDB connection factory. Every query in the project goes through `connect()`.

Views (`transactions`, `articles`, `customers`) are registered over the Parquet files so
that SQL in sql/features/ is portable between batch jobs, tests, and the serving process.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.config import DATA_PROCESSED, SQL


def connect(
    processed: Path = DATA_PROCESSED,
    read_only: bool = True,
    memory_limit: str = "6GB",
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB with views over the processed Parquet files.

    `memory_limit` is not a nicety. Without it, the candidate pivot in
    `src.recall.pipeline.build_candidates` grows unbounded — measured at 4.5 GB for a single
    20k-customer chunk — and the process is OOM-killed silently rather than raising. With a
    limit DuckDB spills the join to disk and finishes slower instead of dying.
    """
    tx_glob = processed / "transactions" / "**" / "*.parquet"
    con = duckdb.connect(config={"memory_limit": memory_limit, "temp_directory": "/tmp/duckdb_spill"})
    con.execute(
        f"""
        CREATE VIEW transactions AS
            SELECT * FROM read_parquet('{tx_glob}', hive_partitioning=true);
        CREATE VIEW articles AS
            SELECT * FROM read_parquet('{processed / "articles.parquet"}');
        CREATE VIEW customers AS
            SELECT * FROM read_parquet('{processed / "customers.parquet"}');
        """
    )
    if not read_only:
        con.execute("SET threads TO 8")
    return con


def load_sql(name: str) -> str:
    """Read a named query from sql/features/, e.g. load_sql('customer_recency')."""
    path = SQL / "features" / f"{name}.sql"
    if not path.exists():
        raise FileNotFoundError(f"no such query: {path}")
    return path.read_text()


def register_customers(con: duckdb.DuckDBPyConnection, customers: list[int]) -> None:
    """Materialise a customer population as the temp table `eval_customers`.

    Every retrieval strategy and feature query joins against this table rather than taking a
    customer list as a parameter, which is what keeps their SQL identical between the offline
    pipeline and the serving snapshot build.
    """
    con.execute("CREATE OR REPLACE TEMP TABLE eval_customers (customer_key BIGINT)")
    con.executemany("INSERT INTO eval_customers VALUES (?)", [(c,) for c in customers])
