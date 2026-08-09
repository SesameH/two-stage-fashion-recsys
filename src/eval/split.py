"""Ground truth extraction for a target week.

Evaluation population: only customers with at least one purchase inside the target week.
Scoring every customer would flood MAP@12 with structural zeros and destroy its ability to
separate models — see README section "Why we score buyers only".
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from src.config import week_window


def ground_truth(con: duckdb.DuckDBPyConnection, as_of: date) -> pd.DataFrame:
    """Purchases inside [as_of, as_of+7d), one row per customer.

    Returns a DataFrame with columns `customer_key` and `articles` (list of int32 ids,
    ordered by first purchase time then item id for determinism).
    """
    start, end = week_window(as_of)
    return con.execute(
        """
        SELECT customer_key,
               list(DISTINCT article_id) AS articles
        FROM transactions
        WHERE t_dat >= CAST(? AS DATE) AND t_dat < CAST(? AS DATE)
        GROUP BY customer_key
        ORDER BY customer_key
        """,
        [start, end],
    ).df()


def eval_customers(con: duckdb.DuckDBPyConnection, as_of: date) -> list[int]:
    """The customer keys we score for the week starting at `as_of`."""
    start, end = week_window(as_of)
    return (
        con.execute(
            """
            SELECT DISTINCT customer_key
            FROM transactions
            WHERE t_dat >= CAST(? AS DATE) AND t_dat < CAST(? AS DATE)
            ORDER BY customer_key
            """,
            [start, end],
        )
        .df()["customer_key"]
        .tolist()
    )
