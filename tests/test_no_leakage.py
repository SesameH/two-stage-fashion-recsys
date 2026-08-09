"""Structural proof that features cannot see the future.

The test builds features twice for the same `as_of`: once against the full transaction log,
once against a log physically truncated to `t_dat < as_of`. If any feature reads a row at or
after the cutoff, the two runs disagree.

This catches the failure mode that unit-testing individual SQL files does not: a new feature
added later that forgets its `t_dat <` predicate.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.config import DATA_PROCESSED, VAL_START
from src.data.db import connect, register_customers
from src.features.builder import build_features
from src.recall.pipeline import build_candidates

pytestmark = pytest.mark.skipif(
    not (DATA_PROCESSED / "articles.parquet").exists(),
    reason="processed data not built; run `make data`",
)

N_CUSTOMERS = 300


@pytest.fixture(scope="module")
def sample_customers():
    con = connect()
    rows = con.execute(
        "SELECT DISTINCT customer_key FROM transactions "
        "WHERE t_dat >= ? AND t_dat < ? + INTERVAL 7 DAY ORDER BY customer_key LIMIT ?",
        [VAL_START, VAL_START, N_CUSTOMERS],
    ).df()["customer_key"].tolist()
    con.close()
    return rows


def _features(cutoff: date | None, customers: list[int]) -> pd.DataFrame:
    con = connect()
    if cutoff is not None:
        # DuckDB cannot prepare a CREATE VIEW, so the cutoff is inlined. `cutoff` is a
        # datetime.date from our own config, never user input.
        con.execute("ALTER VIEW transactions RENAME TO transactions_all")
        con.execute(
            f"CREATE VIEW transactions AS "
            f"SELECT * FROM transactions_all WHERE t_dat < DATE '{cutoff.isoformat()}'"
        )
    register_customers(con, customers)
    cand = build_candidates(con, VAL_START, customers, n_candidates=50)
    feats = build_features(con, cand, VAL_START)
    con.close()
    return feats.sort_values(["customer_key", "article_id"]).reset_index(drop=True)


def test_features_identical_with_and_without_future_rows(sample_customers):
    full = _features(None, sample_customers)
    truncated = _features(VAL_START, sample_customers)

    assert list(full.columns) == list(truncated.columns)
    assert len(full) == len(truncated), "candidate sets diverged, so retrieval reads the future"
    pd.testing.assert_frame_equal(full, truncated, check_dtype=False, rtol=1e-6)


def test_ground_truth_week_is_actually_excluded(sample_customers):
    """Sanity check on the test itself: the validation week must contain rows to hide."""
    con = connect()
    n = con.execute(
        "SELECT count(*) FROM transactions WHERE t_dat >= ? AND t_dat < ? + INTERVAL 7 DAY",
        [VAL_START, VAL_START],
    ).fetchone()[0]
    con.close()
    assert n > 0, "no validation-week rows exist, so the leakage test proves nothing"
