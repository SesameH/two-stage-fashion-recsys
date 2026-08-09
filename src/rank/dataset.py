"""Training-sample construction for the ranking stage.

Two rules decide whether the resulting numbers mean anything:

1. Candidates for the training week are generated with `as_of = TRAIN_START`. Generating
   them from the full history would leak the answer into the candidate set and inflate every
   offline metric.
2. Negatives are downsampled in TRAINING ONLY. Validation keeps every candidate, because
   MAP@12 on a downsampled candidate set is measuring a task that does not exist.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import NEG_SAMPLE_RATIO, week_window
from src.data.db import register_customers
from src.features.builder import CATEGORICAL, build_features
from src.recall.pipeline import build_candidates


def label_candidates(
    con: duckdb.DuckDBPyConnection, candidates: pd.DataFrame, as_of: date
) -> pd.DataFrame:
    """label = 1 if the customer bought that article during [as_of, as_of+7d)."""
    start, end = week_window(as_of)
    con.register("cand_l", candidates)
    out = con.execute(
        """
        WITH bought AS (
            SELECT DISTINCT customer_key, article_id
            FROM transactions
            WHERE t_dat >= CAST(? AS DATE) AND t_dat < CAST(? AS DATE)
        )
        SELECT c.*, CASE WHEN b.customer_key IS NULL THEN 0 ELSE 1 END AS label
        FROM cand_l c
        LEFT JOIN bought b USING (customer_key, article_id)
        """,
        [start, end],
    ).df()
    con.unregister("cand_l")
    return out


def downsample_negatives(
    df: pd.DataFrame, ratio: int = NEG_SAMPLE_RATIO, seed: int = 42
) -> pd.DataFrame:
    """Keep every positive, plus `ratio` negatives per positive, sampled within customer.

    Customers with no positive at all are dropped: a group with no relevant item contributes
    nothing to a LambdaRank gradient, and keeping them only inflates training time.

    `ratio` also sets the training group size (~ratio+1 candidates), and LambdaRank learns a
    ranking *within* a group. Serving ranks 300 candidates per customer, so a small ratio
    trains the model on a task narrower than the one it is deployed on.
    """
    rng = np.random.default_rng(seed)
    pos_per_customer = df.groupby("customer_key")["label"].transform("sum")
    df = df[pos_per_customer > 0]

    keep = df["label"] == 1
    n_neg_target = int(keep.sum() * ratio)
    neg_idx = df.index[~keep]
    if len(neg_idx) > n_neg_target:
        neg_idx = rng.choice(neg_idx, size=n_neg_target, replace=False)

    out = df.loc[keep.index[keep].union(pd.Index(neg_idx))]
    return out.sort_values(["customer_key", "article_id"], kind="stable")


def build_training_set(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    als_model=None,
    downsample: bool = True,
    customers: list[int] | None = None,
    chunk: int = 8000,
    neg_ratio: int = NEG_SAMPLE_RATIO,
) -> pd.DataFrame:
    """Candidates + labels + features for the week starting at `as_of`.

    `customers` defaults to everyone who bought during that week — the same population the
    metric is computed over.
    """
    if customers is None:
        start, end = week_window(as_of)
        customers = (
            con.execute(
                "SELECT DISTINCT customer_key FROM transactions "
                "WHERE t_dat >= CAST(? AS DATE) AND t_dat < CAST(? AS DATE) ORDER BY 1",
                [start, end],
            )
            .df()["customer_key"]
            .tolist()
        )

    # Chunked because the retrieval frame is ~60M rows per week and downsampling discards ~95%
    # of it. Both retrieval and downsampling are per-customer, so chunking cannot change the
    # result — it only bounds peak memory.
    parts = []
    for i in range(0, len(customers), chunk):
        part_customers = customers[i : i + chunk]
        register_customers(con, part_customers)
        candidates = build_candidates(con, as_of, part_customers, als_model=als_model)
        labelled = label_candidates(con, candidates, as_of)
        del candidates
        if downsample:
            labelled = downsample_negatives(labelled, ratio=neg_ratio)
        parts.append(build_features(con, labelled, as_of))
        del labelled
        gc.collect()

    return pd.concat(parts, ignore_index=True).sort_values(
        ["customer_key", "article_id"], kind="stable"
    )


def build_multiweek_training_set(
    con: duckdb.DuckDBPyConnection,
    weeks: list[date],
    als_model=None,
    downsample: bool = True,
    neg_ratio: int = NEG_SAMPLE_RATIO,
) -> pd.DataFrame:
    """Stack several training weeks.

    Each week is built independently with its own `as_of`, so a row's features never see
    anything from its own target week — stacking does not weaken the time split, it just
    repeats it.

    A `week` column is added because the same customer appears once per week and those
    occurrences are *different* ranking queries. Merging them into one LambdaRank group would
    ask the model to rank August candidates against September ones, which is not the task.

    Each week is spilled to Parquet and dropped before the next one starts. Holding all weeks
    in memory OOM-killed the process at week three: the peak is not the training set itself
    but `build_candidates`, which materialises roughly 21M candidate rows per week before
    downsampling throws most of them away.
    """
    tmp = Path(tempfile.mkdtemp(prefix="hm_train_"))
    try:
        for w in weeks:
            part = build_training_set(
                con, w, als_model=als_model, downsample=downsample, neg_ratio=neg_ratio
            )
            part.insert(0, "week", pd.Timestamp(w))
            part = _downcast(part)
            part.to_parquet(tmp / f"{w.isoformat()}.parquet", index=False)
            print(
                f"  week {w}: {len(part):,} rows, {int(part['label'].sum()):,} positives, "
                f"{part['customer_key'].nunique():,} customers",
                flush=True,
            )
            del part
            gc.collect()

        out = pd.concat(
            (pd.read_parquet(p) for p in sorted(tmp.glob("*.parquet"))), ignore_index=True
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Concatenating categoricals whose category sets differ silently yields object dtype, and
    # LightGBM rejects object columns outright. Each week sees a different subset of product
    # groups, so this always triggers with more than one week.
    for col in CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out.sort_values(["week", "customer_key"], kind="stable").reset_index(drop=True)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 on feature columns. Halves the training frame; LightGBM bins to
    255 buckets anyway, so the discarded precision cannot change a split point."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """LightGBM group vector over contiguous (week, customer) runs.

    `week` is absent for single-week training sets, in which case customer alone is the group.
    """
    keys = ["week", "customer_key"] if "week" in df.columns else ["customer_key"]
    return df.groupby(keys, sort=False).size().to_numpy()
