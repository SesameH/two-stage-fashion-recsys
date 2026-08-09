"""Shared contract for retrieval strategies R1-R5.

Every strategy returns a long DataFrame with exactly these columns:

    customer_key  BIGINT
    article_id    INTEGER
    source        str      -- 'r1_repurchase', 'r2_popularity', ...
    rank          int      -- 1-based, within (customer, source)
    score         float    -- strategy-native, NOT comparable across sources

The long format is deliberate. `rank` and `score` per source become ranking-stage features,
and the literature is consistent that retrieval-source signals are among the strongest
features a re-ranker has. Collapsing to a bare candidate set here would throw them away.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["customer_key", "article_id", "source", "rank", "score"]


def empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_key": pd.Series(dtype="int64"),
            "article_id": pd.Series(dtype="int32"),
            "source": pd.Series(dtype="object"),
            "rank": pd.Series(dtype="int32"),
            "score": pd.Series(dtype="float32"),
        }
    )


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Coerce a strategy's raw output into the contract."""
    out = df.copy()
    # Categorical, not object: the long frames reach tens of millions of rows and one Python
    # string reference per row is enough to OOM the training-set build.
    out["source"] = pd.Categorical([source] * len(out), categories=[source])
    out["customer_key"] = out["customer_key"].astype("int64")
    out["article_id"] = out["article_id"].astype("int32")
    out["rank"] = out["rank"].astype("int32")
    out["score"] = out["score"].astype("float32")
    return out[COLUMNS]


def union(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate strategy outputs. Duplicates across sources are kept, not merged —
    the ranking stage needs to see that an item was proposed by several strategies."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return empty()
    return pd.concat(frames, ignore_index=True)


RRF_K = 60.0  # standard damping constant; the metric is insensitive to it in 20-100


def fuse(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Reciprocal Rank Fusion across sources.

    fused_score(customer, article) = sum over sources of w_source / (RRF_K + rank)

    Min-rank fusion was tried first and is actively harmful once the union exceeds the
    candidate budget: it lets one strategy's rank-1 item outrank an item that three
    strategies independently placed in their top ten. RRF rewards that agreement, and being
    rank-based it needs no score calibration between strategies whose scores are not
    remotely comparable.
    """
    w = df["source"].map(weights).fillna(1.0) if weights else 1.0
    scored = df.assign(rrf=w / (RRF_K + df["rank"]))
    return (
        scored.groupby(["customer_key", "article_id"], sort=False)["rrf"]
        .sum()
        .reset_index()
        .sort_values(["customer_key", "rrf"], ascending=[True, False], kind="stable")
    )


def to_candidate_lists(
    df: pd.DataFrame, k: int | None = None, weights: dict[str, float] | None = None
) -> dict[int, list[int]]:
    """Collapse the long frame to per-customer candidate lists ordered by fused score."""
    best = fuse(df, weights)
    out: dict[int, list[int]] = {}
    for customer, article in zip(best["customer_key"], best["article_id"]):
        items = out.setdefault(int(customer), [])
        if k is None or len(items) < k:
            items.append(int(article))
    return out
