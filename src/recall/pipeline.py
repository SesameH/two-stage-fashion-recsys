"""Assemble the final candidate set that the ranking stage consumes.

Strategies play one of two roles, and the distinction is measured, not assumed:

  generators  contribute candidates. Their union, fused by RRF and cut to N_CANDIDATES,
              is the candidate set.
  annotators  contribute only features. Their rank and score are joined onto candidates
              somebody else proposed.

R3 item-kNN is an annotator because the leave-one-out ablation in reports/recall.md shows
its candidates *lower* union recall at a 300 budget (0.2737 -> 0.2766 when removed): the
slots it occupies are worth more to other strategies. Its similarity score is still a real
signal about a candidate, so it is kept as a feature rather than thrown away.

Output is one row per (customer, candidate) with per-source rank/score columns, written to
Parquet so the ranking stage never has to recompute retrieval.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.config import N_CANDIDATES
from src.recall import base, strategies

GENERATORS = ["r1_repurchase", "r2_popularity", "r2b_global", "r5_category", "r6_product_variant"]
ANNOTATORS = ["r3_item_knn", "r3_last_knn"]


def retrieve(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    customers: list[int],
    als_model=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy. Returns (generator rows, annotator rows) in long format."""
    gen = [
        strategies.r1_repurchase(con, as_of, n=50),
        strategies.r2_popularity(con, as_of, n=200),
        strategies.r2b_global_popularity(con, as_of, n=400),
        strategies.r5_category(con, as_of, n=100),
        strategies.r6_product_variant(con, as_of, n=150),
    ]
    if als_model is not None:
        gen.append(strategies.r4_als(als_model, customers, n=50))

    ann = [
        strategies.r3_item_knn(con, as_of, n=150),
        strategies.r3_last_knn(con, as_of, n=100),
    ]
    return base.union(*gen), base.union(*ann)


def build_candidates(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    customers: list[int],
    als_model=None,
    n_candidates: int = N_CANDIDATES,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Candidate set with retrieval features, one row per (customer, candidate).

    Columns: customer_key, article_id, rrf_score, n_sources, then rank_<source> and
    score_<source> for every strategy. Missing means "this strategy did not propose it";
    ranks are filled with a sentinel rather than 0 so LightGBM reads them monotonically.
    """
    gen, ann = retrieve(con, as_of, customers, als_model)

    fused = base.fuse(gen)
    fused["cand_rank"] = fused.groupby("customer_key", sort=False).cumcount() + 1
    candidates = fused[fused["cand_rank"] <= n_candidates].drop(columns="cand_rank")
    candidates = candidates.rename(columns={"rrf": "rrf_score"})

    long = base.union(gen, ann)
    con.register("cand_tmp", candidates)
    con.register("long_tmp", long)

    sources = sorted(long["source"].unique())
    cols = ",\n".join(
        f"       max(CASE WHEN l.source = '{s}' THEN l.rank END)  AS rank_{s},\n"
        f"       max(CASE WHEN l.source = '{s}' THEN l.score END) AS score_{s}"
        for s in sources
    )
    out = con.execute(
        f"""
        SELECT c.customer_key,
               c.article_id,
               c.rrf_score,
               count(DISTINCT l.source) AS n_sources,
{cols}
        FROM cand_tmp c
        LEFT JOIN long_tmp l USING (customer_key, article_id)
        GROUP BY 1, 2, 3
        """
    ).df()
    con.unregister("cand_tmp")
    con.unregister("long_tmp")

    # A missing rank means "not retrieved by this source". 9999 keeps the feature monotone
    # (smaller = better) instead of making absence look like the best possible rank.
    for s in sources:
        out[f"rank_{s}"] = out[f"rank_{s}"].fillna(9999).astype("int32")
        out[f"score_{s}"] = out[f"score_{s}"].fillna(0.0).astype("float32")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)

    return out
