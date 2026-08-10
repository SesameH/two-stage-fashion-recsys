"""Assemble the final candidate set that the ranking stage consumes.

Strategies play one of two roles, and the distinction is measured, not assumed:

  generators  contribute candidates. Their union, fused by RRF and cut to N_CANDIDATES,
              is the candidate set.
  annotators  contribute only features. Their rank and score are joined onto candidates
              somebody else proposed.

R3 item-kNN was an annotator: the leave-one-out ablation in reports/recall.md shows its
candidates *lower* union recall at a 300 budget (0.2737 -> 0.2766 when removed), so its slots
are worth more to other strategies, while its similarity score is still a real signal.

`ANNOTATORS` is now empty, and the reason is a serving constraint rather than a modelling one.
The serving snapshot has no r3 tables — per-customer kNN lists are ~97M rows, which does not
fit the 4 GiB deployment — so `rank_r3_item_knn` and `score_r3_item_knn` were constant at
request time while ranking #10 and #17 by training gain. Measured on 20k validation customers,
scoring the offline candidate set with those columns forced to their absence sentinel and
`n_sources` reduced to generators only cost 1.5% of MAP@12 (0.03398 -> 0.03345) and changed the
top 12 for 64% of customers. A feature the production path cannot supply is not a feature, so
they are gone from both sides instead of skewed on one. Serving r3 properly (persist an
item-item similarity table and derive candidates per request) is the alternative and is
recorded as future work.

The machinery stays: add a name to `ANNOTATORS` and its columns reappear on both paths.

Output is one row per (customer, candidate) with per-source rank/score columns, written to
Parquet so the ranking stage never has to recompute retrieval.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.config import N_CANDIDATES
from src.features.builder import MISSING_RANK, MISSING_SCORE
from src.recall import base, strategies

# Per-strategy retrieval budgets, shared with src/serve/precompute.py. These used to be written
# twice — once here, once as precompute's CLI defaults — which is a silent skew waiting to
# happen: a budget changed on one side alone shifts every `rank_*` feature and the fused score
# for that source only in production.
GENERATOR_BUDGET = {
    "r1_repurchase": 50,
    "r2_popularity": 200,
    "r2b_global": 400,
    "r5_category": 100,
    "r6_product_variant": 150,
}
GENERATORS = list(GENERATOR_BUDGET)

# Available annotators, and the budget each runs with. Enabling one requires the serving
# snapshot to carry the same table, or the feature is constant online — see the module docstring.
ANNOTATOR_FNS = {
    "r3_item_knn": lambda con, as_of: strategies.r3_item_knn(con, as_of, n=150),
    "r3_last_knn": lambda con, as_of: strategies.r3_last_knn(con, as_of, n=100),
}
ANNOTATORS: list[str] = []


def retrieve(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    customers: list[int],
    als_model=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy. Returns (generator rows, annotator rows) in long format."""
    b = GENERATOR_BUDGET
    gen = [
        strategies.r1_repurchase(con, as_of, n=b["r1_repurchase"]),
        strategies.r2_popularity(con, as_of, n=b["r2_popularity"]),
        strategies.r2b_global_popularity(con, as_of, n=b["r2b_global"]),
        strategies.r5_category(con, as_of, n=b["r5_category"]),
        strategies.r6_product_variant(con, as_of, n=b["r6_product_variant"]),
    ]
    if als_model is not None:
        gen.append(strategies.r4_als(als_model, customers, n=50))

    ann = [ANNOTATOR_FNS[name](con, as_of) for name in ANNOTATORS]
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

    # A missing rank means "not retrieved by this source". The sentinel comes from
    # features/builder.py so that this path and the serving path cannot drift apart.
    for s in sources:
        out[f"rank_{s}"] = out[f"rank_{s}"].fillna(MISSING_RANK).astype("int32")
        out[f"score_{s}"] = out[f"score_{s}"].fillna(MISSING_SCORE).astype("float32")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)

    return out
