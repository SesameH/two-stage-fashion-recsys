"""Build the artefacts the API loads at startup.

Run: python -m src.serve.precompute [--with-als]

Online, the request budget does not allow scanning 31M transactions, so everything that
depends only on `as_of` — not on the incoming customer — is materialised here into
`data/serving/`.

Critically, none of the aggregations are re-implemented for serving: this module calls the
same functions the offline pipeline calls (`src.recall.strategies`, `src.features.builder`)
and simply persists their output. A second SQL implementation for the online path is the
single most likely source of training-serving skew, so there isn't one.

`meta.json` records the cutoff the snapshot was built for. A serving artefact that does not
carry its own `as_of` is how offline and online silently diverge.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.config import DATA_PROCESSED, FEATURE_LOOKBACK_DAYS, VAL_START
from src.data.db import connect, load_sql
from src.features.builder import CUSTOMER_GROUP_ALL_SQL, all_customers_sql
from src.recall import strategies as st

SERVING = DATA_PROCESSED.parent / "serving"


def _log(msg: str) -> None:
    print(f"[precompute] {msg}", flush=True)


def register_all_customers(con) -> None:
    """`eval_customers` for the retrieval strategies, covering everyone with any history."""
    con.execute(
        "CREATE OR REPLACE TEMP TABLE eval_customers AS "
        "SELECT DISTINCT customer_key FROM transactions"
    )


def build_faiss_index(item_factors: np.ndarray, out_dir: Path) -> int:
    """FAISS index over ALS item factors.

    Inner product, not L2: ALS scores a (customer, item) pair by dot product, so an L2 index
    would rank by a different quantity than the factorisation was trained to optimise.
    """
    import faiss

    factors = np.ascontiguousarray(item_factors.astype("float32"))
    index = faiss.IndexFlatIP(factors.shape[1])
    index.add(factors)
    faiss.write_index(index, str(out_dir / "als_items.faiss"))
    return index.ntotal


def build_demo_tables(con, as_of: date, out: Path) -> None:
    """Tables the evaluation console needs, which the request path must never touch.

    Two of these deliberately contain data from on or after `as_of`: the console shows which
    predictions were actually correct, and that is only knowable from the target week. They are
    written with a `demo_` prefix and read by separate endpoints so that no amount of careless
    wiring can leak them into a recommendation.
    """
    _log("building demo tables (contain post-cutoff data)")
    week_end = as_of + timedelta(days=7)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE demo_customers AS
        SELECT DISTINCT customer_key FROM transactions
        WHERE t_dat >= DATE '{as_of}' AND t_dat < DATE '{week_end}'
        """
    )

    # POST-CUTOFF: what they actually bought. Ground truth, for hit marking only.
    con.execute(
        f"""COPY (
            SELECT t.customer_key, t.article_id, count(*) AS n
            FROM transactions t JOIN demo_customers d USING (customer_key)
            WHERE t.t_dat >= DATE '{as_of}' AND t.t_dat < DATE '{week_end}'
            GROUP BY 1, 2
        ) TO '{out / "demo_truth.parquet"}' (FORMAT PARQUET)"""
    )

    con.execute(
        f"""COPY (
            SELECT customer_key, article_id, t_dat, price FROM (
                SELECT t.customer_key, t.article_id, t.t_dat, t.price,
                       row_number() OVER (
                           PARTITION BY t.customer_key ORDER BY t.t_dat DESC, t.article_id
                       ) AS rn
                FROM transactions t JOIN demo_customers d USING (customer_key)
                WHERE t.t_dat < DATE '{as_of}'
            ) WHERE rn <= 40
        ) TO '{out / "demo_history.parquet"}' (FORMAT PARQUET)"""
    )

    # The model-facing articles.parquet carries only the columns the model needs.
    con.execute(
        f"""COPY (
            SELECT article_id, prod_name, product_type_name, colour_group_name,
                   product_group_name, index_group_name, detail_desc
            FROM articles
        ) TO '{out / "demo_articles.parquet"}' (FORMAT PARQUET)"""
    )

    # Segments let the console show cold-start and dormant customers, not only easy ones.
    con.execute(
        f"""COPY (
            SELECT d.customer_key,
                   coalesce(h.n_prior, 0) AS n_prior,
                   h.recency,
                   c.age,
                   CASE
                       WHEN h.customer_key IS NULL THEN 'never purchased'
                       WHEN h.recency <= 7  THEN 'active <=7d'
                       WHEN h.recency <= 30 THEN 'active 8-30d'
                       WHEN h.recency <= 90 THEN 'active 31-90d'
                       ELSE 'dormant >90d'
                   END AS recency_segment
            FROM demo_customers d
            LEFT JOIN (
                SELECT t.customer_key, count(*) AS n_prior,
                       DATE_DIFF('day', max(t.t_dat), DATE '{as_of}') AS recency
                FROM transactions t JOIN demo_customers dd USING (customer_key)
                WHERE t.t_dat < DATE '{as_of}'
                GROUP BY 1
            ) h USING (customer_key)
            LEFT JOIN customers c USING (customer_key)
        ) TO '{out / "demo_segments.parquet"}' (FORMAT PARQUET)"""
    )

    for name in ("demo_truth", "demo_history", "demo_articles", "demo_segments"):
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{out / (name + '.parquet')}')"
        ).fetchone()[0]
        _log(f"{name}: {n:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=VAL_START)
    parser.add_argument("--with-als", action="store_true")
    parser.add_argument("--out", type=Path, default=SERVING)
    parser.add_argument("--n-repurchase", type=int, default=50)
    parser.add_argument("--n-variants", type=int, default=150)
    parser.add_argument(
        "--with-demo", action="store_true",
        help="also build the demo console tables (these contain post-cutoff data)",
    )
    args = parser.parse_args()

    as_of, out = args.as_of, args.out
    out.mkdir(parents=True, exist_ok=True)
    con = connect()
    register_all_customers(con)
    t0 = time.perf_counter()
    lb = FEATURE_LOOKBACK_DAYS

    # --- feature blocks, from the same SQL the offline builder uses -------------
    blocks = {
        "customer_features": con.execute(load_sql("customer_recency"), [as_of, as_of, lb]).df(),
        "article_features": con.execute(load_sql("article_popularity"), [as_of, lb]).df(),
        "affinity": con.execute(
            load_sql("customer_article_affinity"), [as_of, as_of, lb]
        ).df(),
        "customer_category": con.execute(
            CUSTOMER_GROUP_ALL_SQL, {"as_of": as_of, "lookback": lb}
        ).df(),
        "customer_productcode": con.execute(
            all_customers_sql("customer_productcode_affinity"), [as_of, as_of, lb]
        ).df(),
    }

    # --- retrieval tables, from the same strategy functions ---------------------
    retrieval = {
        "repurchase": st.r1_repurchase(con, as_of, n=args.n_repurchase)[
            ["customer_key", "article_id", "rank", "score"]
        ],
        "variants": st.r6_product_variant(con, as_of, n=args.n_variants)[
            ["customer_key", "article_id", "rank", "score"]
        ],
        "popularity_global": st.global_popularity_table(con, as_of, n=400),
        "popularity_age": st.age_popularity_table(con, as_of, n=200),
        # R5 is customer-specific and cannot be reduced to a shared table, so it is
        # materialised per customer. Leaving it out of the online path cost measurable MAP.
        "category_candidates": st.r5_category(con, as_of, n=100)[
            ["customer_key", "article_id", "rank", "score"]
        ],
    }

    for name, df in {**blocks, **retrieval}.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        _log(f"{name}: {len(df):,} rows")

    con.execute(
        f"COPY (SELECT article_id, product_code, product_group_name, index_group_name, "
        f"garment_group_name, product_type_no, section_no, department_no, "
        f"perceived_colour_value_id, graphical_appearance_no FROM articles) "
        f"TO '{out / 'articles.parquet'}' (FORMAT PARQUET)"
    )
    con.execute(
        f"COPY (SELECT customer_key, age, fn, active, club_member_status, "
        f"fashion_news_frequency FROM customers) "
        f"TO '{out / 'customers.parquet'}' (FORMAT PARQUET)"
    )

    if args.with_demo:
        build_demo_tables(con, as_of, out)

    meta = {
        "as_of": as_of.isoformat(),
        "with_als": args.with_als,
        "with_demo": args.with_demo,
    }
    if args.with_als:
        from src.baselines.als import fit

        model = fit(con, as_of)
        np.save(out / "als_item_factors.npy", model.model.item_factors)
        np.save(out / "als_article_ids.npy", model.article_ids)
        meta["faiss_vectors"] = build_faiss_index(model.model.item_factors, out)
        _log(f"faiss: {meta['faiss_vectors']:,} vectors")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    _log(f"done in {time.perf_counter() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
