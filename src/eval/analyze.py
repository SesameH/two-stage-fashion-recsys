"""Error analysis: where the model wins, where it fails, and what it never sees.

Run: python -m src.eval.analyze   (requires models/ranker.txt from `make train`)

An aggregate MAP@12 hides which customers the model actually helps. This module cuts the
validation week by customer segment and by item popularity, compares against B2 in every
segment, and measures catalogue coverage and category diversity.

Writes reports/error_analysis.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baselines import heuristics as h
from src.config import MODELS, REPORTS, VAL_START, K
from src.data.db import connect, register_customers
from src.eval.metrics import apk
from src.eval.split import ground_truth
from src.rank.predict import predict_week


def customer_segments(con, as_of: date, customers: list[int]) -> pd.DataFrame:
    """Segment the evaluation population by prior activity, all measured before `as_of`."""
    register_customers(con, customers)
    return con.execute(
        """
        WITH hist AS (
            SELECT t.customer_key,
                   count(*) AS n_prior,
                   DATE_DIFF('day', max(t.t_dat), CAST(? AS DATE)) AS recency
            FROM transactions t
            JOIN eval_customers e USING (customer_key)
            WHERE t.t_dat < CAST(? AS DATE)
            GROUP BY 1
        )
        SELECT e.customer_key,
               coalesce(hi.n_prior, 0) AS n_prior,
               hi.recency,
               c.age,
               CASE
                   WHEN hi.customer_key IS NULL THEN '0 never purchased'
                   WHEN hi.recency <= 7  THEN '1 active <=7d'
                   WHEN hi.recency <= 30 THEN '2 active 8-30d'
                   WHEN hi.recency <= 90 THEN '3 active 31-90d'
                   ELSE '4 dormant >90d'
               END AS recency_segment,
               CASE
                   WHEN coalesce(hi.n_prior, 0) = 0 THEN '0 none'
                   WHEN hi.n_prior <= 5  THEN '1 low (1-5)'
                   WHEN hi.n_prior <= 20 THEN '2 mid (6-20)'
                   WHEN hi.n_prior <= 60 THEN '3 high (21-60)'
                   ELSE '4 very high (60+)'
               END AS frequency_segment,
               CASE
                   WHEN c.age IS NULL THEN 'unknown'
                   WHEN c.age < 25 THEN '<25'
                   WHEN c.age < 35 THEN '25-34'
                   WHEN c.age < 50 THEN '35-49'
                   ELSE '50+'
               END AS age_segment
        FROM eval_customers e
        LEFT JOIN hist hi USING (customer_key)
        LEFT JOIN customers c USING (customer_key)
        """,
        [as_of, as_of],
    ).df()


def per_customer_ap(
    truth: dict[int, list[int]], preds: dict[int, list[int]], customers: list[int]
) -> pd.Series:
    return pd.Series(
        {c: apk(truth[c], preds.get(c, []), K) for c in customers}, name="ap"
    )


def segment_table(seg: pd.DataFrame, column: str, ap_model: pd.Series, ap_b2: pd.Series):
    df = seg.set_index("customer_key")[[column]].copy()
    df["model"] = ap_model
    df["b2"] = ap_b2
    out = df.groupby(column).agg(
        customers=("model", "size"),
        map_model=("model", "mean"),
        map_b2=("b2", "mean"),
    )
    out["lift"] = out["map_model"] / out["map_b2"].replace(0, np.nan) - 1
    return out.round({"map_model": 5, "map_b2": 5, "lift": 3})


def item_popularity_analysis(con, as_of: date, truth, preds, customers) -> pd.DataFrame:
    """Split ground-truth purchases by the item's prior popularity decile: what gets missed?"""
    pop = con.execute(
        """
        SELECT article_id, count(*) AS n
        FROM transactions
        WHERE t_dat < CAST(? AS DATE) AND t_dat >= CAST(? AS DATE) - INTERVAL 30 DAY
        GROUP BY 1
        """,
        [as_of, as_of],
    ).df()
    pop["decile"] = pd.qcut(pop["n"].rank(method="first"), 10, labels=False) + 1
    decile = dict(zip(pop["article_id"], pop["decile"]))

    rows = []
    for c in customers:
        hit = set(preds.get(c, []))
        for a in truth[c]:
            rows.append({"decile": decile.get(a, 0), "hit": int(a in hit)})

    df = pd.DataFrame(rows)
    out = df.groupby("decile").agg(purchases=("hit", "size"), hit_rate=("hit", "mean"))
    out["share_of_truth"] = out["purchases"] / out["purchases"].sum()
    return out.round({"hit_rate": 4, "share_of_truth": 4})


def diversity(con, preds: dict[int, list[int]]) -> dict:
    """Catalogue coverage and category entropy — accuracy is not the only thing that matters."""
    recommended = Counter(a for items in preds.values() for a in items)
    n_catalog = con.execute("SELECT count(*) FROM articles").fetchone()[0]

    groups = con.execute("SELECT article_id, product_group_name FROM articles").df()
    gmap = dict(zip(groups["article_id"], groups["product_group_name"]))
    gcount = Counter()
    for a, n in recommended.items():
        gcount[gmap.get(a, "unknown")] += n

    total = sum(gcount.values())
    probs = np.array([v / total for v in gcount.values()])
    entropy = float(-(probs * np.log2(probs)).sum())

    top = recommended.most_common(1)[0]
    return {
        "distinct_items_recommended": len(recommended),
        "catalogue_coverage": round(len(recommended) / n_catalog, 4),
        "category_entropy_bits": round(entropy, 3),
        "max_category_entropy_bits": round(float(np.log2(len(gcount))), 3),
        "most_recommended_item_share": round(top[1] / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=VAL_START)
    args = parser.parse_args()
    as_of = args.as_of

    con = connect()
    booster = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
    features = json.loads((MODELS / "features.json").read_text())

    gt = ground_truth(con, as_of)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    customers = sorted(truth)
    print(f"scoring {len(customers):,} customers ...")

    preds = predict_week(con, booster, features, as_of, customers)

    register_customers(con, customers)
    bestsellers = h.top_articles(con, as_of, days=7, k=K)
    b2 = h.pad(h.repurchase(con, as_of, lookback_days=90, k=K), customers, bestsellers, k=K)

    ap_model = per_customer_ap(truth, preds, customers)
    ap_b2 = per_customer_ap(truth, b2, customers)
    seg = customer_segments(con, as_of, customers)

    tables = {
        "By prior activity (recency)": segment_table(seg, "recency_segment", ap_model, ap_b2),
        "By purchase frequency": segment_table(seg, "frequency_segment", ap_model, ap_b2),
        "By age": segment_table(seg, "age_segment", ap_model, ap_b2),
    }
    pop_table = item_popularity_analysis(con, as_of, truth, preds, customers)
    div = diversity(con, preds)

    for name, t in tables.items():
        print(f"\n{name}\n{t.to_string()}")
    print(f"\nBy item popularity decile (10 = most popular)\n{pop_table.to_string()}")
    print(f"\nDiversity\n{json.dumps(div, indent=2)}")

    REPORTS.mkdir(exist_ok=True)
    parts = [f"# Error analysis\n\nValidation week `{as_of}`, {len(customers):,} customers.\n"]
    for name, t in tables.items():
        parts.append(f"## {name}\n\n{t.to_markdown()}\n")
    parts.append(
        "## By item popularity decile\n\n"
        "Deciles are over trailing-30-day sales; decile 0 means the article had not sold at "
        "all before the cutoff.\n\n" + pop_table.to_markdown() + "\n"
    )
    parts.append(f"## Diversity\n\n```json\n{json.dumps(div, indent=2)}\n```\n")
    (REPORTS / "error_analysis.md").write_text("\n".join(parts))
    print(f"\nwrote {REPORTS / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
