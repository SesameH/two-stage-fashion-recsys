"""Measure recall@100 per retrieval strategy and for the union.

Run: python -m src.recall.run [--as-of 2020-09-16]

Writes reports/recall.md. recall@100 is the ceiling on everything the ranking stage can
achieve, so this table is the one to check before touching LightGBM.
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import pandas as pd

from src.config import N_CANDIDATES, REPORTS, VAL_START
from src.data.db import connect, register_customers
from src.eval.metrics import recall_at_k
from src.eval.split import ground_truth
from src.recall import base, strategies


def score(
    name: str,
    df: pd.DataFrame,
    truth: dict[int, list[int]],
    customers: list[int],
    k: int,
    elapsed: float,
) -> dict:
    preds = base.to_candidate_lists(df, k=k)
    actual = [truth[c] for c in customers]
    predicted = [preds.get(c, []) for c in customers]
    sizes = [len(p) for p in predicted]
    return {
        "strategy": name,
        f"recall@{k}": round(recall_at_k(actual, predicted, k), 5),
        "coverage": round(sum(1 for s in sizes if s) / len(sizes), 4),
        "avg_candidates": round(sum(sizes) / len(sizes), 1),
        "sec": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=VAL_START)
    parser.add_argument("--k", type=int, default=N_CANDIDATES)
    parser.add_argument("--skip-als", action="store_true")
    args = parser.parse_args()

    as_of, k = args.as_of, args.k
    con = connect()

    gt = ground_truth(con, as_of)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    customers = sorted(truth)
    print(f"as_of={as_of}  eval customers={len(customers):,}  k={k}")

    register_customers(con, customers)

    frames, results = {}, []
    plan = [
        ("R1 repurchase", lambda: strategies.r1_repurchase(con, as_of, n=50)),
        ("R2 popularity (age)", lambda: strategies.r2_popularity(con, as_of, n=200)),
        ("R2b popularity (global)", lambda: strategies.r2b_global_popularity(con, as_of, n=400)),
        ("R3 item-kNN", lambda: strategies.r3_item_knn(con, as_of, n=150)),
        ("R5 category", lambda: strategies.r5_category(con, as_of, n=100)),
        ("R6 product variant", lambda: strategies.r6_product_variant(con, as_of, n=150)),
    ]

    if not args.skip_als:
        from src.baselines.als import fit

        t = time.perf_counter()
        als_model = fit(con, as_of)
        print(f"  (ALS fit {time.perf_counter() - t:.0f}s)")
        plan.append(("R4 ALS", lambda: strategies.r4_als(als_model, customers, n=50)))

    for name, fn in plan:
        t = time.perf_counter()
        df = fn()
        elapsed = time.perf_counter() - t
        frames[name] = df
        results.append(score(name, df, truth, customers, k, elapsed))
        print(f"  {name}: {len(df):,} rows in {elapsed:.1f}s")

    t = time.perf_counter()
    merged = base.union(*frames.values())
    results.append(score("UNION (all)", merged, truth, customers, k, time.perf_counter() - t))

    # Recall vs budget: the curve that justifies N_CANDIDATES.
    union_lists = base.to_candidate_lists(merged, k=1000)
    union_preds = [union_lists.get(c, []) for c in customers]
    actual = [truth[c] for c in customers]
    curve = {b: round(recall_at_k(actual, union_preds, b), 5) for b in (12, 50, 100, 200, 300, 500)}
    print("\nrecall vs budget:", curve)

    # Leave-one-out: how much does each strategy add that no other provides?
    for name in frames:
        others = base.union(*[f for k_, f in frames.items() if k_ != name])
        row = score(f"  union without {name}", others, truth, customers, k, 0.0)
        results.append(row)

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))

    union_recall = results[len(frames)][f"recall@{k}"]
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "recall.md"
    out.write_text(
        f"# Retrieval layer\n\n"
        f"- Validation week starting `{as_of}`, {len(customers):,} customers\n"
        f"- All strategies read only `t_dat < {as_of}`\n"
        f"- Union recall@{k}: **{union_recall}**\n\n"
        f"{df.to_markdown(index=False)}\n\n"
        f"## Recall vs candidate budget (union)\n\n"
        f"| budget | " + " | ".join(str(b) for b in curve) + " |\n"
        "|---|" + "---|" * len(curve) + "\n"
        "| recall | " + " | ".join(str(v) for v in curve.values()) + " |\n\n"
        "The `union without X` rows are a leave-one-out ablation: the drop from the UNION row\n"
        "is the unique contribution of strategy X, which is the only honest way to justify\n"
        "keeping five strategies instead of one.\n"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
