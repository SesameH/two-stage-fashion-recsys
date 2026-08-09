"""Evaluate the baseline ladder B0-B3 on the validation week.

Run: python -m src.baselines.run [--as-of 2020-09-16]

Writes reports/baselines.md. Every baseline sees the identical customer population and the
identical `as_of` cutoff, so the numbers are comparable by construction.
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import pandas as pd

from src.baselines import heuristics as h
from src.config import REPORTS, VAL_START, K
from src.data.db import connect, register_customers
from src.eval.metrics import mapk, recall_at_k
from src.eval.split import ground_truth


def evaluate(
    name: str,
    preds: dict[int, list[int]],
    truth: dict[int, list[int]],
    customers: list[int],
    elapsed: float,
) -> dict:
    """Score a baseline. Customers the baseline cannot serve get an empty list, not a skip —
    coverage is part of what is being measured."""
    actual = [truth[c] for c in customers]
    predicted = [preds.get(c, []) for c in customers]
    served = sum(1 for p in predicted if p)
    return {
        "baseline": name,
        "MAP@12": round(mapk(actual, predicted, K), 5),
        "recall@12": round(recall_at_k(actual, predicted, K), 5),
        "coverage": round(served / len(customers), 4),
        "sec": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=VAL_START)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--skip-als", action="store_true")
    args = parser.parse_args()

    as_of = args.as_of
    con = connect()

    gt = ground_truth(con, as_of)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    customers = sorted(truth)
    print(f"as_of={as_of}  eval customers={len(customers):,}")

    register_customers(con, customers)
    results = []

    t = time.perf_counter()
    bestsellers = h.top_articles(con, as_of, days=7, k=K)
    b0 = dict.fromkeys(customers, bestsellers)
    results.append(evaluate("B0 bestsellers (7d)", b0, truth, customers, time.perf_counter() - t))

    t = time.perf_counter()
    repurchase = h.repurchase(con, as_of, lookback_days=args.lookback, k=K)
    results.append(
        evaluate("B1 repurchase (90d)", repurchase, truth, customers, time.perf_counter() - t)
    )

    t = time.perf_counter()
    b2 = h.pad(repurchase, customers, bestsellers, k=K)
    results.append(
        evaluate("B2 repurchase + fill", b2, truth, customers, time.perf_counter() - t)
    )

    if not args.skip_als:
        from src.baselines.als import fit

        t = time.perf_counter()
        model = fit(con, as_of)
        als_preds = model.recommend(customers, k=K)
        results.append(
            evaluate("B3 ALS (128f, 365d)", als_preds, truth, customers, time.perf_counter() - t)
        )

        t = time.perf_counter()
        b3_padded = h.pad(als_preds, customers, bestsellers, k=K)
        results.append(
            evaluate("B3b ALS + fill", b3_padded, truth, customers, time.perf_counter() - t)
        )

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "baselines.md"
    out.write_text(
        f"# Baseline ladder\n\n"
        f"- Validation week: `{as_of}` to `{as_of.fromordinal(as_of.toordinal() + 7)}`\n"
        f"- Evaluation population: {len(customers):,} customers with >=1 purchase in that week\n"
        f"- All features cut off strictly before `{as_of}`\n\n"
        f"{df.to_markdown(index=False)}\n\n"
        f"`coverage` is the share of evaluated customers the baseline returns any prediction for.\n"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
