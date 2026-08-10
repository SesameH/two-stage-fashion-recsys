"""Rolling multi-week evaluation and candidate-budget sweep.

Run: python -m src.eval.rolling

One validation week cannot separate model quality from that week's promotions. This scores the
same trained model on several consecutive weeks, against B2 on each, and sweeps the candidate
budget. Each week uses `as_of = week start`, so features and candidates for every week are
generated exactly as they would have been in production on that Monday.

The model is NOT retrained per week: the point is to test whether a model trained on
2020-09-09 generalises, not to fit each week separately.

Writes reports/rolling.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import lightgbm as lgb
import pandas as pd

from src.baselines import heuristics as h
from src.config import MODELS, REPORTS, VAL_START, K, write_report
from src.data.db import connect, register_customers
from src.eval.metrics import mapk, recall_at_k
from src.eval.split import ground_truth
from src.rank.predict import predict_week
from src.recall.pipeline import build_candidates


def evaluate_week(con, booster, features, as_of: date, chunk: int = 4000) -> dict:
    gt = ground_truth(con, as_of)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    customers = sorted(truth)
    actual = [truth[c] for c in customers]

    preds = predict_week(con, booster, features, as_of, customers, chunk=chunk)

    register_customers(con, customers)
    bestsellers = h.top_articles(con, as_of, days=7, k=K)
    b2 = h.pad(h.repurchase(con, as_of, lookback_days=90, k=K), customers, bestsellers, k=K)

    model_map = mapk(actual, [preds.get(c, []) for c in customers], K)
    b2_map = mapk(actual, [b2.get(c, []) for c in customers], K)
    return {
        "week": as_of.isoformat(),
        "customers": len(customers),
        "MAP@12": round(model_map, 5),
        "B2": round(b2_map, 5),
        "lift": round(model_map / b2_map - 1, 3) if b2_map else None,
    }


def budget_sweep(
    con, booster, features, as_of: date, budgets: list[int], n_sample: int, chunk: int = 4000
) -> pd.DataFrame:
    """MAP@12 and retrieval recall as a function of the candidate budget."""
    import random

    gt = ground_truth(con, as_of)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    random.seed(0)
    customers = sorted(random.sample(sorted(truth), min(n_sample, len(truth))))
    actual = [truth[c] for c in customers]

    rows = []
    for b in budgets:
        register_customers(con, customers)
        # build_candidates returns the wide, already-fused frame, so ordering is by rrf_score
        # rather than by the long-format rank columns.
        cand = build_candidates(con, as_of, customers, n_candidates=b)
        lists = (
            cand.sort_values(["customer_key", "rrf_score"], ascending=[True, False])
            .groupby("customer_key")["article_id"]
            .apply(list)
            .to_dict()
        )
        rec = recall_at_k(actual, [lists.get(c, []) for c in customers], b)

        preds = predict_week(
            con, booster, features, as_of, customers, n_candidates=b, chunk=chunk
        )
        # One `recall` column, not `recall@{b}`: a per-row column name produces one column per
        # budget and fills the rest of the table with NaN. The cutoff is already the budget column.
        rows.append(
            {
                "budget": b,
                "recall@budget": round(rec, 5),
                "MAP@12": round(mapk(actual, [preds.get(c, []) for c in customers], K), 5),
            }
        )
        print(f"  budget={b}: {rows[-1]}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=3, help="validation weeks, ending at VAL_START")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument(
        "--chunk", type=int, default=4000,
        help="customers scored per batch; lower it if the process is OOM-killed",
    )
    args = parser.parse_args()

    con = connect()
    booster = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
    features = json.loads((MODELS / "features.json").read_text())

    weeks = [VAL_START - timedelta(days=7 * i) for i in range(args.weeks)][::-1]
    print(f"rolling evaluation over {weeks}")
    rolling = pd.DataFrame(
        [evaluate_week(con, booster, features, w, chunk=args.chunk) for w in weeks]
    )
    print("\n" + rolling.to_string(index=False))

    print(f"\ncandidate budget sweep on {VAL_START} ({args.sample:,} sampled customers):")
    sweep = budget_sweep(
        con, booster, features, VAL_START, args.budgets, args.sample, chunk=args.chunk
    )
    print(sweep.to_string(index=False))

    REPORTS.mkdir(exist_ok=True)
    write_report(
        REPORTS / "rolling.md",
        f"# Rolling evaluation\n\n"
        f"One model, scored on consecutive weeks without retraining.\n\n"
        f"Only the row for `{VAL_START}` is a valid generalisation estimate. It is the last\n"
        f"week in the dataset, so it is the only week the model can be scored on without having\n"
        f"been fitted on later data — which is why there is exactly one.\n\n"
        f"Every earlier row is contaminated, in one of two ways: a week inside the training\n"
        f"range is straightforwardly in-sample, and a week *before* the training range was\n"
        f"scored by a model fitted on information from after it. Neither simulates production.\n"
        f"They are reported because the size of the in-sample gap is itself diagnostic: a large\n"
        f"gap means overfitting, and watching it shrink across model versions is how the\n"
        f"multi-week training decision was justified (see README).\n\n"
        + rolling.to_markdown(index=False)
        + "\n\n## Candidate budget\n\n"
        + f"Sampled {args.sample:,} customers from the {VAL_START} validation week.\n\n"
        + sweep.to_markdown(index=False)
        + "\n"
    )
    print(f"\nwrote {REPORTS / 'rolling.md'}")


if __name__ == "__main__":
    main()
