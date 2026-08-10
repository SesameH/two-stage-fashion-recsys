"""Choose the tree count on a week that is neither trained on nor reported.

Run: python -m src.rank.tune_trees

`n_estimators` was a guess for a long time, and the compression curve in reports/serving.md made
it look like an underfitting guess. Settling it on the validation week would have turned the
reported MAP@12 into a tuned number, so the protocol uses three disjoint windows:

    train on the weeks ending `--train-as-of`   (default 4 weeks ending 2020-09-02)
    select the tree count on `--select-as-of`   (default 2020-09-09)
    report on VAL_START                         (2020-09-16, untouched here)

One model is fitted at `--max-trees` and then truncated at predict time via `num_iteration`, so
every row of the sweep comes from the same booster — a tree count is a prefix of a GBDT, which is
what makes this cheap enough to be worth doing properly.

`--bagging-by-query` tests LightGBM's ranking-aware bagging. Row-level bagging is the default and
ignores query boundaries, so with `lambdarank` each tree sees a mutilated candidate list per
customer rather than fewer whole customers.

Writes reports/tuning.md.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta

import lightgbm as lgb
import pandas as pd

from src.config import MODELS, NEG_SAMPLE_RATIO, REPORTS, K, write_report
from src.data.db import connect, register_customers
from src.eval.metrics import mapk
from src.eval.split import ground_truth
from src.features.builder import build_features, feature_columns
from src.rank.dataset import build_multiweek_training_set, build_training_set, group_sizes
from src.rank.train import PARAMS
from src.recall.pipeline import build_candidates


def sweep(
    con,
    booster: lgb.Booster,
    features: list[str],
    as_of: date,
    customers: list[int],
    trees: list[int],
    chunk: int = 3000,
) -> pd.DataFrame:
    """MAP@12 at each tree count, on candidates built once per chunk."""
    truth = {int(r.customer_key): list(r.articles) for r in ground_truth(con, as_of).itertuples()}
    preds: dict[int, dict[int, list[int]]] = {n: {} for n in trees}

    for i in range(0, len(customers), chunk):
        part = customers[i : i + chunk]
        register_customers(con, part)
        feats = build_features(con, build_candidates(con, as_of, part), as_of)
        X = feats[features]
        keys = feats["customer_key"].to_numpy()
        articles = feats["article_id"].to_numpy()
        for n in trees:
            ranked = (
                pd.DataFrame(
                    {"customer_key": keys, "article_id": articles,
                     "score": booster.predict(X, num_iteration=n)}
                )
                .sort_values(["customer_key", "score"], ascending=[True, False], kind="stable")
                .groupby("customer_key")["article_id"]
                .apply(lambda s: s.head(K).tolist())
            )
            preds[n].update({int(c): v for c, v in ranked.items()})
        print(f"  {min(i + chunk, len(customers)):,}/{len(customers):,}", flush=True)

    actual = [truth[c] for c in customers]
    rows = [
        {"trees": n, "MAP@12": round(mapk(actual, [preds[n].get(c, []) for c in customers], K), 5)}
        for n in trees
    ]
    out = pd.DataFrame(rows)
    best = out.loc[out["MAP@12"].idxmax(), "trees"]
    out["vs_best"] = (out["MAP@12"] / out["MAP@12"].max() - 1).round(4)
    print(f"  best: {best} trees")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-as-of", type=date.fromisoformat, default=date(2020, 9, 2))
    parser.add_argument("--train-weeks", type=int, default=4)
    parser.add_argument("--select-as-of", type=date.fromisoformat, default=date(2020, 9, 9))
    parser.add_argument("--max-trees", type=int, default=1500)
    parser.add_argument(
        "--trees", type=int, nargs="+", default=[100, 250, 500, 750, 1000, 1250, 1500]
    )
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument("--bagging-by-query", action="store_true")
    args = parser.parse_args()

    if args.select_as_of <= args.train_as_of:
        raise SystemExit(
            f"--select-as-of ({args.select_as_of}) must be after the last training week "
            f"({args.train_as_of}); otherwise the tree count is chosen in-sample"
        )

    con = connect()
    weeks = [args.train_as_of - timedelta(days=7 * i) for i in range(args.train_weeks)][::-1]
    print(f"training on {[w.isoformat() for w in weeks]} at {args.max_trees} trees")
    train = (
        build_training_set(con, weeks[0], neg_ratio=NEG_SAMPLE_RATIO)
        if len(weeks) == 1
        else build_multiweek_training_set(con, weeks, neg_ratio=NEG_SAMPLE_RATIO)
    )
    features = [f for f in feature_columns(train) if f != "week"]

    params = {**PARAMS, "n_estimators": args.max_trees}
    if args.bagging_by_query:
        params["bagging_by_query"] = True
    model = lgb.LGBMRanker(**params)
    model.fit(
        train[features],
        train["label"],
        group=group_sizes(train),
        categorical_feature=[c for c in features if str(train[c].dtype) == "category"],
    )
    del train

    truth = ground_truth(con, args.select_as_of)
    random.seed(0)
    keys = sorted(random.sample(sorted(truth["customer_key"].astype(int)), args.sample))
    print(f"selecting on {args.select_as_of}, {len(keys):,} customers")
    table = sweep(con, model.booster_, features, args.select_as_of, keys, args.trees)
    print(table.to_string(index=False))

    MODELS.mkdir(exist_ok=True)
    suffix = "_bq" if args.bagging_by_query else "_tune"
    model.booster_.save_model(str(MODELS / f"ranker{suffix}.txt"))

    REPORTS.mkdir(exist_ok=True)
    write_report(
        REPORTS / "tuning.md",
        "# Tree count\n\n"
        f"Trained on {[w.isoformat() for w in weeks]}, tree count selected on "
        f"`{args.select_as_of}`, reported on the validation week elsewhere. "
        f"{len(keys):,} customers, `bagging_by_query={bool(args.bagging_by_query)}`.\n\n"
        "The selection week is disjoint from both the training weeks and the validation week, so\n"
        "choosing a tree count here does not tune the number the README quotes.\n\n"
        + table.to_markdown(index=False)
        + "\n",
    )
    print(json.dumps({"best_trees": int(table.loc[table["MAP@12"].idxmax(), "trees"])}))


if __name__ == "__main__":
    main()
