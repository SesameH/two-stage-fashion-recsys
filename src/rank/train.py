"""Train and evaluate the LightGBM LambdaRank re-ranker.

Run: python -m src.rank.train

Train week and validation week are strictly separated in time. Validation is scored in
customer chunks, without negative downsampling, over the full evaluation population, so the
MAP@12 printed here is directly comparable to the baseline ladder.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import lightgbm as lgb
import pandas as pd

from src.config import (
    MODELS,
    NEG_SAMPLE_RATIO,
    REPORTS,
    TRAIN_START,
    VAL_START,
    K,
)
from src.data.db import connect
from src.eval.metrics import mapk, ndcg_at_k, recall_at_k
from src.eval.split import ground_truth
from src.features.builder import feature_columns
from src.rank.dataset import (
    build_multiweek_training_set,
    build_training_set,
    group_sizes,
)
from src.rank.predict import predict_week

PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [K],
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-as-of", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--val-as-of", type=date.fromisoformat, default=VAL_START)
    parser.add_argument("--with-als", action="store_true", help="add R4 ALS candidates (slow)")
    parser.add_argument("--n-estimators", type=int, default=PARAMS["n_estimators"])
    parser.add_argument(
        "--train-weeks", type=int, default=1,
        help="number of consecutive training weeks, ending at --train-as-of",
    )
    parser.add_argument(
        "--neg-ratio", type=int, default=None,
        help="negatives per positive in training; also sets the LambdaRank group size",
    )
    parser.add_argument("--out-suffix", type=str, default="", help="suffix for model artefacts")
    args = parser.parse_args()

    con = connect()
    t0 = time.perf_counter()

    als_train = als_val = None
    if args.with_als:
        from src.baselines.als import fit

        # One model per cutoff. Reusing a single ALS across both weeks would let validation
        # candidates be generated from factors that saw the validation week.
        als_train = fit(con, args.train_as_of)
        als_val = fit(con, args.val_as_of)
        print(f"ALS fitted ({time.perf_counter() - t0:.0f}s)")

    neg_ratio = args.neg_ratio if args.neg_ratio is not None else NEG_SAMPLE_RATIO
    weeks = [args.train_as_of - timedelta(days=7 * i) for i in range(args.train_weeks)][::-1]
    print(f"building training set for {len(weeks)} week(s): {[w.isoformat() for w in weeks]}")
    if len(weeks) == 1:
        train = build_training_set(
            con, weeks[0], als_model=als_train, downsample=True, neg_ratio=neg_ratio
        )
    else:
        train = build_multiweek_training_set(
            con, weeks, als_model=als_train, downsample=True, neg_ratio=neg_ratio
        )
    features = [f for f in feature_columns(train) if f != "week"]
    print(
        f"  {len(train):,} rows, {train['label'].sum():,} positives "
        f"({train['label'].mean():.2%}), {len(features)} features, "
        f"{train['customer_key'].nunique():,} customers"
    )

    model = lgb.LGBMRanker(**{**PARAMS, "n_estimators": args.n_estimators})
    model.fit(
        train[features],
        train["label"],
        group=group_sizes(train),
        categorical_feature=[c for c in features if str(train[c].dtype) == "category"],
    )
    print(f"  trained ({time.perf_counter() - t0:.0f}s)")

    gt = ground_truth(con, args.val_as_of)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    customers = sorted(truth)
    print(f"scoring validation week {args.val_as_of}, {len(customers):,} customers ...")
    preds = predict_week(con, model, features, args.val_as_of, customers, als_model=als_val)

    actual = [truth[c] for c in customers]
    predicted = [preds.get(c, []) for c in customers]
    result = {
        "MAP@12": round(mapk(actual, predicted, K), 5),
        "recall@12": round(recall_at_k(actual, predicted, K), 5),
        "NDCG@12": round(ndcg_at_k(actual, predicted, K), 5),
        "coverage": round(sum(1 for p in predicted if p) / len(predicted), 4),
        "n_features": len(features),
        "train_rows": len(train),
        "train_weeks": [w.isoformat() for w in weeks],
        "neg_ratio": neg_ratio,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    print("\n" + json.dumps(result, indent=2))

    MODELS.mkdir(exist_ok=True)
    sfx = args.out_suffix
    model.booster_.save_model(str(MODELS / f"ranker{sfx}.txt"))
    (MODELS / f"features{sfx}.json").write_text(json.dumps(features, indent=2))

    imp = (
        pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"ranker{sfx}.md").write_text(
        f"# Ranking stage\n\n"
        f"Train weeks `{[w.isoformat() for w in weeks]}`, "
        f"validation week `{args.val_as_of}`.\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n\n"
        f"## Feature importance (gain, top 30)\n\n{imp.head(30).to_markdown(index=False)}\n"
    )
    print("\ntop features by gain:")
    print(imp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
