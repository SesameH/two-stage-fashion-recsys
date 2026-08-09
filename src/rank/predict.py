"""Scoring a week's candidates with a trained ranker.

Separate from `train.py` so that evaluation and error analysis can score a model without
importing the training entry point, its hyperparameters, or its argument parser.
"""

from __future__ import annotations

from datetime import date

import lightgbm as lgb
import pandas as pd

from src.config import N_CANDIDATES, K
from src.data.db import register_customers
from src.features.builder import build_features
from src.recall.pipeline import build_candidates


def predict_week(
    con,
    model: lgb.LGBMRanker,
    features: list[str],
    as_of: date,
    customers: list[int],
    als_model=None,
    chunk: int = 8000,
    n_candidates: int = N_CANDIDATES,
) -> dict[int, list[int]]:
    """Score every candidate for `customers`, in chunks, and keep the top K per customer.

    Chunking is a memory decision, not a modelling one: 69k customers x 300 candidates x ~45
    features does not fit comfortably in RAM at once, and nothing about the model depends on
    which chunk a customer lands in.
    """
    out: dict[int, list[int]] = {}
    for i in range(0, len(customers), chunk):
        part = customers[i : i + chunk]
        register_customers(con, part)
        cand = build_candidates(con, as_of, part, als_model=als_model,
                                n_candidates=n_candidates)
        feats = build_features(con, cand, as_of)
        scores = model.predict(feats[features])
        ranked = (
            pd.DataFrame(
                {
                    "customer_key": feats["customer_key"].to_numpy(),
                    "article_id": feats["article_id"].to_numpy(),
                    "score": scores,
                }
            )
            .sort_values(["customer_key", "score"], ascending=[True, False], kind="stable")
            .groupby("customer_key")["article_id"]
            .apply(lambda s: s.head(K).tolist())
        )
        out.update({int(c): v for c, v in ranked.items()})
        print(f"    scored {min(i + chunk, len(customers)):,}/{len(customers):,}", flush=True)
    return out
