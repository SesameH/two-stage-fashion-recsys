"""B3: implicit-feedback matrix factorisation (ALS).

Doubles as retrieval strategy R4 — the trained item factors are what the FAISS index will
hold at serving time, so this module returns the factor matrices, not just recommendations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np
import scipy.sparse as sp

# implicit spawns BLAS threads that fight its own OpenMP pool; the library warns about this.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from implicit.als import AlternatingLeastSquares


@dataclass
class ALSModel:
    model: AlternatingLeastSquares
    matrix: sp.csr_matrix
    customer_keys: np.ndarray  # row index -> customer_key
    article_ids: np.ndarray  # col index -> article_id
    customer_pos: dict[int, int]

    def recommend(self, customers: list[int], k: int = 12) -> dict[int, list[int]]:
        """Top-k articles per customer, already-bought items filtered out by implicit."""
        known = [c for c in customers if c in self.customer_pos]
        if not known:
            return {}
        rows = np.array([self.customer_pos[c] for c in known], dtype=np.int32)
        ids, _ = self.model.recommend(
            rows, self.matrix[rows], N=k, filter_already_liked_items=False
        )
        return {c: self.article_ids[ids[i]].tolist() for i, c in enumerate(known)}


def build_matrix(
    con: duckdb.DuckDBPyConnection, as_of: date, lookback_days: int = 180
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Customer x article interaction counts over [as_of - lookback, as_of).

    Confidence is the raw purchase count; ALS's own alpha scaling handles the rest.
    """
    df = con.execute(
        """
        SELECT customer_key, article_id, count(*) AS n
        FROM transactions
        WHERE t_dat < CAST(? AS DATE)
          AND t_dat >= CAST(? AS DATE) - CAST(? AS INTEGER) * INTERVAL 1 DAY
        GROUP BY customer_key, article_id
        """,
        [as_of, as_of, lookback_days],
    ).df()

    customer_keys, rows = np.unique(df["customer_key"].to_numpy(), return_inverse=True)
    article_ids, cols = np.unique(df["article_id"].to_numpy(), return_inverse=True)
    matrix = sp.csr_matrix(
        (df["n"].to_numpy(dtype=np.float32), (rows, cols)),
        shape=(len(customer_keys), len(article_ids)),
    )
    return matrix, customer_keys, article_ids


def fit(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    lookback_days: int = 365,
    factors: int = 128,
    regularization: float = 0.05,
    alpha: float = 40.0,
    iterations: int = 15,
    seed: int = 42,
) -> ALSModel:
    """Defaults are the best of the sweep recorded in reports/baselines.md.

    That sweep spans 0.0065-0.0094 MAP@12 — the whole plausible hyperparameter range sits an
    order of magnitude below B2, so ALS is used as a retrieval source, not as a ranker.
    """
    matrix, customer_keys, article_ids = build_matrix(con, as_of, lookback_days)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
    )
    model.fit(matrix, show_progress=False)
    return ALSModel(
        model=model,
        matrix=matrix,
        customer_keys=customer_keys,
        article_ids=article_ids,
        customer_pos={int(c): i for i, c in enumerate(customer_keys)},
    )
