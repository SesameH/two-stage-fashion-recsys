"""FastAPI recommendation service.

Run: uvicorn src.serve.app:app --port 8000

Request path:
    customer_id (hex string)
      -> customer_key
      -> retrieval from precomputed tables (repurchase, variants, popularity)
      -> feature assembly via the SAME builder used offline
      -> LightGBM scoring
      -> top 12

Everything that does not depend on the request is loaded once at startup and held in memory:
the precomputed tables are small enough (a few hundred MB) that per-request work is a
dictionary lookup plus one model call. `/debug/features` exposes the exact feature row the
model saw, which is what tests/test_parity.py compares against the offline pipeline.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import MODELS, N_CANDIDATES, K
from src.features.builder import MISSING_RANK, MISSING_SCORE, build_features_serving
from src.recall.base import RRF_K
from src.serve.precompute import SERVING

LATENCIES: deque[float] = deque(maxlen=10000)

# Model features the request path could not produce and had to fill with an absence sentinel.
# This should stay empty: a feature the snapshot cannot supply is a feature the model was
# trained on and never sees in production. It is surfaced on /metrics rather than left silent,
# and tests/test_parity.py asserts it is empty after a real request.
UNSERVABLE: set[str] = set()


class State:
    con: duckdb.DuckDBPyConnection
    booster: lgb.Booster
    features: list[str]
    as_of: date
    customer_ids: dict[str, int]
    mode: str


state = State()


def _load(out: Path = SERVING) -> None:
    meta = json.loads((out / "meta.json").read_text())
    state.as_of = date.fromisoformat(meta["as_of"])
    state.booster = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
    state.features = json.loads((MODELS / "features.json").read_text())

    # memory: tables in RAM with customer_key indexes — 3.1 GB resident, p50 39 ms locally.
    # parquet: views over the files, row-group pruning instead of indexes — 834 MB, p50 63 ms,
    # same p99, 7 s cold start instead of 55 s.
    # parquet wins on a laptop and loses on Cloud Run, where the container filesystem is a
    # network-backed overlay (147 ms vs 294 ms p50 in favour of memory). The deployment
    # therefore runs `memory`; see reports/serving.md for both measurements.
    mode = os.environ.get("HM_SERVE_MODE", "memory")
    if mode not in ("memory", "parquet"):
        raise ValueError(f"HM_SERVE_MODE must be 'memory' or 'parquet', got {mode!r}")

    con = duckdb.connect()
    for name in (
        "customer_features",
        "article_features",
        "affinity",
        "customer_category",
        "customer_productcode",
        "repurchase",
        "variants",
        "popularity_global",
        "popularity_age",
        "category_candidates",
        "articles",
        "customers",
    ):
        kind = "TABLE" if mode == "memory" else "VIEW"
        con.execute(
            f"CREATE {kind} {name} AS "
            f"SELECT * FROM read_parquet('{out / (name + '.parquet')}')"
        )
    # Without these, a request full-scans a multi-million-row table. Views cannot be indexed,
    # so parquet mode leans on DuckDB's row-group statistics instead.
    if mode == "memory":
        for table in ("repurchase", "variants", "customer_features", "affinity",
                      "customer_category", "customer_productcode", "category_candidates"):
            con.execute(f"CREATE INDEX idx_{table} ON {table}(customer_key)")
    state.con = con
    state.mode = mode

    id_map = duckdb.connect().execute(
        f"SELECT customer_id, customer_key FROM "
        f"read_parquet('{SERVING.parent / 'processed' / 'customer_id_map.parquet'}')"
    ).df()
    state.customer_ids = dict(zip(id_map["customer_id"], id_map["customer_key"]))
    con.register("id_map_df", id_map)
    con.execute("CREATE TABLE id_map AS SELECT * FROM id_map_df")  # small, always in memory
    con.unregister("id_map_df")
    con.execute("CREATE INDEX idx_id_map ON id_map(customer_key)")

    for t in DEMO_TABLES:
        path = out / f"{t}.parquet"
        if path.exists():
            con.execute(
                f"CREATE {'TABLE' if mode == 'memory' else 'VIEW'} {t} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            if mode == "memory" and t in ("demo_truth", "demo_history", "demo_segments"):
                con.execute(f"CREATE INDEX idx_{t} ON {t}(customer_key)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="H&M two-stage recommender", lifespan=lifespan)


class Recommendation(BaseModel):
    customer_id: str
    articles: list[int]
    n_candidates: int
    latency_ms: float


def retrieve(customer_key: int, n: int = N_CANDIDATES) -> pd.DataFrame:
    """Precomputed retrieval, fused by RRF — the same fusion the offline pipeline uses."""
    return state.con.execute(
        """
        WITH age AS (
            SELECT CAST(floor(coalesce(age, 30) / 10) AS INTEGER) AS bucket
            FROM customers WHERE customer_key = $ck
        ),
        cand AS (
            SELECT article_id, 'r1_repurchase' AS source, rank, score FROM repurchase
              WHERE customer_key = $ck
            UNION ALL
            SELECT article_id, 'r6_product_variant', rank, score FROM variants
              WHERE customer_key = $ck
            UNION ALL
            SELECT article_id, 'r2b_global', rank, score FROM popularity_global
            UNION ALL
            SELECT p.article_id, 'r2_popularity', p.rank, p.score
              FROM popularity_age p JOIN age a ON a.bucket = p.age_bucket
            UNION ALL
            SELECT c.article_id, 'r5_category', c.rank, c.score FROM category_candidates c
              WHERE c.customer_key = $ck
        )
        SELECT article_id,
               sum(1.0 / ($rrf + rank))    AS rrf_score,
               count(DISTINCT source)      AS n_sources,
               coalesce(max(CASE WHEN source = 'r1_repurchase' THEN rank END), $miss)
                   AS rank_r1_repurchase,
               coalesce(max(CASE WHEN source = 'r1_repurchase' THEN score END), 0.0)
                   AS score_r1_repurchase,
               coalesce(max(CASE WHEN source = 'r2_popularity' THEN rank END), $miss)
                   AS rank_r2_popularity,
               coalesce(max(CASE WHEN source = 'r2_popularity' THEN score END), 0.0)
                   AS score_r2_popularity,
               coalesce(max(CASE WHEN source = 'r2b_global' THEN rank END), $miss)
                   AS rank_r2b_global,
               coalesce(max(CASE WHEN source = 'r2b_global' THEN score END), 0.0)
                   AS score_r2b_global,
               coalesce(max(CASE WHEN source = 'r5_category' THEN rank END), $miss)
                   AS rank_r5_category,
               coalesce(max(CASE WHEN source = 'r5_category' THEN score END), 0.0)
                   AS score_r5_category,
               coalesce(max(CASE WHEN source = 'r6_product_variant' THEN rank END), $miss)
                   AS rank_r6_product_variant,
               coalesce(max(CASE WHEN source = 'r6_product_variant' THEN score END), 0.0)
                   AS score_r6_product_variant
        FROM cand
        GROUP BY 1
        ORDER BY rrf_score DESC
        LIMIT $n
        """,
        {"ck": customer_key, "rrf": RRF_K, "n": n, "miss": MISSING_RANK},
    ).df()


def build_request_features(customer_key: int, n_candidates: int = N_CANDIDATES) -> pd.DataFrame:
    """Candidates + features for one customer, through the shared assemble path."""
    cand = retrieve(customer_key, n_candidates)
    if cand.empty:
        return cand
    cand.insert(0, "customer_key", customer_key)
    feats = build_features_serving(state.con, cand, customer_key)
    # A feature the snapshot cannot supply is encoded the way the offline pipeline encodes "not
    # proposed by this source", so the two paths at least agree on what absence looks like. It is
    # still skew — the model was trained on a column that is now constant — so the names are
    # recorded and reported on /metrics instead of being patched silently.
    for f in state.features:
        if f not in feats.columns:
            feats[f] = MISSING_RANK if f.startswith("rank_") else MISSING_SCORE
            UNSERVABLE.add(f)
    return feats


def recommend(customer_key: int, n_candidates: int = N_CANDIDATES) -> tuple[list[int], int]:
    feats = build_request_features(customer_key, n_candidates)
    if feats.empty:
        return [], 0
    scores = state.booster.predict(feats[state.features])
    order = np.argsort(-scores)[:K]
    return feats["article_id"].to_numpy()[order].tolist(), len(feats)


@app.get("/recommend/{customer_id}", response_model=Recommendation)
def recommend_endpoint(customer_id: str, n_candidates: int = Query(N_CANDIDATES, ge=12, le=1000)):
    key = state.customer_ids.get(customer_id)
    if key is None:
        raise HTTPException(404, f"unknown customer_id {customer_id[:16]}...")

    t0 = time.perf_counter()
    articles, n = recommend(key, n_candidates)
    elapsed = (time.perf_counter() - t0) * 1000
    LATENCIES.append(elapsed)

    return Recommendation(
        customer_id=customer_id, articles=articles, n_candidates=n, latency_ms=round(elapsed, 2)
    )


@app.get("/debug/features/{customer_id}")
def debug_features(customer_id: str):
    """The exact feature rows the model scored. Used by the offline/online parity test."""
    key = state.customer_ids.get(customer_id)
    if key is None:
        raise HTTPException(404, "unknown customer_id")
    return json.loads(build_request_features(key).to_json(orient="records"))


# --- Evaluation console -----------------------------------------------------
#
# The `demo_*` tables include purchases from on or after the cutoff — a console that cannot show
# whether a prediction was right is just a list. Separate prefix, separate tables, never read by
# `recommend()`.

DEMO_TABLES = ("demo_truth", "demo_history", "demo_articles", "demo_segments")


def _demo_available() -> bool:
    return all((SERVING / f"{t}.parquet").exists() for t in DEMO_TABLES)


@app.get("/api/demo/customers")
def demo_customers(segment: str | None = None, n: int = Query(20, ge=1, le=200)):
    """A sample of evaluation-week buyers, optionally restricted to one activity segment."""
    if not _demo_available():
        raise HTTPException(503, "demo tables not built; run precompute --with-demo")
    where = "WHERE recency_segment = $seg" if segment else ""
    rows = state.con.execute(
        f"""
        SELECT s.customer_key, m.customer_id, s.recency_segment, s.n_prior, s.recency, s.age
        FROM demo_segments s
        JOIN id_map m USING (customer_key)
        {where}
        ORDER BY random()
        LIMIT $n
        """,
        {"seg": segment, "n": n} if segment else {"n": n},
    ).df()
    return json.loads(rows.to_json(orient="records"))


@app.get("/api/demo/segments")
def demo_segments():
    rows = state.con.execute(
        "SELECT recency_segment, count(*) AS customers FROM demo_segments "
        "GROUP BY 1 ORDER BY 1"
    ).df()
    return json.loads(rows.to_json(orient="records"))


@app.get("/api/demo/explain/{customer_id}")
def demo_explain(customer_id: str):
    """Everything the console shows for one customer, in one round trip.

    Includes the model's top 12 with per-source attribution, the B2 baseline's top 12 for the
    same customer, both marked against ground truth, and the measured latency of the model call.
    """
    if not _demo_available():
        raise HTTPException(503, "demo tables not built; run precompute --with-demo")
    key = state.customer_ids.get(customer_id)
    if key is None:
        raise HTTPException(404, "unknown customer_id")

    t0 = time.perf_counter()
    feats = build_request_features(key)
    if feats.empty:
        raise HTTPException(404, "no candidates for this customer")
    scores = state.booster.predict(feats[state.features])
    latency_ms = (time.perf_counter() - t0) * 1000
    LATENCIES.append(latency_ms)

    order = np.argsort(-scores)[:K]
    top = feats.iloc[order].copy()
    top["score"] = scores[order]

    truth = state.con.execute(
        "SELECT article_id, n FROM demo_truth WHERE customer_key = ?", [key]
    ).df()
    truth_ids = set(truth["article_id"].tolist())

    history = state.con.execute(
        """
        SELECT h.article_id, h.t_dat, h.price, a.prod_name, a.product_type_name,
               a.colour_group_name
        FROM demo_history h LEFT JOIN demo_articles a USING (article_id)
        WHERE h.customer_key = ? ORDER BY h.t_dat DESC LIMIT 15
        """,
        [key],
    ).df()

    meta = state.con.execute("SELECT * FROM demo_articles").df().set_index("article_id")

    def describe(article_id: int, rank: int, score: float | None, sources: list[str]):
        m = meta.loc[article_id] if article_id in meta.index else None
        return {
            "rank": rank,
            "article_id": int(article_id),
            "prod_name": None if m is None else m["prod_name"],
            "product_type": None if m is None else m["product_type_name"],
            "colour": None if m is None else m["colour_group_name"],
            "product_group": None if m is None else m["product_group_name"],
            "score": None if score is None else round(float(score), 4),
            "sources": sources,
            "hit": int(article_id) in truth_ids,
        }

    source_cols = [c for c in top.columns if c.startswith("rank_")]
    model_rows = []
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        srcs = [
            c.removeprefix("rank_")
            for c in source_cols
            if row[c] != MISSING_RANK and not pd.isna(row[c])
        ]
        model_rows.append(describe(row["article_id"], i, row["score"], srcs))

    b2_ids = state.con.execute(
        """
        WITH rep AS (
            SELECT article_id, rank FROM repurchase WHERE customer_key = $ck ORDER BY rank LIMIT $k
        ),
        fill AS (
            SELECT article_id, rank + 1000 AS rank FROM popularity_global
            WHERE article_id NOT IN (SELECT article_id FROM rep) ORDER BY rank
        )
        SELECT article_id FROM (SELECT * FROM rep UNION ALL SELECT * FROM fill)
        ORDER BY rank LIMIT $k
        """,
        {"ck": key, "k": K},
    ).df()["article_id"].tolist()

    model_hits = sum(r["hit"] for r in model_rows)
    b2_rows = [describe(a, i, None, ["b2"]) for i, a in enumerate(b2_ids, start=1)]
    b2_hits = sum(r["hit"] for r in b2_rows)

    return {
        "customer_id": customer_id,
        "as_of": state.as_of.isoformat(),
        "latency_ms": round(latency_ms, 2),
        "n_candidates": len(feats),
        "history": json.loads(history.to_json(orient="records")),
        "truth": {
            "n_articles": len(truth_ids),
            "article_ids": sorted(int(a) for a in truth_ids),
        },
        "model": {
            "top": model_rows,
            "hits": int(model_hits),
            "ap12": round(_ap(model_rows, truth_ids), 4),
        },
        "b2": {"top": b2_rows, "hits": int(b2_hits), "ap12": round(_ap(b2_rows, truth_ids), 4)},
    }


def _ap(rows: list[dict], truth_ids: set) -> float:
    """AP@12 for one customer, on the same definition the offline metric uses.

    The ground truth must be the customer's full purchase set, not the subset that happens to
    have been predicted. `apk` divides by `min(len(actual), 12)`, so passing only the hits makes
    every single-hit-at-rank-1 customer score a perfect 1.0 — the console reported AP that way
    and it flattered both rows.
    """
    from src.eval.metrics import apk

    return apk(truth_ids, [r["article_id"] for r in rows], K)


@app.get("/metrics")
def metrics():
    # Service facts are reported whether or not anything has been served yet; they describe the
    # deployment, not the traffic. Omitting them on an idle instance left the console rendering
    # "as_of undefined" on a cold page load.
    out = {
        "as_of": state.as_of.isoformat(),
        "n_trees": state.booster.num_trees(),
        "serve_mode": state.mode,
        "requests": len(LATENCIES),
        "unservable_features": sorted(UNSERVABLE),
    }
    if LATENCIES:
        arr = np.array(LATENCIES)
        out |= {
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(arr.max()), 2),
        }
    return out


@app.get("/health")
def health():
    return {"status": "ok", "as_of": state.as_of.isoformat()}


STATIC = Path(__file__).parent / "static"

# Images are optional and not part of a public deployment; the console degrades to
# product-type tiles when a file is missing.
IMAGES = SERVING / "images"
if IMAGES.is_dir():
    app.mount("/images", StaticFiles(directory=IMAGES), name="images")


@app.get("/", include_in_schema=False)
def console():
    return FileResponse(STATIC / "index.html")
