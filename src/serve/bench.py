"""Latency benchmark and the model-compression trade-off curve.

Run: python -m src.serve.bench            (in-process, no HTTP)
     python -m src.serve.bench --http     (against a running uvicorn)

Two numbers this produces that a modelling-only project cannot:

  1. p50/p95/p99 of the full request path at several candidate budgets.
  2. What accuracy costs what latency — tree count is truncated, then both MAP@12 and p99 are
     re-measured, so the trade-off is measured rather than assumed.

In-process mode is the default because it isolates the recommender from the HTTP stack; the
`--http` mode adds the server and client overhead back so the two can be compared.

This rewrites `reports/serving.md` above `config.MANUAL_MARKER` only. Everything below it —
container and Cloud Run measurements, which cannot be produced from a laptop process — is
preserved. Before that marker existed, `make bench` deleted them.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd

from src.config import REPORTS, K, write_report
from src.eval.metrics import mapk


def percentiles(samples: list[float]) -> dict[str, float]:
    arr = np.array(samples)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def bench_inprocess(customer_keys: list[int], n_candidates: int, warmup: int = 20) -> dict:
    from src.serve import app as srv

    for ck in customer_keys[:warmup]:
        srv.recommend(ck, n_candidates)

    samples, sizes = [], []
    for ck in customer_keys:
        t = time.perf_counter()
        _, n = srv.recommend(ck, n_candidates)
        samples.append((time.perf_counter() - t) * 1000)
        sizes.append(n)

    return {
        "n_candidates": n_candidates,
        "requests": len(samples),
        "avg_candidates": round(float(np.mean(sizes)), 1),
        **percentiles(samples),
    }


def bench_http(customer_ids: list[str], base: str, n_candidates: int) -> dict:
    import urllib.request

    samples = []
    for cid in customer_ids:
        url = f"{base}/recommend/{cid}?n_candidates={n_candidates}"
        t = time.perf_counter()
        with urllib.request.urlopen(url) as r:
            r.read()
        samples.append((time.perf_counter() - t) * 1000)
    return {"n_candidates": n_candidates, "requests": len(samples), **percentiles(samples)}


def parity(customer_keys: list[int], truth: dict[int, list[int]]) -> dict:
    """Score the same customers through the offline pipeline and the request path.

    This used to be a hand-run comparison whose numbers appeared in the README with no command
    behind them. It is measured here so the claim moves or breaks when the code does.

    The two paths read different storage by design, so equality is the thing worth asserting:
    identical top 12 for every customer means the snapshot, the retrieval budgets and the feature
    blocks all still agree.
    """
    from src.config import VAL_START
    from src.data.db import connect, register_customers
    from src.features.builder import build_features
    from src.recall.pipeline import build_candidates
    from src.serve import app as srv

    online = {ck: srv.recommend(ck)[0] for ck in customer_keys}

    con = connect()
    register_customers(con, list(customer_keys))
    cand = build_candidates(con, VAL_START, list(customer_keys))
    feats = build_features(con, cand, VAL_START)
    con.close()

    scores = srv.state.booster.predict(feats[srv.state.features])
    frame = pd.DataFrame(
        {
            "customer_key": feats["customer_key"].to_numpy(),
            "article_id": feats["article_id"].to_numpy(),
            "score": scores,
        }
    ).sort_values(["customer_key", "score"], ascending=[True, False], kind="stable")
    offline = {
        int(c): v[:K]
        for c, v in frame.groupby("customer_key")["article_id"].apply(list).items()
    }

    scored = [c for c in customer_keys if c in truth]
    overlaps = [
        len(set(offline.get(c, [])) & set(online.get(c, []))) / K for c in customer_keys
    ]
    return {
        "customers": len(customer_keys),
        "MAP@12 offline": round(mapk([truth[c] for c in scored], [offline.get(c, []) for c in scored], K), 5),
        "MAP@12 online": round(mapk([truth[c] for c in scored], [online.get(c, []) for c in scored], K), 5),
        "mean top-12 overlap": round(float(np.mean(overlaps)), 4),
        "identical top 12": round(
            sum(offline.get(c, []) == online.get(c, []) for c in customer_keys) / len(customer_keys), 4
        ),
    }


def compression_curve(customer_keys: list[int], truth: dict[int, list[int]], trees: list[int]):
    """Truncate the booster to fewer trees; measure latency and MAP@12 at each size.

    `num_iteration` on predict is a real deployment lever, not a simulation: the same booster
    file serves every row of this table.
    """
    from src.serve import app as srv

    full = srv.state.booster.num_trees()
    rows = []
    for n_trees in trees:
        preds, samples = {}, []
        for ck in customer_keys:
            t = time.perf_counter()
            feats = srv.build_request_features(ck)
            if feats.empty:
                preds[ck] = []
            else:
                scores = srv.state.booster.predict(
                    feats[srv.state.features], num_iteration=n_trees
                )
                order = np.argsort(-scores)[:K]
                preds[ck] = feats["article_id"].to_numpy()[order].tolist()
            samples.append((time.perf_counter() - t) * 1000)

        scored = [c for c in customer_keys if c in truth]
        rows.append(
            {
                "trees": n_trees,
                "pct_of_full": round(100 * n_trees / full),
                "MAP@12": round(mapk([truth[c] for c in scored], [preds[c] for c in scored], K), 5),
                **percentiles(samples),
            }
        )
        print(f"  trees={n_trees}: {rows[-1]}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="customers to sample")
    parser.add_argument("--http", metavar="BASE_URL", nargs="?", const="http://127.0.0.1:8000")
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 300, 500])
    parser.add_argument("--trees", type=int, nargs="+", default=[500, 250, 100, 50, 25])
    args = parser.parse_args()

    from src.config import VAL_START
    from src.data.db import connect
    from src.eval.split import ground_truth
    from src.serve import app as srv

    srv._load()

    con = connect()
    gt = ground_truth(con, VAL_START)
    truth = {int(r.customer_key): list(r.articles) for r in gt.itertuples()}
    con.close()

    random.seed(0)
    keys = random.sample(sorted(truth), min(args.n, len(truth)))

    print("latency by candidate budget:")
    lat = pd.DataFrame([bench_inprocess(keys, b) for b in args.budgets])
    print(lat.to_string(index=False))

    print("\noffline/online parity:")
    par = parity(keys, truth)
    print(json.dumps(par, indent=2))

    print("\ncompression trade-off:")
    comp = compression_curve(keys, truth, args.trees)

    http_rows = None
    if args.http:
        id_map = {v: k for k, v in list(srv.state.customer_ids.items())}
        cids = [id_map[k] for k in keys if k in id_map][:200]
        http_rows = pd.DataFrame([bench_http(cids, args.http, b) for b in args.budgets])
        print("\nover HTTP:")
        print(http_rows.to_string(index=False))

    REPORTS.mkdir(exist_ok=True)
    parts = [
        "# Serving\n",
        (
            f"Sample: {len(keys)} customers from the validation week. "
            f"Model: {srv.state.booster.num_trees()} trees, "
            f"{len(srv.state.features)} features.\n"
        ),
        "## Latency by candidate budget (in-process)\n",
        lat.to_markdown(index=False) + "\n",
        "## Offline/online parity\n",
        (
            "The offline pipeline and the request path scored on the same customers. They read "
            "different storage by design, so an identical top 12 is the assertion worth making.\n"
        ),
        pd.DataFrame([par]).T.rename(columns={0: "value"}).to_markdown() + "\n",
        "## Compression trade-off\n",
        (
            "Tree count truncated at predict time via `num_iteration`; MAP@12 recomputed on "
            "the same customers. Candidate budget fixed at the default.\n"
        ),
        comp.to_markdown(index=False) + "\n",
    ]
    if http_rows is not None:
        parts += ["## Over HTTP (uvicorn, single worker)\n", http_rows.to_markdown(index=False) + "\n"]

    write_report(REPORTS / "serving.md", "\n".join(parts))
    print(f"\nwrote {REPORTS / 'serving.md'}")
    print(json.dumps({"latency": lat.to_dict("records")}, indent=2)[:400])


if __name__ == "__main__":
    main()
