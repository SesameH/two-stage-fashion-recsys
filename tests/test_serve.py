"""Serving-layer logic that is not covered by the offline metric tests.

Two things are checked here, and both existed as bugs before the test did:

1. The console's per-customer AP@12 must use the customer's full ground truth as the
   denominator. Passing only the predicted-and-correct items makes `min(len(actual), 12)`
   collapse to the hit count, and every customer with one hit at rank 1 scores a perfect 1.0.
2. Every feature the model expects must be produced by the request path. A feature the snapshot
   cannot supply gets filled with an absence sentinel, which is silent skew — the model was
   trained on a column that is constant in production.
"""

from __future__ import annotations

import pytest

from src.serve.precompute import SERVING


def _rows(ids, truth):
    return [{"article_id": a, "hit": a in truth} for a in ids]


class TestConsoleAveragePrecision:
    def test_denominator_is_the_full_truth_set_not_the_hits(self):
        from src.serve.app import _ap

        truth = {10, 20, 30}
        rows = _rows([10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12], truth)
        # One hit at rank 1 out of three purchases: 1.0 * (1/3).
        assert _ap(rows, truth) == pytest.approx(1 / 3)

    def test_perfect_prediction_scores_one(self):
        from src.serve.app import _ap

        truth = {10, 20, 30}
        assert _ap(_rows([10, 20, 30, 1, 2], truth), truth) == pytest.approx(1.0)

    def test_no_hits_scores_zero(self):
        from src.serve.app import _ap

        truth = {10, 20}
        assert _ap(_rows([1, 2, 3], truth), truth) == 0.0

    def test_rank_matters(self):
        from src.serve.app import _ap

        truth = {10, 20, 30}
        early = _ap(_rows([10, 1, 2], truth), truth)
        late = _ap(_rows([1, 2, 10], truth), truth)
        assert early > late


needs_snapshot = pytest.mark.skipif(
    not (SERVING / "meta.json").exists(),
    reason="serving snapshot not built; run `make serve-data`",
)


@needs_snapshot
def test_retrieval_features_match_the_offline_pipeline():
    """`rrf_score` and `n_sources` must mean the same thing on both paths.

    test_parity.py compares the four *feature blocks* on a fixed candidate set; it deliberately
    supplies its own retrieval columns, so it cannot see a disagreement in how candidates are
    fused or counted. This closes that gap: it is how a stale annotator list was caught changing
    `n_sources` for 6.3% of candidate rows while every feature-block test stayed green.
    """
    import json
    from datetime import date

    import pandas as pd

    from src.data.db import connect, register_customers
    from src.recall.pipeline import build_candidates
    from src.serve import app

    app._load()
    as_of = date.fromisoformat(json.loads((SERVING / "meta.json").read_text())["as_of"])

    con = connect()
    customers = con.execute(
        "SELECT DISTINCT customer_key FROM transactions "
        "WHERE t_dat >= ? AND t_dat < ? + INTERVAL 7 DAY ORDER BY customer_key LIMIT 10",
        [as_of, as_of],
    ).df()["customer_key"].tolist()
    register_customers(con, customers)
    offline = build_candidates(con, as_of, customers)
    con.close()

    online = pd.concat(
        [app.retrieve(int(ck)).assign(customer_key=int(ck)) for ck in customers],
        ignore_index=True,
    )

    both = offline.merge(
        online[["customer_key", "article_id", "rrf_score", "n_sources"]],
        on=["customer_key", "article_id"],
        suffixes=("_off", "_on"),
    )
    assert len(both) > 0.99 * len(offline), "candidate sets diverged between the two paths"
    assert (both["n_sources_off"] == both["n_sources_on"]).all(), "n_sources counts differ"
    assert (both["rrf_score_off"] - both["rrf_score_on"]).abs().max() < 1e-6, "RRF fusion differs"


@needs_snapshot
def test_every_model_feature_is_servable():
    """The request path must produce all of models/features.json without sentinel patching."""
    from src.serve import app

    app._load()
    key = next(iter(app.state.customer_ids.values()))
    feats = app.build_request_features(key)
    assert not feats.empty, "no candidates for the sampled customer"
    assert not app.UNSERVABLE, (
        f"model features the serving snapshot cannot supply: {sorted(app.UNSERVABLE)}. "
        "Either add them to precompute.py or remove them from the model."
    )
