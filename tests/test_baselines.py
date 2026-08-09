"""Unit tests for the baseline padding logic — the part of B2 that is easy to get subtly wrong."""

from src.baselines.heuristics import pad


def test_pad_fills_to_k_without_duplicates():
    out = pad({1: [10, 11]}, customers=[1], filler=[11, 20, 21], k=4)
    assert out[1] == [10, 11, 20, 21]


def test_pad_serves_customers_with_no_history():
    out = pad({}, customers=[1], filler=[20, 21, 22], k=2)
    assert out[1] == [20, 21]


def test_pad_truncates_overlong_history():
    out = pad({1: [10, 11, 12]}, customers=[1], filler=[20], k=2)
    assert out[1] == [10, 11]


def test_pad_runs_short_when_filler_is_exhausted():
    # Better to return fewer than k than to invent items; MAP@12 does not punish short lists.
    out = pad({1: [10]}, customers=[1], filler=[10], k=5)
    assert out[1] == [10]


def test_pad_covers_every_requested_customer():
    out = pad({1: [10]}, customers=[1, 2, 3], filler=[20], k=1)
    assert set(out) == {1, 2, 3}
