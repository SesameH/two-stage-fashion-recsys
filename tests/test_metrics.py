"""Hand-computed cases for the metrics module. If these drift, every other number is meaningless."""

import math

import pytest

from src.eval.metrics import apk, mapk, ndcg_at_k, recall_at_k


class TestApk:
    def test_single_hit_at_rank_1(self):
        assert apk([1], [1, 2, 3]) == pytest.approx(1.0)

    def test_single_hit_at_rank_2(self):
        # precision at the hit = 1/2, divided by min(1, 12) = 1
        assert apk([1], [2, 1, 3]) == pytest.approx(0.5)

    def test_two_hits_ranks_1_and_3(self):
        # (1/1 + 2/3) / min(2, 12) = 1.6666.../2
        assert apk([1, 2], [1, 3, 2]) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_no_hits(self):
        assert apk([1, 2], [3, 4, 5]) == 0.0

    def test_empty_actual_is_zero(self):
        assert apk([], [1, 2, 3]) == 0.0

    def test_empty_prediction_is_zero(self):
        assert apk([1], []) == 0.0

    def test_denominator_capped_at_k(self):
        # 15 truths, 12 perfect predictions: sum(i/i for i in 1..12) / min(15, 12) = 1.0
        actual = list(range(15))
        assert apk(actual, list(range(12)), k=12) == pytest.approx(1.0)

    def test_duplicate_predictions_credited_once_but_still_consume_a_slot(self):
        # Matches the official Kaggle apk: the repeated "1" scores nothing, yet rank 2 is
        # spent on it, so the hit on "2" is credited at rank 3 (2/3), not rank 2.
        assert apk([1, 2], [1, 1, 2]) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_duplicate_truth_does_not_inflate_the_denominator(self):
        # Buying the same article twice in the target week is one thing to predict, not two.
        assert apk([1, 1], [1, 9]) == pytest.approx(1.0)

    def test_beyond_k_is_ignored(self):
        assert apk([1], [0] * 12 + [1], k=12) == 0.0

    def test_order_of_actual_is_irrelevant(self):
        assert apk([2, 1], [1, 2]) == apk([1, 2], [1, 2])


class TestMapk:
    def test_mean_over_customers(self):
        actual = [[1], [1], [1]]
        predicted = [[1, 2], [2, 1], [3, 4]]  # 1.0, 0.5, 0.0
        assert mapk(actual, predicted) == pytest.approx(0.5)

    def test_zero_truth_customer_dilutes_the_score(self):
        # This is exactly why we evaluate only on customers who bought in the val week.
        assert mapk([[1], []], [[1], [1]]) == pytest.approx(0.5)

    def test_empty_input(self):
        assert mapk([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mapk([[1]], [[1], [2]])


class TestRecallAtK:
    def test_partial_recall(self):
        assert recall_at_k([[1, 2, 3, 4]], [[1, 2, 9]], k=100) == pytest.approx(0.5)

    def test_cutoff_is_respected(self):
        assert recall_at_k([[1]], [[9, 1]], k=1) == 0.0

    def test_customers_without_truth_are_skipped_not_zeroed(self):
        assert recall_at_k([[1], []], [[1], [1]], k=10) == pytest.approx(1.0)


class TestNdcgAtK:
    def test_hit_at_rank_2(self):
        assert ndcg_at_k([[1]], [[2, 1]]) == pytest.approx(1 / math.log2(3))

    def test_perfect_ranking(self):
        assert ndcg_at_k([[1, 2]], [[1, 2, 3]]) == pytest.approx(1.0)

    def test_no_hits(self):
        assert ndcg_at_k([[1]], [[2, 3]]) == 0.0
