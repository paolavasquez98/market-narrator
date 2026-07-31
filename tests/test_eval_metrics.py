import pytest

from finrag.eval.metrics import hit_rate, mean_reciprocal_rank, rank_of


def test_rank_of_finds_position_1_based():
    assert rank_of("B", ["A", "B", "C"]) == 2


def test_rank_of_returns_none_when_absent():
    assert rank_of("Z", ["A", "B", "C"]) is None


def test_hit_rate_hand_computed():
    # 3 of 4 queries found their relevant doc somewhere in the results.
    ranks = [1, None, 3, 2]
    assert hit_rate(ranks) == pytest.approx(0.75)


def test_hit_rate_of_empty_list_is_zero():
    assert hit_rate([]) == 0.0


def test_mrr_hand_computed():
    # 1/1 + 0 + 1/3 + 1/2, divided by 4 queries.
    ranks = [1, None, 3, 2]
    expected = (1 / 1 + 0 + 1 / 3 + 1 / 2) / 4
    assert mean_reciprocal_rank(ranks) == pytest.approx(expected)


def test_mrr_rewards_higher_ranks_more_than_hit_rate_does():
    # Same hit rate (both found), but MRR should differ: rank 1 beats rank 5.
    best_case = [1, 1]
    worst_case = [5, 5]

    assert hit_rate(best_case) == hit_rate(worst_case) == 1.0
    assert mean_reciprocal_rank(best_case) > mean_reciprocal_rank(worst_case)


def test_mrr_of_empty_list_is_zero():
    assert mean_reciprocal_rank([]) == 0.0
