import numpy as np
import pytest

from motion_retrieval.metrics import ranks_from_similarity, recall_at_k, weighted_score


def test_perfect_ranking_scores_one():
    assert weighted_score([1, 1, 1]) == pytest.approx(1.0)


def test_never_in_top_10_scores_zero():
    assert weighted_score([11, 20]) == 0.0


def test_recall_at_k():
    ranks = [1, 2, 5, 11]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.75
    assert recall_at_k(ranks, 10) == 0.75


def test_rank_two_is_worth_less_than_rank_one():
    assert weighted_score([2]) < weighted_score([1])
    # Rank 2 is missed only by Recall@1, which has the biggest weight (1 / 1).
    weights = np.array([1 / k for k in range(1, 11)])
    assert weighted_score([2]) == pytest.approx(1 - 1 / weights.sum())


def test_ranks_from_similarity():
    sim = np.array([
        [0.9, 0.1, 0.0],  # right text first
        [0.8, 0.5, 0.1],  # right text second
        [0.7, 0.6, 0.2],  # right text third
    ])
    assert list(ranks_from_similarity(sim)) == [1, 2, 3]
