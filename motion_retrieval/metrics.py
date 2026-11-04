"""The competition metric.

For each query motion, the model ranks the candidate texts. Recall@k is the
share of queries where the right text is in the top k. The leaderboard score
is a weighted average of Recall@1 ... Recall@10 with weight 1/k, so getting
the right answer first counts much more than getting it tenth.
"""

import numpy as np

KS = range(1, 11)


def recall_at_k(ranks, k):
    """ranks[i] = position (1 = first) of the right text for query i."""
    return float(np.mean(np.asarray(ranks) <= k))


def weighted_score(ranks):
    weights = np.array([1 / k for k in KS])
    recalls = np.array([recall_at_k(ranks, k) for k in KS])
    return float((weights * recalls).sum() / weights.sum())


def ranks_from_similarity(sim):
    """sim is a square matrix: row i = query i, column i = its right text.

    Returns the rank of the right text for each row.
    """
    sim = np.asarray(sim)
    correct = np.diag(sim)[:, None]
    # Rank = 1 + number of texts with a strictly higher score than the right one.
    return 1 + (sim > correct).sum(axis=1)


def summary(ranks):
    return {
        "R@1": recall_at_k(ranks, 1),
        "R@3": recall_at_k(ranks, 3),
        "R@5": recall_at_k(ranks, 5),
        "R@10": recall_at_k(ranks, 10),
        "score": weighted_score(ranks),
    }
