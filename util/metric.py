from collections import defaultdict

import pandas as pd

from data.similarity import Similarity


def precision_recall_at_k(predictions: list, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value (i.e., x[0])
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = min(len(user_ratings), k)
        # n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # n_rec_k = sum((true_r >= threshold) for (est, true_r) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        # n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
        n_rel_and_rec_k = sum((true_r >= threshold) for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def gini_index(predictions: pd.DataFrame, k=10):
    """
    Python implementation of
    [Gini index](https://github.com/RankSys/RankSys/blob/master/RankSys-diversity/src/main/java/es/uam/eps/ir/ranksys/diversity/sales/metrics/GiniIndex.java#L48)
    :param predictions: dataframe, columns are: ['uid iid r_ui est']
    :param k:
    :return:
    """
    top_k = predictions.sort_values(['uid', 'est'], ascending=False).groupby('uid').head(k)
    cs = top_k.iid.value_counts(ascending=True).reset_index(drop=True)
    n_items = predictions.iid.nunique()
    free_norm = predictions.uid.nunique() * k
    gi = ((2 * (cs.index + (n_items - cs.shape[0]) + 1) - n_items - 1) * (cs / free_norm)).sum()
    gi /= n_items - 1
    # gi = 1 - gi
    return gi


def mean_intra_user_diversity(predictions, n, similarity: Similarity):
    """

    :param predictions: list of Prediction objects
    :param n: The number of recommendation to output for each user.
    :param similarity: The similarity class
    :return:
    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, *_ in predictions:
        top_n[uid].append((iid, est))

    n_users = len(top_n)
    _sum = 0
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, recommendations in top_n.items():
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recs = recommendations[:n]
        _sum += intra_user_diversity(top_recs, similarity)

    return _sum / n_users


def intra_user_diversity(recommendations: list, similarity: Similarity):
    """
    see: 3.2 Average Intra-List Distance in
        Castells, P., Hurley, N. J., & Vargas, S. (2015).
        Novelty and Diversity in Recommender Systems. In F. Ricci, L. Rokach, & B. Shapira (Eds.),
        Recommender Systems Handbook (pp. 881–918). Springer US. https://doi.org/10.1007/978-1-4899-7637-6_26

    :param recommendations: list of tuples <item, rating>
    :param similarity:
    :return:
    """
    k = len(recommendations)
    _sum = 0
    for i, _ in recommendations:
        for j, _ in recommendations:
            _sum += 1 - similarity.get_similarity(i, j)
    return _sum / (k * (k + 1))
