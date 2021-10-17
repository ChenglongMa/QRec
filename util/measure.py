import math
from collections import defaultdict


class Measure(object):
    def __init__(self):
        pass

    @staticmethod
    def ratingMeasure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:' + str(mae) + '\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse) + '\n')
        return measure

    @staticmethod
    def hits(origin, res):
        hitCount = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hitCount[user] = len(set(items).intersection(set(predicted)))
        return hitCount

    @staticmethod
    def rankingMeasure(origin, res, N):
        measure = []
        for n in N:
            top_n = {}  # len(top_n) == n_users, format: <uid, list(<iid, est>)>
            for user in res:
                top_n[user] = res[user][:n]
            indicators = []
            if len(origin) != len(top_n):
                print('The Lengths of test set and top_n set are not match!')
                exit(-1)
            # hits = Measure.hits(origin, top_n)
            # prec = Measure.precision(hits, n)
            # indicators.append('Precision:' + str(prec) + '\n')
            # recall = Measure.recall(hits, origin)
            # indicators.append('Recall:' + str(recall) + '\n')

            # TODO: updated by mcl
            precisions, recalls = Measure.precision_recall_at_k(origin, res, k=n, threshold=2)  # TODO: update threshold
            # Precision and recall can then be averaged over all users
            precision = sum(prec for prec in precisions.values()) / len(precisions)
            indicators.append('Precision:' + str(precision) + '\n')
            recall = sum(rec for rec in recalls.values()) / len(recalls)
            indicators.append('Recall:' + str(recall) + '\n')
            F1 = Measure.F1(precision, recall)
            indicators.append('F1:' + str(F1) + '\n')
            # MAP = Measure.MAP(origin, top_n, n)
            # indicators.append('MAP:' + str(MAP) + '\n')
            NDCG = Measure.NDCG(origin, top_n, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            # AUC = Measure.AUC(origin,res,rawRes)
            # measure.append('AUC:' + str(AUC) + '\n')
            measure.append('Top ' + str(n) + '\n')
            measure += indicators
        return measure

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    # @staticmethod
    # def precision(origin, top_n, threshold):
    #     precs = [len([i for i, _ in irs if origin[u][i] >= threshold]) / len(irs) for u, irs in top_n.items()]
    #     return np.mean(precs)

    @staticmethod
    def precision_recall_at_k(true_rs: dict, pred_rs: dict, k, threshold):
        # pred_dict = defaultdict(dict)
        # for uid, irs in pred_rs.items():
        #     for iid, est in irs:
        #         pred_dict[uid][iid] = est

        user_est_true = defaultdict(list)
        for uid, irs in pred_rs.items():
            for iid, est in irs:
                true_r = true_rs[uid][iid]
                user_est_true[uid].append((est, true_r))
        # for uid, irdict in true_rs.items():
        #     for iid, true_r in irdict.items():
        #         est = pred_dict[uid][iid]
        #         user_est_true[uid].append((est, true_r))

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

    # @staticmethod
    # def MAP(origin, res, N):
    #     sum_prec = 0
    #     for user in res:
    #         hits = 0
    #         precision = 0
    #         for n, item in enumerate(res[user]):
    #             if item[0] in origin[user]:
    #                 hits += 1
    #                 precision += hits / (n + 1.0)
    #         sum_prec += precision / min(len(origin[user]), N)
    #     return sum_prec / len(res)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)

    @staticmethod
    def recall(hits, origin):
        recallList = [hits[user] / len(origin[user]) for user in hits]
        recall = sum(recallList) / len(recallList)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return error / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(error / count)
