import numpy as np

from base.recommender import Recommender


class ItemMean(Recommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(ItemMean, self).__init__(conf, trainingSet, testSet, fold)

    def initModel(self):
        self.item_mean_list = np.full(self.num_items, fill_value=self.data.globalMean)
        for name, rating in self.data.itemMeans.items():
            ind = self.data.item[name]
            self.item_mean_list[ind] = rating

    def predictForRating(self, u, i):
        if self.data.containsItem(i):
            return self.data.itemMeans[i]
        else:
            return self.data.globalMean

    def predictForRanking(self, u):
        if self.data.containsUser(u):
            return self.item_mean_list.copy()
        else:
            return np.full(self.num_items, fill_value=self.data.globalMean)
