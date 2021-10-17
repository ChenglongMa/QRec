# coding:utf8
from base.recommender import Recommender
import numpy as np


class MostPopular(Recommender):
    # Recommend the most popular items for every user
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(MostPopular, self).__init__(conf, trainingSet, testSet, fold)

    def initModel(self):
        self.popularItemList = np.random.random(self.data.trainingSize()[1])
        for itemName in self.data.trainSet_i:
            ind = self.data.item[itemName]
            self.popularItemList[ind] = len(self.data.trainSet_i[itemName])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            return self.popularItemList.copy()
        else:
            return [self.data.globalMean] * self.num_items
