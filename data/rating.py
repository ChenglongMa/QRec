import numpy as np
from util.config import Config, LineConfig
import random
from collections import defaultdict


class Rating(object):
    'data access control'

    def __init__(self, config, trainingSet, testSet):
        self.config = config
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.user = {}  # map user names to id
        self.item = {}  # map item names to id, format: itemName: itemInnerID
        self.id2user = {}
        self.id2item = {}  # format: itemInnerID: itemName
        self.userMeans = {}  # mean values of users's ratings
        self.itemMeans = {}  # mean values of items's ratings, format: itemName: mean rating
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)  # test set in the form of [user][item]=rating
        self.testSet_i = defaultdict(dict)  # test set in the form of [item][user]=rating
        self.rScale = []  # rating scale
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()

    def __generateSet(self):
        scale = set()
        # if validation is conducted, we sample the training data at a given probability to form the validation set,
        # and then replacing the test data with the validation data to tune parameters.
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount() * float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]
        for i, entry in enumerate(self.trainingData):
            userName, itemName, rating = entry
            # makes the rating within the range [0, 1].
            # rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            # self.trainingData[i][2] = rating
            # order the user
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            if self.evalSettings.contains('-predict'):
                self.testSet_u[entry] = {}
            else:
                userName, itemName, rating = entry
                self.testSet_u[userName][itemName] = rating
                self.testSet_i[itemName][userName] = rating

    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total == 0:
            self.globalMean = 0
        else:
            self.globalMean = total / len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            self.userMeans[u] = sum(self.trainSet_u[u].values()) / len(self.trainSet_u[u])

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values()) / len(self.trainSet_i[c])

    def getUserId(self, u):
        if u in self.user:
            return self.user[u]

    def getItemId(self, i):
        if i in self.item:
            return self.item[i]

    def trainingSize(self):
        """
        Return tuple of <n_users, n_items, n_ratings>
        """
        return (len(self.user), len(self.item), len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u), len(self.testSet_i), len(self.testData))

    def contains(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsUser(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self, i):
        'whether item is in training set'
        if i in self.item:
            return True
        else:
            return False

    def userRated(self, u):
        return list(self.trainSet_u[u].keys()), list(self.trainSet_u[u].values())

    def itemRated(self, i):
        return list(self.trainSet_i[i].keys()), list(self.trainSet_i[i].values())

    def userInTest(self, u):
        return list(self.testSet_u[u].keys()), list(self.testSet_u[u].values()),

    def itemInTest(self, i):
        return list(self.testSet_i[i].keys()), list(self.testSet_i[i].values())

    def allUsers(self):
        return self.user.keys()

    def allItems(self):
        return self.item.keys()

    def row(self, u):
        k, v = self.userRated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        k, v = self.itemRated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.userRated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m

    # def row(self,u):
    #     return self.trainingMatrix.row(self.getUserId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getItemId(c))

    def sRow(self, u):
        return self.trainSet_u[u]

    def sCol(self, c):
        return self.trainSet_i[c]

    def rating(self, u, c):
        if self.contains(u, c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0], self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
