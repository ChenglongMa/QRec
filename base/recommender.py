# QRec is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
import os

import pandas as pd

from data.rating import Rating
from util.io import FileIO
from util.config import LineConfig
from util.log import Log
from os.path import abspath
from time import strftime, localtime, time
from util.measure import Measure
from util.metric import precision_recall_at_k, gini_index
from util.qmath import find_k_largest


class Recommender(object):
    def __init__(self, conf, trainingSet, testSet, fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.iteration = -1
        self.test_on = ''
        self.suffix = ''
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        if self.evalSettings.contains('-cold'):
            # evaluation on cold-start users
            threshold = int(self.evalSettings['-cold'])
            removedUser = {}
            for user in self.data.testSet_u:
                if user in self.data.trainSet_u and len(self.data.trainSet_u[user]) > threshold:
                    removedUser[user] = 1
            for user in removedUser:
                del self.data.testSet_u[user]
            testData = []
            for item in self.data.testData:
                if item[0] not in removedUser:
                    testData.append(item)
            self.data.testData = testData

        self.num_users, self.num_items, self.train_size = self.data.trainingSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.algorName, self.algorName + self.foldInfo + ' ' + currentTime)
        # save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(f'{k} = {self.config[k]}')

    def readConfiguration(self):
        self.algorName = self.config['model.name']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])
        self.iteration = self.config['iteration']
        self.test_on = self.config['testOn']
        self.suffix = f'{self.test_on}_iter_{self.iteration}'

    def printAlgorConfig(self):
        "show model's configuration"
        print('Algorithm:', self.config['model.name'])
        print('Ratings dataset:', abspath(self.config['ratings']))
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(LineConfig(self.config['evaluation.setup'])['-testSet']))
        # print dataset statistics
        print('Training set size: (user count: %d, item count %d, record count: %d)' % (self.data.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' % (self.data.testSize()))
        print('=' * 80)
        # print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = LineConfig(self.config[self.config['model.name']])
            for key in args.keys():
                parStr += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def buildModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    # for rating prediction
    def predictForRating(self, u, i):
        pass

    # for item prediction
    def predictForRanking(self, u):
        pass

    def checkRatingBoundary(self, prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction, 3)

    def evalRatings(self):
        algo_name = self.algorName
        res = list()  # used to contain the text of the result
        res.append('uid,iid,r_ui,est\n')
        # predict
        for ind, entry in enumerate(self.data.testData):
            user, item, rating = entry
            # predict
            prediction = self.predictForRating(user, item)
            # denormalize
            # prediction = denormalize(prediction,self.data.rScale[-1],self.data.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.data.testData[ind].append(pred)
            res.append(user + ',' + item + ',' + str(rating) + ',' + str(pred) + '\n')

        # currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir'] + algo_name
            os.makedirs(outDir, exist_ok=True)
            fileName = self.suffix + '-rating-predictions' + self.foldInfo + '.csv'
            FileIO.writeFile(outDir, fileName, res)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        outDir = self.output['-dir'] + algo_name
        os.makedirs(outDir, exist_ok=True)
        fileName = self.suffix + '-measure-rating' + self.foldInfo + '.csv'
        self.measure, mae, rmse = Measure.ratingMeasure(self.data.testData)
        evaDF = pd.DataFrame([
            [algo_name, 'MAE', mae, -1, self.iteration, self.test_on],
            [algo_name, 'RMSE', rmse, -1, self.iteration, self.test_on]
        ], columns='algo metric value k iteration stage'.split())
        evaDF.to_csv(f'{outDir}/{fileName}', index=False)

        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

    def evalRanking(self):
        algo_name = self.algorName
        if self.ranking.contains('-topN'):
            Ns = self.ranking['-topN'].split(',')
            Ns = [int(num) for num in Ns]
            k = int(Ns[-1])
            if k > 100 or k < 0:
                print('N can not be larger than 100! It has been reassigned with 10')
                k = 10
            if k > len(self.data.item):
                k = len(self.data.item)
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet_u)
        # rawRes = {}
        predictions = []
        for user, ir_dict in self.data.testSet_u.items():
            candidates = self.predictForRanking(user)
            for item_name, true_r in ir_dict.items():
                predictions.append([user, item_name, true_r, candidates[self.data.item[item_name]]])
        pred_df = pd.DataFrame(predictions, columns='uid iid r_ui est'.split())

        # currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        suffix = f'{self.suffix}_top_{k}'
        if self.isOutput:
            outDir = self.output['-dir'] + algo_name
            os.makedirs(outDir, exist_ok=True)
            fileName = suffix + '-items' + self.foldInfo + '.csv'
            pred_df.to_csv(f'{outDir}/{fileName}', index=False)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        if self.evalSettings.contains('-predict'):
            # no evalutation
            exit(0)
        # Rank evaluation
        eva_res = []
        outDir = self.output['-dir'] + algo_name
        os.makedirs(outDir, exist_ok=True)
        fileName = suffix + '-measure-rank' + self.foldInfo + '.csv'
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=2)  # TODO: update threshold
        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        eva_res.append([algo_name, 'Precision', precision, k, self.iteration, self.test_on])
        eva_res.append([algo_name, 'Recall', recall, k, self.iteration, self.test_on])

        gini = gini_index(pred_df, k=k)
        eva_res.append([algo_name, 'Gini', gini, k, self.iteration, self.test_on])
        self.log.add('###Evaluation Results###')
        resDF = pd.DataFrame(eva_res, columns='algo metric value k iteration stage'.split())
        resDF.to_csv(f'{outDir}/{fileName}', index=False)
        print('The result of %s %s:\n' % (self.algorName, self.foldInfo))
        print(resDF)

    def evalRankingOld(self):
        """
        @Deprecated by mcl
        """
        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = int(top[-1])
            if N > 100 or N < 0:
                print('N can not be larger than 100! It has been reassigned with 10')
                N = 10
            if N > len(self.data.item):
                N = len(self.data.item)
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet_u)
        # rawRes = {}
        for i, user in enumerate(self.data.testSet_u):
            line = user + ':'
            candidates = self.predictForRanking(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                candidates[self.data.item[item]] = 0
            ids, scores = find_k_largest(N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names, scores))
            if i % 100 == 0:
                print(self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount))
            for item in recList[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.testSet_u[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name'] + '@' + currentTime + '-top-' + str(
                N) + 'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, self.recOutput)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        if self.evalSettings.contains('-predict'):
            # no evalutation
            exit(0)
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        # load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' % self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' % self.foldInfo)
            self.initModel()
            print('Building Model %s...' % self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.buildModel_tf()
                else:
                    self.buildModel()
            except ImportError:
                self.buildModel()
        # rating prediction or item ranking
        print('Predicting %s...' % self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()
        # save model
        if self.isSaveModel:
            print('Saving model %s...' % self.foldInfo)
            self.saveModel()
        return self.measure
