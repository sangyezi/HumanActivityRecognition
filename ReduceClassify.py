from Classify import Classifier
from ReduceDimension.PCAReduceDimension import PCAReduceDimension
from ReduceDimension.LDAReduceDimension import LDAReduceDimension
import numpy as np
from config import *
import copy

class ReduceClassifier (Classifier):
    def __init__(self, outputFile):
        Classifier.__init__(self, outputFile)
        self._createdLearner = False
        self._dimReduced = False


    def createLearner(self, learner, **kwargs):
        if learner == "PCA":
            self.learner = PCAReduceDimension(**kwargs)
        elif learner == "LDA":
            self.learner = LDAReduceDimension(**kwargs)
        self._createdLearner = True

    def loadData(self, xdataDir, ydataDir):
        Classifier.loadData(self, xdataDir, ydataDir)
        self.xTrainDataNoTransform = copy.deepcopy(self.xTrainData)
        self.xDevDataNoTransform = copy.deepcopy(self.xDevData)

    def reduceDimension(self):

        if self._dataLoaded and self._createdLearner:
            for i in range(crossValidationFold):
                trainingTimes = []
                transformTrainTimes = []
                transformTestTimes  = []
                trainingTimes.append(self.learner.train(self.xTrainDataNoTransform[i], self.yTrainData[i]))

                [transformTime, self.xTrainData[i]] = self.learner.reduceDimension(self.xTrainDataNoTransform[i])
                transformTrainTimes.append(transformTime)

                [transformTime, self.xDevData[i]] = self.learner.reduceDimension(self.xDevDataNoTransform[i])
                transformTestTimes.append(transformTime)

            self.outputFile.write("%16.2f %16.2f %16.2f %16.2f %16.2f %16.2f\n" %
                                  (np.mean(trainingTimes)*1000,         np.std(trainingTimes)*1000,
                                   np.mean(transformTrainTimes)*1000,   np.std(transformTrainTimes)*1000,
                                   np.mean(transformTestTimes)*1000,    np.std(transformTestTimes)*1000))
            self.outputFile.flush()
            self._dimReduced = True
        else:
            raise RuntimeError("Error: need to create Learner and load data first")

    def metAllRequirement(self):
        return self._dimReduced