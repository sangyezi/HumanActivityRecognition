from Classification.SVM import SVM
from Classification.NeuralNetwork import NeuralNetwork
from Classification.LogisticRegression import LogisticRegression
from Classification.GradientBoostingTree import GradientBoostingTree
from Classification.RandomForest import RandomForest
from config import *
import numpy as np

class Classifier:
    def __init__(self, outputFile):
        self.outputFile = outputFile
        self._dataLoaded = False
        self._createdClassifier = False
        self.xTrainData = []
        self.yTrainData = []
        self.xDevData   = []
        self.yDevData   = []


    def loadData(self, xdataDir, ydataDir):
        for i in range(crossValidationFold):
            self.xTrainData.append(np.load(xdataDir + "xTrain" + str(i) + ".npy"))
            self.yTrainData.append(np.load(ydataDir + "yTrain" + str(i) + ".npy"))
            self.xDevData.append(np.load(xdataDir + "xDev" + str(i) + ".npy"))
            self.yDevData.append(np.load(ydataDir + "yDev" + str(i) + ".npy"))
        self._dataLoaded = True

    def createClassifier(self, classifier, **kwargs):
        if classifier == "SVM":
            self.clf = SVM(**kwargs)
        elif classifier == "NeuralNetwork":
            self.clf = NeuralNetwork(**kwargs)
        elif classifier == "LogisticRegression":
            self.clf = LogisticRegression(**kwargs)
        elif classifier == "GradientBoostingTree":
            self.clf = GradientBoostingTree(**kwargs)
        elif classifier == "RandomForest":
            self.clf = RandomForest(**kwargs)
        else:
            raise RuntimeError("Error: specified wrong classifier")
        self._createdClassifier = True

    def classify(self):
        trainingTimes = []
        testingTimes = []
        accuracies = []
        if self.metAllRequirement():
            for i in range(crossValidationFold):
                trainingTimes.append(self.clf.train(self.xTrainData[i], self.yTrainData[i]))
                [testingTime, accuracy] = self.clf.test(self.xDevData[i], self.yDevData[i])
                testingTimes.append(testingTime)
                accuracies.append(accuracy)
            self.outputFile.write("%16.2f %16.2f %16.2f %16.2f %15.2f%% %15.2f%%\n" %
                                  (np.mean(trainingTimes)*1000, np.std(trainingTimes)*1000,
                                   np.mean(testingTimes)*1000,   np.std(testingTimes)*1000,
                                   np.mean(accuracies)*100,     np.std(accuracies)*100))
            #self.outputFile.flush()
        else:
            raise RuntimeError("Error: need to createClassifier and loadData first")


    def metAllRequirement(self):
        return self._createdClassifier and self._dataLoaded
