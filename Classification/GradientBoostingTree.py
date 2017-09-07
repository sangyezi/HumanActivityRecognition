import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from Classification.ClassifierBaseClass import ClassifierBaseClass
from config import *


class GradientBoostingTree (ClassifierBaseClass):
    def __init__(self, **kwargs):
        ClassifierBaseClass.__init__(self, **kwargs)

        self.loss = kwargs.get("loss", "deviance") #

        self.learning_rate = kwargs.get("learning_rate", 0.1) #

        self.n_estimators = kwargs.get("n_estimators", 100)

        self.max_depth = kwargs.get("max_depth", 3) #

        self.min_samples_split = kwargs.get("min_samples_split", 2)

        self.min_samples_leaf = kwargs.get("min_samples_leaf", 1)

        self.min_weight_fraction_leaf = kwargs.get("min_weight_fraction_leaf", 0.)

        self.subsample = kwargs.get("subsample",1.0)

        self.max_features = kwargs.get("max_features", None)

        self.max_leaf_nodes = kwargs.get("max_leaf_nodes", None)

        self.init = kwargs.get("init", None)

        self.verbose = kwargs.get("verbose", 0)

        self.random_state = kwargs.get("random_state", None)

        self.presort = kwargs.get("presort", "auto")

    def setClassifer(self):
        self.classifier =  GradientBoostingClassifier(loss=self.loss,
                                                      learning_rate=self.learning_rate,
                                                      n_estimators=self.n_estimators,
                                                      max_depth=self.max_depth,
                                                      min_samples_split=self.min_samples_split,
                                                      min_samples_leaf=self.min_samples_leaf,
                                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                      subsample=self.subsample,
                                                      max_features=self.max_features,
                                                      max_leaf_nodes=self.max_leaf_nodes,
                                                      init=self.init,
                                                      verbose=self.verbose,
                                                      random_state=self.random_state,
                                                      presort=self.presort)


def main():
    clf = GradientBoostingTree(max_depth=3, learning_rate = 0.1)

    xdataDir = ldaReducedDataDir
    xdataDir = pcaRedcuedDataDir
    xdata = dataDir

    ydataDir = dataDir
    trainingTimes = []
    testingTimes = []
    accuracies = []
    for i in range(crossValidationFold):
        xTrain = np.load(xdataDir + "xTrain" + str(i) + ".npy")
        yTrain = np.load(ydataDir + "yTrain" + str(i) + ".npy")
        trainingTimes.append(clf.train(xTrain, yTrain))

        xTest = np.load(xdataDir + "xDev" + str(i) + ".npy")
        yTest = np.load(ydataDir + "yDev" + str(i) + ".npy")
        [testingTime, accuracy] = clf.test(xTest, yTest)
        testingTimes.append(testingTime)
        accuracies.append(accuracy)

        print "i = %d, training time: %.2f ms, testing time: %.2f ms, accuracy: %.1f %%" \
              % (i, trainingTimes[i] * 1000, testingTime * 1000, accuracy * 100)

    xTrain = np.load(xdataDir + "xTrain.npy")
    yTrain = np.load(ydataDir + "yTrain.npy")
    trainingTime = clf.train(xTrain, yTrain)
    xTest = np.load(xdataDir + "xTest.npy")
    yTest = np.load(ydataDir + "yTest.npy")
    [testingTime, accuracy] = clf.test(xTest, yTest)
    print "training time: %.2f ms, testing time: %.2f ms, accuracy: %.1f %%" \
              % ( trainingTime * 1000, testingTime * 1000, accuracy * 100)

if __name__ == "__main__":
    main()