import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classification.ClassifierBaseClass import ClassifierBaseClass
from config import *

class RandomForest (ClassifierBaseClass):
    def __init__(self, **kwargs):
        ClassifierBaseClass.__init__(self, **kwargs)

        self.n_estimators = kwargs.get("n_estimators", 10)

        self.criterion = kwargs.get("criterion", "gini")

        self.max_features = kwargs.get("max_features", "auto")

        self.max_depth = kwargs.get("max_depth", None)

        self.min_samples_split = kwargs.get("min_samples_split", 2)

        self.min_samples_leaf = kwargs.get("min_samples_leaf", 1)

        self.min_weight_fraction_leaf = kwargs.get("min_weight_fraction_leaf", 0.)

        self.max_leaf_nodes = kwargs.get("max_leaf_nodes", None)

        self.bootstrap = kwargs.get("bootstrap", True)

        self.oob_score = kwargs.get("oob_score", False)

        self.n_jobs = kwargs.get("n_jobs", 1)

        self.random_state = kwargs.get("random_state", None)

        self.verbose = kwargs.get("verbose", 0)

        self.warm_start = kwargs.get("warm_start", False)

        self.class_weight = kwargs.get("class_weight", None)


    def setClassifer(self):
        self.classifier =  RandomForestClassifier(n_estimators=self.n_estimators,
                                                  criterion=self.criterion,
                                                  max_features=self.max_features,
                                                  max_depth=self.max_depth,
                                                  min_samples_split=self.min_samples_split,
                                                  min_samples_leaf=self.min_samples_leaf,
                                                  min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  bootstrap=self.bootstrap,
                                                  oob_score=self.oob_score,
                                                  n_jobs=self.n_jobs,
                                                  random_state=self.random_state,
                                                  warm_start=self.warm_start,
                                                  class_weight=self.class_weight)

def main():
    clf = RandomForest(n_estimators=50)

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