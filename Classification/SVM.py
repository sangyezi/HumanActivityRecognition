import numpy as np
from sklearn import svm

from Classification.ClassifierBaseClass import ClassifierBaseClass
from config import *


class SVM (ClassifierBaseClass):
    def __init__(self, **kwargs):
        ClassifierBaseClass.__init__(self, **kwargs)

        self.C = kwargs.get("C", 1.0)

        self.kernel = kwargs.get("kernel", "rbf")

        self.degree = kwargs.get("degree",3)

        self.gamma  = kwargs.get("gamma", "auto")

        self.coef0 = kwargs.get("coef0", 0.0)

        self.probability = kwargs.get("probability", False)

        self.shrinking = kwargs.get("shrinking", True)

        self.tol = kwargs.get("tol", 1e-3)

        self.class_weight = kwargs.get("class_weight", None)

        self.verbose = kwargs.get("verbose", False)

        self.max_iter = kwargs.get("max_iter", -1)

        self.decision_function_shape = kwargs.get("decision_function_shape", None)

        self.random_state = kwargs.get("random_state", None)


    def setClassifer(self):
        self.classifier = svm.SVC(kernel=self.kernel,
                                  C=self.C,
                                  degree=self.degree,
                                  gamma=self.gamma,
                                  coef0=self.coef0,
                                  probability=self.probability,
                                  shrinking=self.shrinking,
                                  tol=self.tol,
                                  class_weight=self.class_weight,
                                  verbose=self.verbose,
                                  max_iter=self.max_iter,
                                  decision_function_shape=self.decision_function_shape,
                                  random_state=self.random_state)



def main():
    clf = SVM(kernel="linear")

    xdataDir = ldaReducedDataDir
    xdataDir = pcaRedcuedDataDir
    xdataDir = dataDir

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