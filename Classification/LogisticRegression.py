import numpy as np
from sklearn import linear_model

from Classification.ClassifierBaseClass import ClassifierBaseClass
from config import  *


class LogisticRegression (ClassifierBaseClass):
    def __init__(self, **kwargs):
        ClassifierBaseClass.__init__(self, **kwargs)

        self.penalty = kwargs.get("penalty", "l2")

        self.dual = kwargs.get("dual", False)

        self.C = kwargs.get("C", 1.0)

        self.fit_intercept = kwargs.get("fit_intercept", True)

        self.intercept_scaling = kwargs.get("intercept_scaling", 1)

        self.class_weight = kwargs.get("class_weight", None)

        self.max_iter = kwargs.get("max_iter", 100)

        self.random_state = kwargs.get("random_state", None)

        self.solver = kwargs.get("solver", "liblinear")

        self.tol = kwargs.get("tol", 1e-3)

        self.multi_class = kwargs.get("multi_class", "ovr")

        self.verbose = kwargs.get("verbose", 0)

        self.warm_start = kwargs.get("warm_start", False)

        self.n_jobs = kwargs.get("n_jobs", 1)


    def setClassifer(self):
        self.classifier = linear_model.LogisticRegression(penalty=self.penalty,
                                                          dual=self.dual,
                                                          C=self.C,
                                                          fit_intercept=self.fit_intercept,
                                                          intercept_scaling=self.intercept_scaling,
                                                          class_weight=self.class_weight,
                                                          max_iter=self.max_iter,
                                                          random_state=self.random_state,
                                                          solver=self.solver,
                                                          tol=self.tol,
                                                          multi_class=self.multi_class,
                                                          verbose=self.verbose,
                                                          warm_start=self.warm_start,
                                                          n_jobs=self.n_jobs)


def main():
    clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')

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