import numpy as np
from sknn.mlp import Classifier, Layer
from config import *

from Classification.ClassifierBaseClass import ClassifierBaseClass

class NeuralNetwork (ClassifierBaseClass):
    def __init__(self, **kwargs):
        ClassifierBaseClass.__init__(self, **kwargs)

        self.HiddenLayers = kwargs.get("HiddenLayers", Layer("Maxout", units=20, pieces=2))

        self.OutputLayer = kwargs.get("OutputLayer", Layer("Softmax"))

        self.learning_rate = kwargs.get("learning_rate",0.001)

        self.n_iter = kwargs.get("n_iter", 25)

    def setClassifer(self):
        self.classifier = Classifier(layers=[self.HiddenLayers, self.OutputLayer],
                                     learning_rate=self.learning_rate,
                                     n_iter=self.n_iter)

def main():
    clf = NeuralNetwork()

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