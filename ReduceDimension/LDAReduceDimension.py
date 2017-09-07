from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from config import *
import timeit
from ReduceDimensionBaseClass import  ReduceDimensionBaseClass

class LDAReduceDimension (ReduceDimensionBaseClass):
    def __init__(self, **kwargs):
        ReduceDimensionBaseClass.__init__(self, **kwargs)

        self.n_components = kwargs.get("n_components", 2)

        self.learner = LinearDiscriminantAnalysis(n_components=self.n_components)

    def train(self, xdata, ydata):
        start_train = timeit.default_timer()
        self.learner.fit(xdata, ydata)
        stop_train = timeit.default_timer()
        self.trained = True
        return (stop_train - start_train)



def main():
    n_components= 5
    learner = LDAReduceDimension(n_components=n_components)

    if not os.path.exists(ldaReducedDataDir):
        os.makedirs(ldaReducedDataDir)

    trainingTimes = []
    transformTimes =[]
    for i in range(crossValidationFold):
        xTrain = np.load(dataDir + "xTrain" + str(i) + ".npy")
        yTrain = np.load(dataDir + "yTrain" + str(i) + ".npy") # diff here
        xDev = np.load(dataDir + "xDev" + str(i) + ".npy")
        trainingTimes.append(learner.train(xTrain, yTrain)) # diff here

        [transformTime, xTrainReduced] = learner.reduceDimension(xTrain)
        transformTimes.append(transformTime)
        [transformTime, xDevReduced] = learner.reduceDimension(xDev)
        transformTimes.append(transformTime)

        print "i = %d, training time: %.2f ms, transform xTran time: %.2f ms, transform xDev: %.2f ms" \
              % (i, trainingTimes[i] * 1000, transformTimes[i*2] * 1000, transformTimes[i*2+1] * 1000)

        # np.save(ldaReducedDataDir + "xTrain" + str(i) + "dim" + str(n_components) + ".npy", xTrainReduced)
        np.save(ldaReducedDataDir + "xTrain" + str(i)  + ".npy", xTrainReduced)
        # np.save(ldaReducedDataDir + "xDev" + str(i) +  "dim" + str(n_components) + ".npy", xDevReduced)
        np.save(ldaReducedDataDir + "xDev" + str(i) + ".npy", xDevReduced)


    xTrain = np.load(dataDir + "xTrain.npy")
    yTrain = np.load(dataDir + "yTrain.npy") # diff here
    xTest = np.load(dataDir + "xTest.npy")

    trainingTime = learner.train(xTrain, yTrain) # diff here

    [transformTrainTime, xTrainReduced] = learner.reduceDimension(xTrain)
    [transformTestTime, xTestReduced] = learner.reduceDimension(xTest)

    print "training time: %.2f ms, transform xTran time: %.2f ms, transform xTest: %.2f ms"  \
              % (trainingTime * 1000, transformTrainTime * 1000, transformTestTime * 1000)

    # np.save(ldaReducedDataDir + "xTrain" + "dim" + str(n_components) + ".npy", xTrainReduced)
    np.save(ldaReducedDataDir + "xTrain"  + ".npy", xTrainReduced)
    # np.save(ldaReducedDataDir + "xTest" + "dim" + str(n_components) + ".npy", xTestReduced)
    np.save(ldaReducedDataDir + "xTest"  + ".npy", xTestReduced)

if __name__ == "__main__":
    main()