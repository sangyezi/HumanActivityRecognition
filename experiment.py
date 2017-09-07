from Classify import Classifier
from ReduceClassify import ReduceClassifier
from config import *
from sknn.mlp import Layer

def writeParamDict(outputFile, params):
    for key in params.keys():
        outputFile.write("%10s : %10s, " % (key, params[key]))
    outputFile.write("\n")
    #outputFile.flush()



if not os.path.exists(expResultDir):
    os.makedirs(expResultDir)



def experiment(classifier, paramValues, paramDefaultValue):
    outputFile = open(expResultDir + classifier + ".txt", 'w')

    outputFile.write("trainingTimeMean " +
                     " trainingTimeSTD " +
                     " testingTimeMean " +
                     "  testingTimeSTD " +
                     "    AccuracyMean " +
                     "     AccuracySTD\n")

    clf = Classifier(outputFile)

    clf.loadData(dataDir, dataDir)


    for key in paramValues.keys():
        for value in paramValues[key]:
            params = paramDefaultValue
            params[key] = value
            writeParamDict(outputFile, params)
            clf.createClassifier(classifier,**params)
            clf.classify()

    outputFile.close()

def experiment_withReduceDimension(classifier, classifierParams,
                                   learner, learnerParamValues, learnerParamDefaultValue):
    outputFile = open(expResultDir + learner + "_" + classifier + ".txt", 'w')

    outputFile.write("trainingTimeMean " +
                     " trainingTimeSTD " +
                     "redTrainTimeMean " +
                     " redTrainTimeSTD " +
                     " redTestTimeMean " +
                     "  redTestTimeSTD\n")

    outputFile.write("trainingTimeMean " +
                     " trainingTimeSTD " +
                     " testingTimeMean " +
                     "  testingTimeSTD " +
                     "    AccuracyMean " +
                     "     AccuracySTD\n")
    clf = ReduceClassifier(outputFile)

    clf.loadData(dataDir, dataDir)

    writeParamDict(outputFile, classifierParams)

    for key in learnerParamValues.keys():
        for value in learnerParamValues[key]:
            params = learnerParamDefaultValue
            params[key] = value
            writeParamDict(outputFile, params)
            clf.createLearner(learner, **params)
            clf.reduceDimension()
            clf.createClassifier(classifier, **classifierParams)
            clf.classify()


def RunSVM():
    classifier = "SVM"
    paramValues = {"kernel" : ["poly", "rbf", "linear"],
                        "C"  : [0.1, 0.5, 1.0, 2.0]}
    paramDefaultValue = {"kernel" : "linear",
                              "C" : 1.0 }
    experiment(classifier, paramValues, paramDefaultValue)

def RunNeuralNetwork():
    classifier = "NeuralNetwork"
    hiddenLayerValues = []
    for i in (range(20, 101, 20) + range(150, 401, 50)):
        hiddenLayerValues.append(Layer("Maxout", units = i, pieces = 2))
    for i in range(20, 101, 20):
        hiddenLayerValues.append(Layer("Sigmoid", units = i, pieces = 2))
    paramValues = {"HiddenLayers" : hiddenLayerValues}
    paramDefaultValue ={"HiddenLayers" : Layer("Maxout", units = 100, pieces = 2)}
    experiment(classifier, paramValues, paramDefaultValue)



def RunGradientBoostingTree():
    classifier = "GradientBoostingTree"
    paramValues = {"max_depth": [2,4,6],
                   "learning_Rate": [0.05, 0.1, 0.15]}
    paramDefaultValue = {"max_depth": 3,
                         "learning_Rate": 0.1}
    experiment(classifier, paramValues, paramDefaultValue)

def RunRandomForest():
    classifier = "RandomForest"
    paramValues = {'n_estimators':range(5, 60, 10)}
    paramDefaultValue = {'n_estimators':10}
    experiment(classifier, paramValues, paramDefaultValue)


def RunLogisticRegression():
    classifier = "LogisticRegression"
    paramValues = {"penalty": ["l1", "l2"],
                   "C": [ 0.1, 0.5, 1.0, 2, 5]}
    paramDefaultValue = {"penalty": "l2",
                         "C": 1.0}
    experiment(classifier, paramValues, paramDefaultValue)


def RunLDA_LogisticRegression():
    classifier = "LogisticRegression"
    learner = "LDA"
    learnerParamValues = {"n_components": range(1,6)}
    learnerParamDefaultValue = {"n_components": 5}
    classifierParams = {"penalty": "l2",
                        "C": 2.0}
    experiment_withReduceDimension(classifier, classifierParams,
                learner, learnerParamValues, learnerParamDefaultValue)


def RunLDA_SVM():
    classifier = "SVM"
    learner = "LDA"
    learnerParamValues = {"n_components": range(1,6)}
    learnerParamDefaultValue = {"n_components": 5}

    classifierParams = {"kernel" : "linear",
                         "C": 1.0}
    experiment_withReduceDimension(classifier, classifierParams,
                                   learner, learnerParamValues, learnerParamDefaultValue)

def RunPCA_SVM():
    classifier = "SVM"
    learner = "PCA"
    learnerParamValues = {"n_components": range(1, 10) + range(20, 101, 10)}
    learnerParamDefaultValue = {"n_components": 5}

    classifierParams = {"kernel" : "linear",
                         "C": 1.0}
    experiment_withReduceDimension(classifier, classifierParams,
                                   learner, learnerParamValues, learnerParamDefaultValue)

def main():
    # RunSVM()  #finished

    # RunLogisticRegression() #finished

    #  RunGradientBoostingTree takes about 6hrs to finish:

    # RunGradientBoostingTree() #finished

    # RunRandomForest() #finished

    # RunNeuralNetwork() #finished

    # RunLDA_SVM()

    # RunPCA_SVM()

    RunLDA_LogisticRegression()

if __name__ == "__main__":
    main()