import shutil
import numpy as np
import random
from sklearn import preprocessing

from config import *

def generateKfoldCrossValidataData(fold = 5, isShuffle = True, isPreProcessing = False, dataDir = dataDir, startOver = True):
    if startOver and os.path.exists(dataDir):
        # clean previously generated data
        shutil.rmtree(dataDir)

    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    # load original data
    xTrain = np.loadtxt(oriDataDir + "X_train.txt")
    yTrain = np.loadtxt(oriDataDir + "Y_train.txt")
    idTrain = np.loadtxt(oriDataDir + "subject_train.txt")


    xTest = np.loadtxt("data/X_test.txt")
    yTest = np.loadtxt("data/Y_test.txt")
    idTest = np.loadtxt("data/subject_test.txt")



    # feature normalization
    if isPreProcessing:
        # do something
        xTrain = preprocessing.scale(xTrain)
        xTest  = preprocessing.scale(xTest)

    np.save(dataDir + "xTrain.npy", xTrain)
    np.save(dataDir + "yTrain.npy", yTrain)
    np.save(dataDir + "idTrain.npy", idTrain)

    np.save(dataDir + "xTest.npy", xTest)
    np.save(dataDir + "yTest.npy", yTest)
    np.save(dataDir + "idTest.npy", idTest)

    # Train Data => Train/Dev Data
    xData = xTrain
    yData = yTrain
    idData = idTrain

    total = xData.size / xData[0].size
    index = range(total)

    # randomly shuffle the data
    if isShuffle:
       random.shuffle(index)

    # partition data for Train and Test
    for i in range(fold):
        st = total / fold * i
        ed = total / fold * (i + 1)
        if i == fold - 1:
            ed = total
        np.save(dataDir + "xTrain" + str(i) + ".npy", np.concatenate((xData[0:st],xData[ed:total])))
        np.save(dataDir + "xDev" + str(i),xData[st:ed])
        np.save(dataDir + "yTrain" + str(i) + ".npy", np.concatenate((yData[0:st],yData[ed:total])))
        np.save(dataDir + "yDev" + str(i),yData[st:ed])
        np.save(dataDir + "idTrain" + str(i) + ".npy", np.concatenate((idData[0:st],idData[ed:total])))
        np.save(dataDir + "idDev" + str(i),idData[st:ed])
        print "i = " + str(i) + ", st = " + str(st) + ", ed = " + str(ed)

    # test if the data are partitioned correctly
    for i in range(fold):
        xTrain = np.load(dataDir + "xTrain" + str(i) + ".npy")
        print "i = " + str(i)
        print " train data set:  number of data points = " + str(xTrain.size/xTrain[0].size) \
              + ", number of features = " + str(xTrain[0].size)
        xTest = np.load(dataDir + "xDev" + str(i) + ".npy")
        print " test data set:  number of data points = " + str(xTest.size/xTest[0].size) \
              + ", number of features = " + str(xTest[0].size)



def main():
    generateKfoldCrossValidataData(fold = crossValidationFold, isShuffle=False)

if __name__ == "__main__":
    main()