## read data
import numpy as np
from config import *


Xtrain = np.loadtxt(oriDataDir + 'X_train.txt')
Ytrain = np.loadtxt(oriDataDir + 'Y_train.txt')
IDtrain = np.loadtxt(oriDataDir + 'subject_train.txt')

Xtest = np.loadtxt(oriDataDir + 'X_test.txt')
Ytest = np.loadtxt(oriDataDir + 'Y_test.txt')
IDtest = np.loadtxt(oriDataDir + 'subject_test.txt')




########### our revised method based on logistic regression
## first train the original losigtic regression model
#### logistic regression
from collections import Counter
from sklearn import linear_model
clf4 = linear_model.LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')   
clf4.fit(Xtrain, Ytrain)


allAccuracy = []
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    predictClass = []
    accuracy= 0
    for i in range(len(Xtest)):
        if(len(predictClass)>=5):
            count = Counter(predictClass[-5:])
            avg = count.most_common()[0][0]
            prob = clf4.predict_proba(Xtest[i].reshape(1,-1))[0]
            if(avg<1.5):
                prob[0] = prob[0] + sum(prob[1:])*alpha
                prob[1:] = prob[1:]*(1-alpha)
            if(avg>=1.5 and avg<2.5):
                prob[1] = prob[1] + (sum(prob[:1])+sum(prob[2:]))*alpha
                prob[:1] = prob[:1]*(1-alpha)
                prob[2:] = prob[2:]*(1-alpha)
            if(avg>=2.5 and avg<3.5):
                prob[2] = prob[2] + (sum(prob[:2])+sum(prob[3:]))*alpha
                prob[:2] = prob[:2]*(1-alpha)
                prob[3:] = prob[3:]*(1-alpha)
            if(avg>=3.5 and avg<4.5):
                prob[3] = prob[3] + (sum(prob[:3])+sum(prob[4:]))*alpha
                prob[:3] = prob[:3]*(1-alpha)
                prob[4:] = prob[4:]*(1-alpha)
            if(avg>=4.5 and avg<5.5):
                prob[4] = prob[4] + (sum(prob[:4])+sum(prob[5:]))*alpha
                prob[:4] = prob[:4]*(1-alpha)
                prob[5:] = prob[5:]*(1-alpha)
            if(avg>=5.5):
                prob[5] = prob[5] + sum(prob[:5])*alpha
                prob[:5] = prob[:5]*(1-alpha)
            predictClass.append(prob.tolist().index(max(prob))+1)
    
            
        else:
            predictClass.append(clf4.predict(Xtest[i])[0])
        
        
        accuracy = accuracy + (predictClass[-1] == Ytest[i])    
    allAccuracy.append(accuracy/(1.0*len(Xtest)))
    
print "The accuracy:"   
print allAccuracy

# the direct accuracy on testing data is
print "the direct accuracy on testing data is:"
print clf4.score(Xtest, Ytest)   # 0.960298608755

