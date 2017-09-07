## read data
from config import *

import numpy as np


Xtrain = np.loadtxt(oriDataDir + 'X_train.txt')
Ytrain = np.loadtxt(oriDataDir + 'Y_train.txt')
IDtrain = np.loadtxt(oriDataDir + 'subject_train.txt')

Xtest = np.loadtxt(oriDataDir + 'X_test.txt')
Ytest = np.loadtxt(oriDataDir + 'Y_test.txt')
IDtest = np.loadtxt(oriDataDir + 'subject_test.txt')





#### feature importances with forest of trees
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(Xtrain, Ytrain)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(Xtrain.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(Xtrain.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(Xtrain.shape[1]), indices)
plt.xlim([-1, Xtrain.shape[1]])
plt.show()