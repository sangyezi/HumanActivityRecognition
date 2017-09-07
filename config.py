import inspect
import os
"""
define global variables needed for configurations in the packaage
"""

baseDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# used for storing original data
oriDataDir =  baseDir + "/" + "data/"

# used for storing cleaned data
dataDir = baseDir + "/" + "data/npy/"

# used for storing pcaReducedData
pcaRedcuedDataDir = baseDir + "/" +"data/npy/PCAReduced/"

# used for storing ldaReducedData
ldaReducedDataDir = baseDir + "/" + "data/npy/LDAReduced/"

# used for storing experiment result
expResultDir = baseDir + "/" + "expResult/"


crossValidationFold = 5
