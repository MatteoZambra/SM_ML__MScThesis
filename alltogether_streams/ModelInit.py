
"""
This script is run to initialize the model, except for the last layer, 
which needs to be created according to the data set chosen.
All layers but the last one are initialized, model saved as h5 file and
the fetched to coomplete the initialization.

This script is called by the main method if the user wants the model to be
initialized fully from scratch. Otherwise, the model previously saved is loaded
and the last layer added according to the number of output neurons
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal, Orthogonal

# note: RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
#       Orthogonal(gain = 1.0, seed = None)

import pickle


def model_init(M):
    
    print("Model initialise. Only first two layers")
    
    model = Sequential()
    
    model.add(Dense(input_dim = M, units = 20,
                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                bias_initializer = RandomNormal(mean = 0.0, 
                                                stddev = 0.1, 
                                                seed = None),
                activation = 'relu'))
    model.add(Dense(input_dim = M, units = 10,
                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                bias_initializer = RandomNormal(mean = 0.0, 
                                                stddev = 0.1, 
                                                seed = None),
                activation = 'relu'))
                
    model.save("Model/model.h5")
#enddef
    
def model_lastLayer(model,nCat):
    
    model.add(Dense(units = nCat,
                    kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                    bias_initializer = RandomNormal(mean = 0.0,
                                                    stddev = 0.1,
                                                    seed = None),
                    activation = 'softmax'))
    
    return model
#enddef



# ============================================================================

dataSet_clean = False

if (dataSet_clean == True):
    fileID = open(r'DataSets_levels/DataSet_list_lev2.pkl', 'rb')
else:
    fileID = open(r'DataSets_levels/DataSet_list_lev4.pkl', 'rb')
#end

DataSet = pickle.load(fileID)
fileID.close()

X = DataSet[0]
M = X.shape[1]

#model_init(M)
