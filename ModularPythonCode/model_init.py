# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:27:58 2019

@author: Matteo
"""

import plotFunctions as pfn


from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal, Orthogonal

# note: RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
#       Orthogonal(gain = 1.0, seed = None)

import seaborn as sns
sns.set(color_codes = True)

import numpy as np

import pickle

# ============================================================================



dataSet_clean = True

if (dataSet_clean == True):
    fileID = open(r'DataSet_list_clean.pkl', 'rb')
else:
    fileID = open(r'DataSet_list_noise.pkl', 'rb')
#end

DataSet = pickle.load(fileID)
fileID.close()

X = DataSet[0]
Y = DataSet[1]

M = X.shape[1]
nCat = Y.shape[1]


# ___________________________________________

model = Sequential()

normal_init = RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
#orth_init = Orthogonal(gain = 1.0, seed = None)

model.add(Dense(input_dim = M, units = 20,
                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                bias_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None),
                activation = 'relu'))
model.add(Dense(input_dim = M, units = 10,
                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                bias_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None),
                activation = 'relu'))
model.add(Dense(units = nCat,
                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                bias_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None),
                activation = 'softmax'))

weights_pre = np.asarray(model.get_weights())
fileID = open(r'weights_pre.pkl', 'wb')
pickle.dump(weights_pre, fileID)
fileID.close()

numLayers = len(model.layers)


pfn.plotter(weights_pre, numLayers, plot_hist = False)
pfn.plotter(weights_pre, numLayers, plot_hist = True)



model.save("model.h5")

"""
NOTA: qui si può pensare di esportare il model di Keras come hf5, in modo
che fino a qui sono fatti i pesi init, quindi non si toccano più, cioè non
si inizializzano più. Poi: nell'altro file apro direttamente i pesi init
"""
