# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:55:57 2019

@author: Matteo
"""

import plotFunctions as pfn

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set(color_codes = True)

import numpy as np
import pickle

import networkPlot as npl



plot_net = False

fileID = open(r'DataSet_list_clean.pkl', 'rb')

DataSet_clean = pickle.load(fileID)
fileID.close()

fileID = open(r'DataSet_list_noise.pkl', 'rb')

DataSet_noise = pickle.load(fileID)
fileID.close()

X_clean = DataSet_clean[0]
Y_clean = DataSet_clean[1]

X_noise = DataSet_noise[0]
Y_noise = DataSet_noise[1]



Xtrain_clean, Xtest_clean, Ytrain_clean, Ytest_clean = train_test_split(
        X_clean, Y_clean, test_size = 0.3, random_state = 20)

Xtrain_noise, Xtest_noise, Ytrain_noise, Ytest_noise = train_test_split(
        X_noise, Y_noise, test_size = 0.3, random_state = 20)


model = load_model('model.h5')

if (plot_net == True):
    plotNet = npl.plotNet(model.get_weights(), asGraph = False)
    plotNet.plotNetFunction()
#end

es1 = EarlyStopping(monitor='val_acc', mode='auto', patience = 30, verbose = 1)
es2 = EarlyStopping(monitor='val_loss', mode='auto',patience = 20, verbose = 1)

sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.6, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

params = model.get_weights()
paramsInit = params

for I in range(10):
    
    print("\nSuperepoch: {}, goal 1\n".format(I))
    model.set_weights(params)
    model.fit(Xtrain_clean, Ytrain_clean,
          validation_split = 0.1, epochs = 100, verbose = 0, callbacks = [es1, es2])
    print("Model evaluation on test data: loss and accuracy\n",
      model.evaluate(Xtest_clean,Ytest_clean, verbose = 1))
    params = model.get_weights()
    
    print("\nSuperepoch: {}, goal 2\n".format(I))
    model.set_weights(params)
    model.fit(Xtrain_noise, Ytrain_noise,
          validation_split = 0.1, epochs = 100, verbose = 0, callbacks = [es1, es2])
    print("Model evaluation on test data: loss and accuracy\n",
      model.evaluate(Xtest_noise,Ytest_noise, verbose = 1))
    params = model.get_weights()
    
#end

fileID = open(r'params_MVG.pkl', 'wb')
pickle.dump(params, fileID)
fileID.close()


paramsInit = np.asarray(paramsInit)
params = np.asarray(params)
numLayers = len(model.layers)
pfn.jointPlotter(numLayers, paramsInit, params, plot_hist = True)
pfn.jointPlotter(numLayers, paramsInit, params, plot_hist = False)










