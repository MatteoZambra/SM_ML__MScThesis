# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:56:55 2019

@author: Matteo
"""
import plotFunctionsExp as pfn

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

import numpy as np
import pickle

import networkPlot as npl


plot_net_init = False
plots = True
dataSet_clean = True

if (dataSet_clean == True):
    fileID = open(r'DataSet_list_clean.pkl', 'rb')
    DataSet = pickle.load(fileID)
    fileID.close()
else:
    fileID = open(r'DataSet_list_noise.pkl', 'rb')
    DataSet = pickle.load(fileID)
    fileID.close()
#end


X = DataSet[0]
Y = DataSet[1]

print(type(X)," Dim: ",X.shape,"\n",X)
print(type(Y)," Dim: ",Y.shape,"\n",Y)
print("\n")

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size = 0.3, random_state = 20)

model = load_model('model.h5')

if (plot_net_init == True):
    plotNet = npl.plotNet(model.get_weights(), asGraph = False)
    plotNet.plotNetFunction()
#end

es1 = EarlyStopping(monitor='val_acc', mode='auto', patience = 30, verbose = 1)
es2 = EarlyStopping(monitor='val_loss', mode='auto',patience = 20, verbose = 1)

sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.6, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

history = model.fit(Xtrain, Ytrain, 
                    validation_split = 0.1, epochs = 100, verbose = 1, callbacks = [es1,es2])

if (plots == True):
    sns.set_style("ticks")
    plt.figure(figsize=(8,6))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
#end


print("Model evaluation on test data: loss and accuracy\n",
    model.evaluate(Xtest,Ytest, verbose = 1))


params = model.get_weights()

if dataSet_clean == True:
    fileID = open(r'params_clean_SG.pkl', 'wb')
    pickle.dump(params, fileID)
    fileID.close()
else:
    fileID = open(r'params_noise_SG.pkl', 'wb')
    pickle.dump(params, fileID)
    fileID.close()
#end


if (plots == True):
    fileID = open(r'weights_pre.pkl', 'rb')
    weights_pre = pickle.load(fileID)
    fileID.close()
    weights_post = np.asarray(params)
    numLayers = len(model.layers)
    pfn.jointPlotter(numLayers, weights_pre, weights_post, plot_hist = True)
    pfn.jointPlotter(numLayers, weights_pre, weights_post, plot_hist = False)
#end




