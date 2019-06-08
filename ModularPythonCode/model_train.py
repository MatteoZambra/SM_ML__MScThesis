# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:27:27 2019

@author: Matteo
"""

"""
Adesso i pesi init sono stati creati, quindi non li tocchiamo più. 
Qui li carichiamo ed effettivamente trainiamo il modello. Poi magari non
serve nemmeno esportare tutto il model daccapo, basterebbero solo i pesi

"""

import plotFunctions as pfn

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping

# note: RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
#       Orthogonal(gain = 1.0, seed = None)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

#from scipy import stats
import numpy as np
import pickle





plots = True
dataSet_clean = True

if (dataSet_clean == True):
    fileID = open(r'DataSet_list_clean.pkl', 'rb')
else:
    fileID = open(r'DataSet_list_noised.pkl', 'rb')
#end

DataSet = pickle.load(fileID)
fileID.close()

X = DataSet[0]
Y = DataSet[1]

print(type(X)," Dim: ",X.shape,"\n",X)
print(type(Y)," Dim: ",Y.shape,"\n",Y)
print("\n")

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = 20)

print("X: train dims = {}, test dims = {}\n".format(Xtrain.shape, Xtest.shape))
print("Y: train dims = {}, test dims = {}\n".format(Ytrain.shape, Ytest.shape))

M = Xtrain.shape[1]
nCat = Ytrain.shape[1]


model = load_model("model.h5")

es1 = EarlyStopping(monitor='val_acc', mode='auto', patience = 30, verbose = 1)
es2 = EarlyStopping(monitor='val_loss', mode='auto',patience = 20, verbose = 1)

sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.6, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

history = model.fit(Xtrain, Ytrain, 
                    validation_split = 0.1, epochs = 100, verbose = 1, callbacks = [es1,es2])


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


print("Model evaluation on test data: loss and accuracy\n",
      model.evaluate(Xtest,Ytest, verbose = 1))


weights_post = np.asarray(model.get_weights())
if dataSet_clean == True:
    fileID = open(r'weights_clean.pkl', 'wb')
    pickle.dump(weights_post, fileID)
    fileID.close()
else:
    fileID = open(r'weights_noisy.pkl', 'wb')
    pickle.dump(weights_post, fileID)
    fileID.close()
#end


if (plots == True):
    fileID = open(r'weights_pre.pkl', 'rb')
    weights_pre = pickle.load(fileID)
    fileID.close()

    numLayers = len(model.layers)
    pfn.jointPlotter(numLayers, weights_pre, weights_post, plot_hist = True)
    pfn.jointPlotter(numLayers, weights_pre, weights_post, plot_hist = False)
#end
    


