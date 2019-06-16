# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:56:55 2019

@author: Matteo
"""


from keras.models import load_model

import pickle

import NetworkPlot as npl
import ModelTrain as mt
import MotifsDetect as md
import ModelInit as mdm



plots = True
plot_net = True
dataSet_lev = False
nodesSubgraph = 4


if (dataSet_lev == True):
    fileID = open(r'DataSets_levels/DataSet_list_lev2.pkl', 'rb')
    DataSet = pickle.load(fileID)
    fileID.close()
else:
    fileID = open(r'DataSets_levels/DataSet_list_lev4.pkl', 'rb')
    DataSet = pickle.load(fileID)
    fileID.close()
#end


X = DataSet[0]
Y = DataSet[1]

print(type(X)," Dim: ",X.shape,"\n",X)
print(type(Y)," Dim: ",Y.shape,"\n",Y)
print("\n")


model_zero = load_model('Model/model.h5')
layer1_lev2 = model_zero.get_weights()[0]
layer2_lev2 = model_zero.get_weights()[2]
model = mdm.model_lastLayer(model_zero, Y.shape[1])


#%%
weights_pre = model_zero.get_weights()

if (plot_net == True):
    plotNet = npl.plotNet(weights_pre, asGraph = False)
    plotNet.plotNetFunction()
#end

params_post = mt.model_train(model, X,Y, 0.3, plots, mvg = True)

#%%
if (plot_net == True):
    plotNet = npl.plotNet(params_post, asGraph = False)
    plotNet.plotNetFunction()
#end

md.motifDetector(params_post, nodesSubgraph, plotGraph = False, cutoff = 0.35)








