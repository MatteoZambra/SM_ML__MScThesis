# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:56:55 2019

@author: Matteo
"""


#from keras.models import load_model

import pickle

import NetworkPlot as npl
import ModelTrain as mt
import MotifsDetect as md
import ModelInit as mdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_style("ticks")


#%%
def plotCovMat(X):
    X_df = pd.DataFrame(X)
    covMat = X_df.cov()
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    im = ax.matshow(covMat)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im,cax=cax)
#    plt.show()
#enddef

#%%



plotCov = False
plots = True
plot_net = True
dataSet_id = 2
nodesSubgraph = 4


if (dataSet_id == 1):
    fileID = open(r'DataSets/TreeLev2_DS_list.pkl', 'rb')
    DataSet = pickle.load(fileID)
    fileID.close()
else:
    fileID = open(r'DataSets/Clusters_DS_list.pkl', 'rb')
    DataSet = pickle.load(fileID)
    fileID.close()
#end

X = DataSet[0]
Y = DataSet[1]

print(type(X)," Dim: ",X.shape,"\n",X)
print(type(Y)," Dim: ",Y.shape,"\n",Y)
print("\n")

if plotCov == True:
    plotCovMat(X)
#end

#model_zero = load_model('Model/model.h5')
#layer1_lev2 = model_zero.get_weights()[0]
#layer2_lev2 = model_zero.get_weights()[2]
#model = mdm.model_lastLayer(model_zero, Y.shape[1])

model = mdm.model_init(X,Y)


#%%
weights_pre = model.get_weights()

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








