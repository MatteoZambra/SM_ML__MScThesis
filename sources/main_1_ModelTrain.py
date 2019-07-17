# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:56:55 2019

@author: Matteo
"""


#from keras.models import load_model

import pickle
from keras.models import load_model

import NetworkPlot as npl
import ModelTrain as mt
#import MotifsDetect as md
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



plotCov = True
plots = True
plot_net = True
model_save = False
modelInit = False
#dataSet_id = 1
nodesSubgraph = 4
mvg = False



print("\nChoose data set:\n(1) Binary Tree\n(2) Independent Clusters")
askIn = True
while (askIn == True):
    dataset_id = int(input("Only integer: "))
    if (dataset_id == 1):
        askIn = False
        dataset_id = 'TreeLev2_DS_list.pkl'
    elif (dataset_id == 2):
        askIn = False
        dataset_id = 'Clusters_DS_list.pkl'
    #end
#end

path = r'DataSets/' + dataset_id
fileID = open(path, 'rb')
DataSet = pickle.load(fileID)
fileID.close()


X = DataSet[0]
Y = DataSet[1]

print(type(X)," Dim: ",X.shape,"\n",X)
print(type(Y)," Dim: ",Y.shape,"\n",Y)
print("\n")

if plotCov == True:
    plotCovMat(X)
#end

if (modelInit == True):
    model = mdm.model_init(X,Y)
else:
    model = load_model("Model/model_init.h5")
#end

#%%
weights_pre = model.get_weights()

if (plot_net == True):
    plotNet = npl.plotNet(weights_pre, asGraph = False)
    plotNet.plotNetFunction()
#end

params_post = mt.model_train(model, X,Y, 0.3, 
                             plots, mvg)
#%%
if (plot_net == True):
    plotNet = npl.plotNet(params_post, asGraph = False)
    plotNet.plotNetFunction()
#end


#md.motifDetector(weights_pre, nodesSubgraph, plotGraph = False, cutoff = 0.25)








