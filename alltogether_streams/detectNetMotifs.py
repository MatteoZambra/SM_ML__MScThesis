# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 06:59:20 2019

@author: Matteo
"""

import pickle
import motifsDetect as md
import networkPlot as npl

plot_net = True
nodesSubgraph = 4
dataSet_clean = True
SG = False

if (SG == False):
    fileID = open(r'params_MVG.pkl', 'rb')
    params = pickle.load(fileID)
    fileID.close()
else:
    if (dataSet_clean == True):
        fileID = open(r'params_clean_SG.pkl', 'rb')
        params = pickle.load(fileID)
        fileID.close()
    else:
        fileID = open(r'params_noise_SG.pkl', 'rb')
        params = pickle.load(fileID)
        fileID.close()
    #end
#end


if (plot_net == True):
    plotNet = npl.plotNet(params, asGraph = False)
    plotNet.plotNetFunction()
#end

md.motifDetector(params, nodesSubgraph, plotGraph = False)
