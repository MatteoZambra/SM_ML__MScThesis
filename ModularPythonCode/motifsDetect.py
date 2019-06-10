# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:08:44 2019

@author: Matteo
"""

import numpy as np
import subprocess
import os
import networkPlot as npl

def motifDetector(weights, nodesSubgraph, plotGraph):

    _weights = np.asarray(weights)
    
    numLayers = int(_weights.shape[0]/2)
    wghts = []
    biases = []
    
    for i in range(numLayers):
        j = 2*i
        print(j,(_weights[j].T).shape)
        wghts.append(_weights[j])
        j = 2*i + 1
        print(j,(_weights[j].T).shape)
        biases.append(_weights[j])
    #enddo
    
    wghts = np.asarray(wghts)
    
    layers = {}
    
    nodes = np.arange(1,wghts[0].shape[0]+1)
    dictTmp = {1 : nodes}
    layers.update(dictTmp)
    
    for i in range(1,numLayers):
        
        offset = layers[i][-1] + 1
        nodes = np.arange(offset, wghts[i].shape[0] + offset)
        dictTmp = {i+1 : nodes}
        layers.update({i+1 : nodes})
        
        if (i == numLayers-1):
            
            offset = layers[i+1][-1] + 1
            layers.update({i+2 : np.arange(offset, wghts[i].shape[1] + offset)})
        #endif
    #enddo
        
    cutoff = 0.35
    
    links = wghts
    
    for k in range(wghts.shape[0]):
        for i in range(wghts[k].shape[0]):
            for j in range(wghts[k].shape[1]):
                if (wghts[k][i,j] <= -cutoff or wghts[k][i,j] >= cutoff):
                    links[k][i,j] = 1
                else:
                    links[k][i,j] = 0
                #endif
            #enddo
        #enddo
    #enddo
    
    if (plotGraph == True):
        plotNet = npl.plotNet(weights, asGraph = True)
        plotNet.plotNetFunction()
    #end
        
    AdjMat = []
    
    for k in range(numLayers):
        iRange = range(links[k].shape[0])
        jRange = range(links[k].shape[1])
        for i in iRange:
            for j in jRange:
                if (links[k][i,j] != 0.0):
                    toAdd = layers[k+2][j]
                    listElem = layers[k+1][i]
                    AdjMat.append([listElem,toAdd,1])
                #endif
            #enddo
        #enddo
    #enddo
    
    print(len(AdjMat))
    print(AdjMat[0])
    
    """
    with open('inputGraph.txt', 'w') as f:
        for row in range(len(AdjMat)):
            f.write("%s\n" % AdjMat[row])
    """
    
    with open('inputGraph.txt', 'w') as f:
        for i in range(len(AdjMat)):
            for j in range(len(AdjMat[i])):
                f.write("%s " % AdjMat[i][j])
            #enddo
            f.write("\n")
        #enddo
    #endwith
    
    strIn = ['../UA_prog/mfinder1.2/mfinder1.2.exe','inputGraph.txt', '-s', str(nodesSubgraph)]
    mfinderRep = subprocess.run(strIn, stdout = subprocess.PIPE).stdout.decode('utf-8')
    print(mfinderRep)
        
    os.system("inputGraph_OUT.txt")
    
#end
    
    
    
    
    
    
    
    
    
    