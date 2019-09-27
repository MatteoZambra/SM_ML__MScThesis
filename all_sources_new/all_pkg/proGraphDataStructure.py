"""
  
  Initialisation by means of the call
  
      model = keras.models.load_model("model.h5")
      
  model is exported from keras.models
  
      g= proGraph(model)
      
  g contains
  
      . Nodes: dictionary with {ID : deg(ID)}
      . Edges : Pandas DataFrame with {ID : (i,j) , weight(i,j)}
      . AdjMat : list of adjacency lists

"""

import numpy as np
import pandas as pd
from random import sample


class proGraph:
    
    def __init__(self,model):
    
        """
        due to the way keras saves the model, the following
        serves the purpose to save weights and biases in different
        data structures.
        
        The graph data structure, in terms of adjacency lists and,
        for ease, pandas DataFrames, is subsequently saved
        """
        
        self.layers = {}
        self.Nodes = {}
        self.Edges = {}
        self.AdjMat = []
        self.AdjLists = {}
        
        params = np.asarray(model.get_weights())
        numLayers = int(len(params)/2)
        
        weights = []
        biases = []
        
        for l in range(numLayers):
            
            j = 2*l
            weights.append(params[j])
            
            j = 2*l + 1
            biases.append(params[j])
        #enddo
        
        # layers 
        
        layers = {}
        
        nodes = np.arange(1, weights[0].shape[0]+1)
        dictTmp = {1 : nodes}
        layers.update(dictTmp)
        
        for i in range(1, numLayers):
            
            offset = layers[i][-1] + 1
            nodes = np.arange(offset, weights[i].shape[0] + offset)
            dictTmp = {i+1 : nodes}
            layers.update(dictTmp)
            
            if (i == numLayers-1):
                offset = layers[i+1][-1] + 1
                dictTmp = {i+2 : np.arange(offset, weights[i].shape[1] + offset)}
                layers.update(dictTmp)
            #end
        #end
                
        
        # edges and adjacency lists
        
        AdjMat = []
        Nodes = {}
        Edges = {}
        ne = 1
        
        sizeN = layers[numLayers][-1]
        for k in range(sizeN):
            adj = []
            AdjMat.append(adj)
        #end
        
        for k in range(layers[1][-1]):
            Nodes[k+1] = 0
        #end
        
        for k in range(numLayers):
            iRange = range(weights[k].shape[0])
            jRange = range(weights[k].shape[1])
            offset = layers[k+1][-1]
            for i in iRange:
                for j in jRange:
                    toAdd = layers[k+2][j]
                    listElem = layers[k+1][i]
                    AdjMat[listElem-1].append(toAdd)
                    Edges.update({ne : [(listElem, toAdd), 
                                        float(weights[k][i,j]) ]})
                    Nodes[offset+j+1] = float(biases[k][j])
                    ne += 1
                #end
            #end
        #end
            
        idx = 1
        AdjLists = {}
        
        for adjs in AdjMat:
            AdjLists.update({idx : adjs})
            idx += 1
        #end
        
        
        self.layers = layers
        self.Nodes = Nodes
        self.Edges = Edges
        self.AdjMat = AdjMat
        self.AdjLists = AdjLists
    
    #end
    
    
    def GetNodes(self):
        
        return self.Nodes
    #end
    
    def GetEdges(self):
        
        return self.Edges
    #end
    
    def GetAdjLists(self):
        
        return self.AdjLists
    #end
    
    def BinarizeGraph(self,Edges,cutoff):
    
        """
        Not used
        """
        
        for _,i in enumerate(Edges):
            if (Edges[i][1] < -cutoff or \
                Edges[i][1] > cutoff):
                Edges[i][1] = 1
            else:
                Edges[i][1] = 0
            #end
        
        return Edges
    #end
            
    
    
    def moveFrom(self, i, AdjLists, Edges, Nodes):
    
        """
        Probabilistic decision: multiply each possible
        pathway weight for a random number in [0,1]. 
        The walk goes where the greaters value is
        """
    
        weights = []
        listRun = AdjLists[i] 
        for j in listRun:
            w = Edges.loc[Edges["edge"] == (i,j)].iloc[0]["strength"]
            
            p = np.random.uniform()
            weights.append(w*p)
        #end
        
        return listRun[weights.index(max(weights))]
    #end
    
    def RandomWalk(self, layers, AdjLists, Nodes, Edges):
    
        """
        Random walk. It was an idea for the inspection
        of preferential walks on the graph, but not used
        """
        
        edgePassed = []

        for I in range(500):
            
            edgePassedWalk = []
            
            print("Walk ",I)
            go = True
            iPrev = int(sample(layers[1].tolist(),1)[0])
            
            while (go == True):
                
                iNext = self.moveFrom(iPrev, AdjLists, Edges, Nodes)
                if (iNext in layers[4].tolist()):
                    go = False
                #end
                
                edgePassedWalk.append( (iPrev, iNext) )
                
                idxList = Edges.index[Edges["edge"] == (iPrev,iNext)].tolist()
                idx = idxList[0]
                Edges.at[idx,"pass"] += 1
                
                iPrev = iNext
            #end
            edgePassed.append(edgePassedWalk)
        #end
    #end
#end
