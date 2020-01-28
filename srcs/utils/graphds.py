"""
  
  Initialisation by means of the call
  
      model = keras.models.load_model("model.h5")
      
  model is exported from keras.models
  
      g = proGraph(model)
      
  g contains
  
      . Nodes  : dictionary with {ID : deg(ID)}
      . Edges  : dictionary with {ID : (i,j) , weight(i,j)}
      . AdjMat : list of adjacency lists

"""

import numpy as np
from random import sample


class proGraph:
    
    def __init__(self,model):
    
        """
        due to the way keras saves the model, the following
        serves the purpose to save weights and biases in different
        data structures. In particular, keras.model.get_weights()
        returns a list. 
        Assume we have a network with input layer, hidden layer and 
        output layer. Then
            
            params = model.get_weights()
        
        contains [W1, b1, W2, b2], where 
        
            ~ W1 \in R^{N_hidden x N_input}, 
            ~ b1 \in R^{N_hidden}
            ~ W2 \in R^{N_output x N_hidden},
            ~ b2 \in R^{N_output}
            
        Thus W1 = params[0], W2 = params[2], b1 = params[1], b2 = params[3].
        Even indices of params contain weights matrices and odd indices point
        to biases vectors. Both matrices and vectors are np.ndarray types.
        
        The graph data structure is represented in terms of lists and
        dictionaries, for the purposes of this work.
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
            """
            l = 0, ... , numLayers-1.
            j assumes even and odd values alternatively. Accordingly, weights
            matrices and biases are stored in proper lists.
            """
            
            j = 2*l
            weights.append(params[j])
            
            j = 2*l + 1
            biases.append(params[j])
        #enddo
        
        # layers 
        
        layers = {}
        
        # FIRST LAYER
        nodes = np.arange(1, weights[0].shape[0]+1)
        dictTmp = {1 : nodes}
        layers.update(dictTmp)
        
        # SUBSEQUENT LAYERS
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
        
        """
        For the purposes of this work, a nested loop is sustainable.
        This whole class should be revised in case of larger models
        """
        for k in range(numLayers):
            iRange = range(weights[k].shape[0])
            jRange = range(weights[k].shape[1])
            offset = layers[k+1][-1]
            for i in iRange:
                for j in jRange:
                    toAdd = layers[k+2][j]
                    listElem = layers[k+1][i]
                    AdjMat[listElem-1].append(toAdd)
                    Edges.update({ne : [(listElem, toAdd), listElem-1, toAdd-1,
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
        Not used.
        The FANMOD motifs mining tool gives the possibility to search for 
        either weighted or unweighted networks. If the latter, then the 
        third column of the graph.txt file (containing the categories 
        associated to the edge encoded by the respective row -- source and
        target nodes) is neglected.
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
