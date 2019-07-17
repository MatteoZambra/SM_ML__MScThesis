"""
  eeeeeesticazzi
  intanto è fatta. 
  Per inizializzare gli dai un oggetto model, cioè
  
      model = keras.models.load_model("model.h5")
      
  gli passi sta cosa,
  
      g= proGraph(model)
      
  e quindi in g abbiamo
  
      . Nodes: dizionario con {ID : deg(ID)}
      . Edges : Pandas DataFrame con {ID : (i,j) , weight(i,j)}
      . AdjMat : lista di liste

"""

#from keras.models import load_model
import numpy as np
import pandas as pd
from random import sample


class proGraph:
    
    def __init__(self,model):
        
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
        Edges = {}
        ne = 1
        
        sizeN = layers[numLayers][-1]
        for k in range(sizeN):
            adj = []
            AdjMat.append(adj)
        #end
        
        for k in range(numLayers):
            iRange = range(weights[k].shape[0])
            jRange = range(weights[k].shape[1])
            for i in iRange:
                for j in jRange:
                    toAdd = layers[k+2][j]
                    listElem = layers[k+1][i]
                    AdjMat[listElem-1].append(toAdd)
                    Edges.update({ne : [(listElem, toAdd), 
                                        weights[k][i,j]]})
                    ne += 1
                #end
            #end
        #end
        
        Edges= pd.DataFrame.from_dict(Edges, orient = "index",
                                      columns = ["edge", "strength"])
        
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
        
#        Nodes degrees :: nooo
#        Nodes = {}
#        for _,idx in enumerate(layers):
#            
#            [Nodes.update({layers[idx][i]:0}) for i in range(len(layers[idx]))]
#        #end
#        
#        for _,k in enumerate(layers):
#            print(k)
#            if (k <= numLayers):
#                nodLayer = layers[k].tolist()
#                for i in nodLayer:
#                    print(i)
#                    print(AdjMat[i-1])
#                    Nodes[i] = len(AdjMat[i-1])
#                #end
#            #end
#        #end
        
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
    
    
    
    def moveFrom(self, i, AdjLists, Edges, Nodes):
    
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
    
    
#    
#model = load_model(r"C:\Users\Matteo\Desktop\MasterThesis\newThread\alltogether_streams\Model\model_init.h5")