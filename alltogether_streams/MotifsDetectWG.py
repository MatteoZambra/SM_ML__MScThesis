
"""
Retrieve models (init and trained) from disk.
Then, all that lays is rows 18 -> end, is what there is
objectively coded in the proGraphDataStructure.py script
"""

#import keras
from keras.models import load_model
import numpy as np
import pandas as pd
import spectrumSplit as ssp


#%%
model = load_model(r"C:\Users\Matteo\Desktop\MasterThesis\newThread\alltogether_streams\Model\model_init.h5")


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

EdgesDF = pd.DataFrame.from_dict(Edges, orient = "index",
                               columns = ["edge", "param"])


NodesDF = pd.DataFrame.from_dict(Nodes, orient = "index",
                               columns = ["param"])

weights = np.asarray(EdgesDF["param"])
biases = np.asarray(NodesDF["param"])

counts,binsWeights = ssp.smartBins(weights, 
                                   bwMethod = 'silverman',
                                   factor = 0.5,
                                   colors = 7)
print(counts)
EdgesDF = ssp.CategoriseWeightsBiases(EdgesDF,binsWeights)

counts,binsBiases  = ssp.smartBins(biases, 
                                   bwMethod = 'silverman',
                                   factor = 0.33,
                                   colors = 8)
print(counts)
NodesDF = ssp.CategoriseWeightsBiases(NodesDF,binsBiases)


with open('inputColoredGraph.txt','w') as f:
    for _,i in enumerate(AdjLists):
        
        for j in range(len(AdjLists[i])):
            l = AdjLists[i][j]
            aspe = EdgesDF.loc[EdgesDF["edge"] == (i,l)]
            f.write("%s %s " % (i-1, l-1))
            f.write("%s %s %s " % (int(NodesDF.loc[i,"cats"]),
                                   int(NodesDF.loc[l,"cats"]),
                                   int(aspe.at[aspe.index[0],"cats"])))
            f.write("\n")
        #end
    #end
#

f.close()





