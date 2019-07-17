
"""
Retrieve models (init and trained) from disk.
Then, all that lays is rows 18 -> end, is what there is
objectively coded in the proGraphDataStructure.py script
"""

#import keras
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import spectrumSplit as ssp
import proGraphDataStructure as pg


askIn = True
while (askIn == True):
    write_file = input("Write file? (Y/N): ")
    if (write_file == "Y" or write_file == "N" or\
        write_file == "y" or write_file == "n"):
        askIn = False
    #end
#end
if (write_file == "Y" or write_file == "y"):
    print("Graph file is being written")
else:
    print("No graph file writing")
#end



path = os.getcwd()
path += r"\Model\model_trained_mvg.h5"
model = load_model(path)


graph = pg.proGraph(model)

Edges = graph.GetEdges()
Nodes = graph.GetNodes()

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
                                   colors = 7)
print(counts)
NodesDF = ssp.CategoriseWeightsBiases(NodesDF,binsBiases)




if (write_file == "Y" or write_file == "y"):
    print("Writing Graph File\n")
    
    AdjLists = graph.GetAdjLists()
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
    f.close()
#end



