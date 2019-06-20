# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:19:53 2019

@author: Matteo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pandas as pd
import networkx as nx
import pickle

#%%

#plt.rc('text', usetex = True)
#plt.rc('font', family = 'sans-serif')


def scatterPlot(vertPerClass, verts):
    
    vertsDf = pd.DataFrame.from_dict(verts, 
                               orient = 'index', 
                               columns = ['x1', 'x2', 'class'])

    fig = plt.figure(figsize=(10,8))

    for k in range(len(vertPerClass)):
    
        plt.scatter(vertsDf[vertsDf['class']==k+1]['x1'],
                    vertsDf[vertsDf['class']==k+1]['x2'],
                    s = 80)
#       plt.xlabel(r"$x_{1}$", fontsize = 15)
#       plt.ylabel(r"$x_{2}$", fontsize = 15)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Generated points scatterplot")
    #end
    plt.show()
    
    return fig
#enddef

def plotGraph(vertPerClass, verts, edges, edgesStrengths):
    
    vertsDf = pd.DataFrame.from_dict(verts, 
                               orient = 'index', 
                               columns = ['x1', 'x2', 'class'])

    fig = plt.figure(figsize=(10,8))

    for k in range(len(vertPerClass)):
    
        plt.scatter(vertsDf[vertsDf['class']==k+1]['x1'],
                    vertsDf[vertsDf['class']==k+1]['x2'],
                    s = 80)
#       plt.xlabel(r"$x_{1}$", fontsize = 15)
#       plt.ylabel(r"$x_{2}$", fontsize = 15)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Generated points graph")
    #end
    
    for edge in edges:
        
        x_coord = (verts[edge[0]][0] , verts[edge[1]][0])
        y_coord = (verts[edge[0]][1] , verts[edge[1]][1])
        
        strength = edgesStrengths[edge]
        line = lines.Line2D(x_coord, y_coord, linewidth = strength * 1.5,
                            color = 'k')
        fig.gca().add_line(line)
            
    fig.show()
#enddef




moments = [
            [ [1,1],   [[10, 0.0],[0.0, 10]] ],
            [ [1,51],  [[10, 0.0],[0.0, 10]] ],
            [ [51,1],  [[10, 0.0],[0.0, 10]] ],
            [ [51,51], [[10, 0.0],[0.0, 10]] ],
            [ [25,25], [[10, 0.0],[0.0, 10]] ]
        ]

#vertPerClass = np.random.randint(8,25, size=4)
vertPerClass = [5, 5, 5, 5]
verts = {}

#%% creazione dell'insieme V di G

cnt = 0
for k in range(len(vertPerClass)):
    
    for i in range(vertPerClass[k]):
        
        x = np.random.multivariate_normal(moments[k][0],
                                          moments[k][1],
                                          (1,)).reshape((2,))
        
        verts.update({cnt+1 : [x[0], x[1] , k+1]})
        cnt += 1
    #end
#end

scatterPlot(vertPerClass, verts)

#%% edges
    
listVert = list(verts.keys())
N = len(listVert)
    
G = nx.Graph()
G.add_nodes_from( list(verts.keys()) )

adjLists = {}

for ind,key in enumerate(verts):
    
    adjLists.update({key : [i for i in range(1, N+1) if i != key]})
#end


edges = []
edgesStrengths = {}

for i in listVert:
    for j in listVert:
        
        if ( i != j and ( (i,j) not in edges and (j,i) not in edges ) ):
                
            newEdge = (i,j)
            edges.append(newEdge)
            
            x_coord = (verts[newEdge[0]][0] , verts[newEdge[1]][0])
            y_coord = (verts[newEdge[0]][1] , verts[newEdge[1]][1])
            r = np.array([x_coord[0] - x_coord[1], y_coord[0] - y_coord[1]])
            strength = 1./np.linalg.norm(r)
            edgesStrengths.update({newEdge : strength})
        #end
    #end
#end


#for edge in edges:
#    
#    x_coord = (verts[edge[0]][0] , verts[edge[1]][0])
#    y_coord = (verts[edge[0]][1] , verts[edge[1]][1])
#    r = np.array([x_coord[0] - x_coord[1], y_coord[0] - y_coord[1]])
#    strength = 1./np.linalg.norm(r)
#    edgesStrengths.update({edge : strength})
##end

G.add_edges_from(edges)


#%% Melting

maxStrength = max(edgesStrengths.values())
minStrength = min(edgesStrengths.values())

temperatures = np.linspace(1/maxStrength, 1/minStrength, 10)


for T in temperatures:
    
    for edge in edges:
        
        if (edgesStrengths[edge] < 1/T):
            
            edges.remove(edge)
        #end
    #end
#end


plotGraph(vertPerClass, verts, edges, edgesStrengths)

graph = [verts,edges]

fileID = open(r'ClustersExp/clusters.pkl','wb')
pickle.dump(graph,fileID)
fileID.close()


#%% salvare su disco struttura dati opportuna.
# nota: all'euristica fisica del problema ci penso in un altro momento

# https://ermongroup.github.io/cs228-notes/inference/sampling/
# potrebbe essere utile per sampling.

fileID = open(r'C:\Users\Matteo\Desktop\MasterThesis\newThread\alltogether_streams\ClustersExp\clusters.pkl','rb')
#fileID = open(r'ClustersExp\clusters.pkl', 'rb')
graph = pickle.load(fileID)
fileID.close()

verts = graph[0]
edges = graph[1]

edgesDF = pd.DataFrame(edges, columns = ['source', 'target'])
vertsDF = pd.DataFrame.from_dict(verts, orient = 'index', 
                                 columns = ['x1','x2','Class'])
Nc = max(vertsDF['Class'])

#verts1 = vertsDF[vertsDF['Class'] == 1]
#verts1 = verts1.index.tolist()
#edges1 = []
#tmp = []
#msk = (edgesDF['source'] == k+1) | (edgesDF['target'] == k+1)
vertsDF['Topological'] = pd.Series(np.ones(vertsDF.shape[0]),
                                   index = vertsDF.index)
#%%
#
#for node in verts1:
#    
#    tmp = edgesDF[(edgesDF['source']==node) | 
#            (edgesDF['target']==node)]
#    print(tmp)
##    edges1.append([tmp[0], tmp[1]])
#    for i in range(tmp.shape[0]):
#        edge_add = [tmp.iloc[i]['source'], tmp.iloc[i]['target']]
#        if edge_add not in edges1:
#            edges1.append( edge_add )
            
for k in range(Nc):
    
    verts_k = vertsDF[vertsDF['Class'] == k+1]
    verts_k = verts_k.index.tolist()
    edges_k = []
    tmp = []
    
    for node in verts_k:
        
        msk = (edgesDF['source'] == node) | (edgesDF['target'] == node)
        tmp = edgesDF[msk]
        for i in range(tmp.shape[0]):
            edge_add = [tmp.iloc[i]['source'], tmp.iloc[i]['target']]
            
            if edge_add not in edges_k:
                edges_k.append(edge_add)
            #end
        #end        
    #end
    for node in verts_k:
        
        lefts = 0; rights = 0
        for edge in edges_k:
            
            if (edge[0] == node):
                lefts += 1
            elif (edge[1] == node):
                rights += 1
                vertsDF.at[node,'Topological'] = int(
                        vertsDF.at[edge[0],'Topological'] + 1)
            #end
        
        #end
    #end
#end
    










        