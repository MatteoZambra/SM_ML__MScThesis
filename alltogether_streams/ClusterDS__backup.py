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

def plotGraph(vertsPerClass, verts, edges, edgesStrengths):
    
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
            [ [26,26], [[10, 0.0],[0.0, 10]] ]
        ]

vertPerClass = np.random.randint(8,20, size=5)
#vertPerClass = [5, 5, 5, 5]
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

#%% Scatterplot
#
#vertsDf = pd.DataFrame.from_dict(verts, 
#                               orient = 'index', 
#                               columns = ['x1', 'x2', 'class'])
#
#plt.figure(figsize=(10,8))
#
#for k in range(len(vertPerClass)):
#    
#    plt.scatter(vertsDf[vertsDf['class']==k+1]['x1'],
#                vertsDf[vertsDf['class']==k+1]['x2'],
#                s = 80)
##    plt.xlabel(r"$x_{1}$", fontsize = 15)
##    plt.ylabel(r"$x_{2}$", fontsize = 15)
#    plt.xlabel("x1")
#    plt.ylabel("x2")
#    plt.title("Generated points scatterplot")
##end
    
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
    
    
#%%

edges = []
#edges = [ (i,j) for j in listVert for i in listVert 
#         if ((i,j) not in edges  or (j,i) not in edges and i != j) ]

for i in listVert:
    for j in listVert:
        
        if ( i != j and ( (i,j) not in edges and (j,i) not in edges ) ):
            
            edges.append((i,j))
        #end
    #end
#end
    
edgesStrengths = {}
    
for edge in edges:
    
    x_coord = (verts[edge[0]][0] , verts[edge[1]][0])
    y_coord = (verts[edge[0]][1] , verts[edge[1]][1])
    r = np.array([x_coord[0] - x_coord[1], y_coord[0] - y_coord[1]])
    strength = 1./np.linalg.norm(r)
    edgesStrengths.update({edge : strength})
#    
#    line = lines.Line2D(x_coord, y_coord, linewidth = strength * 1.5,
#                        color = 'k')
#    plt.gca().add_line(line)

#plt.show()
    
#plotGraph(vertPerClass, verts, edges, edgesStrengths)

G.add_edges_from(edges)


#%% Melting

maxStrength = max(edgesStrengths.values())
minStrength = min(edgesStrengths.values())

temperatures = np.linspace(1/maxStrength, 1/minStrength, 20)


for T in temperatures:
    
    for edge in edges:
        
        if (edgesStrengths[edge] < 1/T):
            
            edges.remove(edge)
        #end
    #end
#end
    
    
plt.figure(figsize=(10,8))

#for k in range(len(vertPerClass)):
#    
#    plt.scatter(vertsDf[vertsDf['class']==k+1]['x1'],
#                vertsDf[vertsDf['class']==k+1]['x2'],
#                s = 80)
##    plt.xlabel(r"$x_{1}$", fontsize = 15)
##    plt.ylabel(r"$x_{2}$", fontsize = 15)
#    plt.xlabel("x1")
#    plt.ylabel("x2")
#    plt.title("Generated points scatterplot")
##end    
#
#for edge in edges:
#    
#    x_coord = (verts[edge[0]][0] , verts[edge[1]][0])
#    y_coord = (verts[edge[0]][1] , verts[edge[1]][1])
#    
#    line = lines.Line2D(x_coord, y_coord, 
#                        linewidth = 0.5,
#                        color = 'k')
#    plt.gca().add_line(line)
##end

plotGraph(vertPerClass, verts, edges, edgesStrengths)
















        