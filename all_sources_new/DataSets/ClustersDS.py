
"""
    INDEPENDENT CLUSTERS DATA SET
    
    graphDataStructure is a class that wraps some utilities
    crafted `ad hoc': the knowledge of the actual topological 
    and architectural constraints imposed by the neural 
    network `per se' drastically simplifies most of the work
    that one should perform in a sharply more Computer Science
    fashion, graph algorithms and so forth.
    
    The idea is simple:
        . generate a cloud of 31 points
        . grouped in 4 clusters;
        . link each point with any other;
        . in a simulated annealing fashion,
          remove the edges between the vertices
          that are more "dissimilar".
          Here dissimilarity may simply be the 
          distance between these. 
          Melting schedule is crucial.
        . If the melting process was nealy 
          designed, one should end up with
          four independent and NOT fully 
          connected graphs.
        . In a PGMs fashion, perform ancestral 
          sampling (See Chap.16 of the book "Deep
          Learning" by I.Goodfellow, Y.Bengio, 
          A.Courville, 2016) only for the class
          chosen. In this way, only the random variables
          of that class are different, and classification
          is almost straightforward
         
    Once that done, labelling in done by means of onehots.
    
    NOTE: this script only deal with the GRAPH DATA SET, which 
    has nothing to do with the graph that will subsequently come
    from the neural network topology.
    
    The means and the matrix covariances of the (four) points 
    clouds to generate are fixed a priori, and given to the method
    which serves to generate such points.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pandas as pd
import pickle

class graphDataStructure:
    
    def __init__(self, Nclasses):
        
        """
        For consistency with the other data set, it is 
        thought wiser to retain the input and output dimensions
        That is: 31 vertices in the global graph,
                 4  belongingness classes.
        """
        
        self.verts = {}
        self.edges = []
        self.edgesStrengths = {}
#        self.vertsPerClass = np.random.randint(10,13, size = Nclasses) # or,
        self.vertsPerClass = [8, 7, 9, 7]
    #end
    
    def scatterPlot(self):
        
        vertsDf = pd.DataFrame.from_dict(self.verts, 
                               orient = 'index', 
                               columns = ['x1', 'x2', 'class'])

        plt.figure(figsize=(8,6))
        
        for k in range(len(self.vertsPerClass)):
    
            plt.scatter(vertsDf[vertsDf['class']==k+1]['x1'],
                        vertsDf[vertsDf['class']==k+1]['x2'],
                        s = 80)
                        
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Generated points scatterplot")
        #end
        plt.show()
    #end
    
    def plotGraph(self):
    
        vertsDf = pd.DataFrame.from_dict(self.verts, 
                                         orient = 'index', 
                                         columns = ['x1', 'x2', 'class'])

        fig = plt.figure(figsize=(8,6))

        for k in range(len(self.vertsPerClass)):
    
            plt.scatter(vertsDf[vertsDf['class']==k+1]['x1'],
                        vertsDf[vertsDf['class']==k+1]['x2'],
                        s = 80)
                        
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Generated points graph")
        #end
    
        for edge in self.edges:
        
            x_coord = (self.verts[edge[0]][0] , self.verts[edge[1]][0])
            y_coord = (self.verts[edge[0]][1] , self.verts[edge[1]][1])
        
            strength = self.edgesStrengths[edge]
            line = lines.Line2D(x_coord, y_coord, linewidth = strength * 1.5,
                                color = 'k')
            fig.gca().add_line(line)
                
        fig.show()
    #enddef
    
    
    def VerticesGeneration(self, moments):
    
        """
        moments: list of lists with means 
        and covariances 
        A dictionary is thought neat for the ease
        it yields to query for vertices.
        Note that the values of the dict are:
            . coordinates and
            . class of belongingness.
            
        This latter is simply the index of the 
        loop round, loop over the number of data
        groups indeed.
        """
        
        verts = {}
        cnt = 0
        for k in range(len(self.vertsPerClass)):
            
            for i in range(self.vertsPerClass[k]):
                
                x = np.random.multivariate_normal(moments[k][0],
                                                  moments[k][1],
                                                  (1,)).reshape((2,))
                
                verts.update({cnt+1 : [x[0], x[1] , k+1]})
                cnt += 1
            #end
        #end
        
        self.verts = verts
        self.scatterPlot()
        return verts
    #end
    
    def EdgesGeneration(self):
        
        edges = []
        edgesStrengths = {}
        verts = self.verts
        
        listVert = list(verts.keys())
        
        for i in listVert:
            for j in listVert:
                
                # note that for the sake of indirectedness, one must account
                # only the edges (i,j) with i > j in the loops progression
                if (i != j and ((i,j) not in edges and (j,i) not in edges)):
                        
                    newEdge = (i,j)
                    edges.append(newEdge)
                    
                    x_coord = (verts[newEdge[0]][0] , verts[newEdge[1]][0])
                    y_coord = (verts[newEdge[0]][1] , verts[newEdge[1]][1])
                    
                    r = np.array([x_coord[0] - x_coord[1], 
                                  y_coord[0] - y_coord[1]])
                                  
                    
                    # strength = 1 / (norm of the distance btwn two verts)
                    strength = 1./np.linalg.norm(r)
                    edgesStrengths.update({newEdge : strength})
                #end
            #end
        #end
        
        self.edges = edges
        self.edgesStrengths = edgesStrengths
        self.plotGraph()
    #end
    
    def SimulatedMelting(self,meltSched):
        
        # melting schedule setup
        maxStrength = max(self.edgesStrengths.values())
        minStrength = min(self.edgesStrengths.values())

        temperatures = np.linspace(1/maxStrength, 1/minStrength, meltSched)


        # heuristics could be improved
        
        for T in temperatures:
    
            for edge in self.edges:
        
                if (self.edgesStrengths[edge] < 1/T):
            
                    self.edges.remove(edge)
                #end
            #end
        #end
        
        self.plotGraph()
    #end
    
    def TopologicalOrdering(self):
    
        """
        Topological ordering is an integer denoting the 
        position of a vertex in the directed graph which
        represents the relationship btwn these RVs
        """
        
        print("Topological Ordering")
        
        # pandas DataFrames come particularly handy for the 
        # query they render possible.
        # the edges data frame contains 
        #    . source : vertx from which edge starts
        #    . target : vertx where this edge ends up
        # vertices data frame features
        #    . coordinates
        #    . class
        # Two columns are added to this latter: 
        #    . topological : in which one shall write the
        #                    topological order, once found
        #    . ancestry    : list of all the vertices from 
        #                    which edges start to indirectly
        #                    end up in that vertex
        
        edgesDF = pd.DataFrame(self.edges, columns = ['source', 'target'])
        vertsDF = pd.DataFrame.from_dict(self.verts,
                                         orient = 'index',
                                         columns = ['x1','x2','class'])
        Nc = max(vertsDF['class'])
        vertsDF['topological'] = pd.Series(np.ones(vertsDF.shape[0]),
                                             index = vertsDF.index)
        vertsDF['ancestry'] = pd.Series([[] for i in range(vertsDF.shape[0])],
                                        index = vertsDF.index)
        
        for k in range(Nc):
            
            # for each class some local variables are thought clearer:
            # for vertices and edges of that single class
            
            verts_k = vertsDF[vertsDF['class'] == k+1]
            verts_k = verts_k.index.tolist()
            edges_k = []
            tmp = []
            
            for node in verts_k:
                
                # collecting the edges which have as ends the nodes under
                # consideration in the loop over nodes of each group
                
                msk = (edgesDF['source']==node) | (edgesDF['target']==node)
                tmp = edgesDF[msk]
                for i in range(tmp.shape[0]):
                    edge_add = [tmp.iloc[i]['source'], tmp.iloc[i]['target']]
                    if edge_add not in edges_k:
                        edges_k.append(edge_add)
                    #endif
                #enddo
            #enddo
            
            # new cycle again over nodes because HERE the list of 
            # the edges of this class is complete. Otherwise it would 
            # be worked the incomplete list.
            
            for node in verts_k:
                
                tmp = []
                for edge in edges_k:
                    
                    if (edge[1] == node):
                        tmp.append(edge[0])
                    #endif
                #end
                vertsDF.at[node,'ancestry'] = tmp
                
                for t in tmp:
                    for _t in vertsDF.at[t,'ancestry']:
                        if (_t not in vertsDF.at[node,'ancestry']):
                            vertsDF.at[node,'ancestry'].append(_t)
                        #endif
                    #enddo
                #enddo
                                                    
                topos = []
                if (len(tmp) != 0):
                    for t in range(len(tmp)):
                        topos.append(vertsDF.at[tmp[t],'topological'])
                    #enddo
                    vertsDF.at[node,'topological'] = max(topos) + 1
                #endif
            #enddo
        
        #enddo
        return vertsDF, edgesDF
    #enddef
    
    def DataSetGeneration(self, vertsDF, edgesDF, M):
        
        print("DataSet generation")
        
        N = vertsDF.shape[0]
        X = np.zeros((M,N))
        Nc = max(vertsDF['class'])
        Y = np.zeros((M,Nc))
        
        for i in range(M):
            
            X[i,:] = -1
            
            label = np.random.randint(1, Nc+1)
            verts_ = vertsDF[vertsDF['class'] == label].sort_values(
                                                        'topological')
            
            for v in verts_.index.tolist():
                
                """Like a Pro: ancestral sampling.
                    poi magari vedi se riesci a elaborare un modo per
                    fare sampling dalla condizionale p(x2 | x1)
                """
                
                if (len(verts_.at[v,'ancestry']) == 0):
                    X[i, v-1] = 1
                else:
                    X[i, v-1] = len(verts_.at[v,'ancestry'])
                #endif
            #enddo
            
            for j in range(Nc):
                if (j+1 == label):
                    Y[i,j] = 1
                #endif
            #enddo
            
        #enddo
        
        DataSet = [X,Y]
        fileID = open(r'Clusters_DS_list.pkl', 'wb')
        pickle.dump(DataSet,fileID)
        fileID.close()
        
        return DataSet
    #enddef
                
                
    
#endclass
    
    
# DEPLOYMENT    

varFA = 8

moments = [
            [ [1,1],   [[varFA, 0.0],[0.0, varFA]] ],
            [ [1,51],  [[varFA, 0.0],[0.0, varFA]] ],
            [ [51,1],  [[varFA, 0.0],[0.0, varFA]] ],
            [ [51,51], [[varFA, 0.0],[0.0, varFA]] ],
            [ [25,25], [[varFA, 0.0],[0.0, varFA]] ]
        ]

Nc = 4
meltSched = 15
M = 2000

graph = graphDataStructure(Nc)

graph.VerticesGeneration(moments)
graph.EdgesGeneration()
graph.SimulatedMelting(meltSched)

vertsDF, edgesDF = graph.TopologicalOrdering()
DataSet = graph.DataSetGeneration(vertsDF, edgesDF, M)
X = DataSet[0]
Y = DataSet[1]


fig = plt.figure(figsize = (10,10))
X_df = pd.DataFrame(X)
covMat = X_df.corr()
plt.matshow(covMat)
plt.colorbar()
plt.show()


