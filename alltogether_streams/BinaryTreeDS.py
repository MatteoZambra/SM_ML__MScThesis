# -*- coding: utf-8 -*-
"""
Binary data set generation
"""

import numpy as np
from numpy import random

from math import floor
import pickle


class BinaryTreeDataSet:

    def __init__(self,Bf,D,M, lev):

        self.Bf = Bf
                              # Bf : branching factor
                              # D : livelli dell'albero
        self.N = Bf**(D) - 1  # tutti i nodi
        self.n = Bf**(D-1) -1 # tutti i nodi NON foglie
        self.P = Bf**(D-1)    # tutti i nodi foglie
        self.M = M            # quanti pattern (data items)
        
        self.lev = lev


        print('\nTree characts:\n')
        print('   Nodes (feats) : N = ',self.N)
        print('Nodes NOT leaves : n = ',self.n)
        print('    Nodes leaves : P = ',self.P)
        print('        Patterns : M = ',self.M)
    #end

    def patternGenerator(self):

        N = self.N
        n = self.n

        tree = np.zeros(N)
        outcomes = [-1,1]
        e = 0.3
        tree[0] = outcomes[random.randint(0,2)]

        if (tree[0] == 1):
            tree[1] = 1
            tree[2] = -1
        else:
            tree[1] = -1
            tree[2] = 1
        #end

        for k in range(1,n):
            if (tree[k] == 1):
                p = random.rand()
                if (p > e):
                    tree[2*k + 1] = tree[k]
                    tree[2*k + 2] = (-1.) * tree[k]
                else:
                    tree[2*k + 1] = (-1.) * tree[k]
                    tree[2*k + 2] = tree[k]
                #endif
            else:
                tree[2*k + 1] = -1
                tree[2*k + 2] = -1
            #endif
        #enddo

        return tree
    #end


    def dataSetGenerator(self):

        #print('in dataset generator')
        N = self.N; M = self.M
        X = np.zeros((M,N))

        for m in range(M):
            X[m,:] = self.patternGenerator()
        #end

        return X
    #end

    def labelsGenerator(self, X, lev):

        Y = np.eye(X.shape[0])
        print("DataSet: {} data entry having {} features each\n".format(X.shape[0], X.shape[1]))

        def view1D(a): # a is array
            a = np.ascontiguousarray(a)
            void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            return a.view(void_dt).ravel()
        #enddef

        # Get 1D view
        # lowerBound = Bf**lev - 1
        uppBound = self.Bf**(lev + 1) - 2

        X_ = X[:,  : uppBound+1 ]
        a1D = view1D(X_)

        # Perform broadcasting to get outer equality match
        mask = a1D[:,None]==a1D

        # Get indices of pairwise matches
        n = len(mask)
        mask[np.tri(n, dtype=bool)] = 0
        idx = np.argwhere(mask)

        # Run loop to assign equal rows in Y
        for (i,j) in zip(idx[:,0],idx[:,1]):
            Y[j] = Y[i]
        #enddo

        check = np.zeros(Y.shape[0])
        listZeroCol = []
        listNoZeroCol = []
        for i in range(Y.shape[1]):
            if (np.all( (Y[:,i] == check), axis = 0)):
                #print(i)
                listZeroCol.append(i)
            #endif
        #enddo

        listNoZeroCol = [i for i in range(Y.shape[1]) if i not in listZeroCol]
        Y = Y[:,listNoZeroCol]

        return Y

    #enddef

    def dataSetNoiser(self, X, flipFraction = 5.):

        MaxFlip = floor(X.shape[0]*X.shape[1]/flipFraction)
        X_ = X

        if (MaxFlip < 5):
            print("non flippi niente eh")
        #end

        for k in range(MaxFlip):
            rowIdx = np.random.randint(0,X.shape[0])
            colIdx = np.random.randint(0,X.shape[1])
            #print(rowIdx,colIdx)
            X_[rowIdx,colIdx] = (-1.) * X[rowIdx,colIdx]
        #end

        return X_

    #enddef
    
    
    def DataSet_creator(self, flipFraction):
        
#        treeData = BinaryTreeDataSet(Bf,D,M)
        X = self.dataSetGenerator()
        Y = self.labelsGenerator(X,self.lev)
        print("\n",type(X),"\n",X)
        print("\n",type(Y),"\n",Y)
        
        print(Y.shape," quindi il numero di classi diverse è {}".format(Y.shape[1]))
        
        DataSet = [X,Y]
        fileID = open(r'DataSets_levels/DataSet_list_lev4.pkl', 'wb')
        pickle.dump(DataSet, fileID)
        fileID.close()
        
#        X = self.dataSetNoiser(X, flipFraction)
#        DataSet = [X,Y]
#        fileID = open(r'DataSets/DataSet_list_noise.pkl', 'wb')
#        pickle.dump(DataSet, fileID)
#        fileID.close()
#endclass

# ------------------ Deployment

Bf = 2
D = 5
M = 2000
lev = 4
flipFraction = 5.

bt = BinaryTreeDataSet(Bf,D,M,lev)
bt.DataSet_creator(flipFraction)


