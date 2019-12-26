"""
    BINARY DATA SET GENERATION
    
    Binary tree data set generator object, which
    
        . is initialized with branching factor, levels
        . performs sampling of features vectors as explained
          in the main text
"""

import numpy as np
from numpy import random

from math import floor
import pickle

import pandas as pd
import matplotlib.pyplot as plt


class BinaryTreeDataSet:

    def __init__(self,Bf,D,M, lev):

        self.Bf = Bf
                              # Bf : branching factor
                              # D : levels. D = 1, ..., L+1
        self.N = Bf**(D) - 1  # all the nodes
        self.n = Bf**(D-1) -1 # all the nodes \ leaves
        self.P = Bf**(D-1)    # all the leaves
        self.M = M            # how many pattern (data items)
        
        self.lev = lev


        print('\nTree characts:\n')
        print('   Nodes (feats) : N = ',self.N)
        print('Nodes NOT leaves : n = ',self.n)
        print('    Nodes leaves : P = ',self.P)
        print('        Patterns : M = ',self.M)
    #end

    def patternGenerator(self):
        """
        This method generates one only pattern, i.e. the collection
        of all the nodes of the tree. Starting from root, whose value
        is the outcome of a random variable x ~ U({-1, +1}), 
        subsequent flips are performed probabilistically, according
        to the threshold value e = .. fixed a priori
        """

        N = self.N
        n = self.n

        tree = np.zeros(N)
        outcomes = [-1,1]
        e = 0.3
        tree[0] = outcomes[random.randint(0,2)]
        
        # only for the chilren of the root node it is
        # used this convention. If the value sampled for
        # the root node is +1, then the left child is 
        # the one to inherit +1 and vice versa
        
        if (tree[0] == 1):
            tree[1] = 1
            tree[2] = -1
        else:
            tree[1] = -1
            tree[2] = 1
        #end
        
        # from the children of the children of root
        # the values are inherited flippingly

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
    
        """
        One single pattern is generated as many times as 
        the user desires.
        """

        #print('in dataset generator')
        N = self.N; M = self.M
        X = np.zeros((M,N))

        for m in range(M):
            X[m,:] = self.patternGenerator()
        #end

        return X
    #end

    def labelsGenerator(self, X, lev):
    
        """
        Primally, the labels matrix is set to be the identity. Then
        the data set is explored to check whether two rows equal. If
        them do, then the label of the second is set equal to the 
        label of the former. Then zero columns are erased. This yields
        a matrix whose rows are one-hot vectors, labelling each data row.
        Most of the following code is taken from
        > https://stackoverflow.com/a/56295460/9136498
        """

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
        print('upper bound = ', uppBound)
        
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
    
        """
        D E P R E C A T E D
        """

        MaxFlip = floor(X.shape[0]*X.shape[1]/flipFraction)
        X_ = X

        if (MaxFlip < 5):
            print("no flip")
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
    
        """
        wrapper method that calls data and labels creation, so that
        it is possible to perform a single call in main, and write 
        DS on disk.
        """
        
#        treeData = BinaryTreeDataSet(Bf,D,M)
        X = self.dataSetGenerator()
        Y = self.labelsGenerator(X,self.lev)
        print("\n",type(X),"\n",X)
        print("\n",type(Y),"\n",Y)
        
        print(Y.shape," : number of classes is {}".format(Y.shape[1]))
        
        # DataSet = [X,Y]
        # fileID = open(r'TreeLev2_DS_list.pkl', 'wb')
        # pickle.dump(DataSet, fileID)
        # fileID.close()
        
#        X = self.dataSetNoiser(X, flipFraction)
#        DataSet = [X,Y]
#        fileID = open(r'DataSets/DataSet_list_noise.pkl', 'wb')
#        pickle.dump(DataSet, fileID)
#        fileID.close()
        return [X,Y]
#endclass

# ------------------ Deployment

Bf = 2
D = 5
M = 2000
lev = 1
flipFraction = 5.

bt = BinaryTreeDataSet(Bf,D,M,lev)
[X,Y] = bt.DataSet_creator(flipFraction)



fig = plt.figure(figsize = (10,10))
X_df = pd.DataFrame(X)
covMat = X_df.cov()
plt.matshow(covMat)
plt.colorbar()
plt.show()