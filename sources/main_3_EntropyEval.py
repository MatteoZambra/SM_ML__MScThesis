
"""
Load edited FANMOD results and store all the subgraphs of a given ID in 
a list. Of each one, subsequently, one can evaluate entropy, to check
which ones are really significative. The more the homogeneity of a motif,
the more it could be significant, in that it is more likely it to encode 
some particular feature of the domain.
"""

import os
import numpy as np
from math import log


class MotifObj:
    def __init__(self, motifID, subgrID, AdjMat):
        self.motifID = motifID
        self.subgrID = subgrID
        self.AdjMat  = AdjMat
        self.entropy = 0.0
    #end
    
    def EntropyCompute(self):
        
        flatmat = self.AdjMat.flatten()
        flatmat = np.asarray(flatmat, dtype=int)
        nonzeros = np.count_nonzero(flatmat)
        frequencies = np.bincount(flatmat)
        frequencies = (frequencies / nonzeros).tolist()
        del frequencies[0]
        frequencies[:] = (val for val in frequencies if val != 0.0)
        
        entropy = 0.0
        for freq in frequencies:
            if (freq != 0):
                entropy -= freq*log(freq)
            #end
        #end
        
        self.entropy = entropy
    #end
#end




path = os.getcwd()
path += r"\FanMod__results\_s_4\_out_init.txt"

coloredMotifs = {}
lineNumber = 0
AdjMat = np.zeros((4,4))
i = 0
count = 1


with open(path) as f:
    
    for line in f:
        
        line = line[:9]
    
        if (line == "\n"):
            print(lineNumber, "empty line: nothing")
            AdjMat = np.zeros((4,4))
            i = 0
        else:
            if (line[:4] != "    "):
                motifID = int(line[:4])
            #end
            j = 0
            tmp_row = line[5:9]
            for elem in tmp_row:
                AdjMat[i,j] = int(elem)
                j += 1
            #end
            i += 1
        #end
        
        if (i == 4):
            
#            if (4.0 not in AdjMat.flatten().tolist()):
                
            newMotif = MotifObj(motifID,count,AdjMat)
            newMotif.EntropyCompute()
            
            if (motifID in coloredMotifs):
                coloredMotifs[motifID].append(newMotif)
            else:
                coloredMotifs.update( { motifID : [newMotif] } )
            #end
            count += 1
        #end
        #end
        
        lineNumber += 1
    #end
    
#end

#%%

for ID in coloredMotifs:
    
    coloredMotifs[ID] = [motif for motif in coloredMotifs[ID] \
                  if (4. not in motif.AdjMat.flatten().tolist())]
#end

#%%


























