

"""
Motif simply intended as a group of interconnected units in a network,
needs to be computationally defined with a dedicated data type. 
The FANMOD program returns a comprehensive array of motifs features, including
the adjacency matrix, the identifier, the frequency, the z-score and so on.

Other than that, when it comes to deling with colored networks, the heterogeneity 
of the connections strengths --translated by the preprocessing utilities to categories-- 
is thought to be well encoded by the entropy intended as information theoretic, as 
done by Choodbar et al. (2012) ``Motif mining in weighted networks''
https://www.dcc.fc.up.pt/~pribeiro/pubs/pdf/choobdar-damnet2012.pdf

This metric is not being used in the analyses. It could however be useful in
further inspections
"""


import numpy as np
#import os
import csv
from math import log
from itertools import islice



class MotifObj:
    def __init__(self, motifID, subgrID, AdjMat, freq,
                 mean_freq, std_dev, Zscore, pValue, entropy):
        self.motifID = motifID
        self.subgrID = subgrID
        self.AdjMat  = AdjMat
        self.frequency = freq
        self.mean_freq = mean_freq
        self.std_dev = std_dev
        self.Zscore = Zscore
        self.pValue = pValue
        self.entropy = entropy
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
    
#def getpath(weighted, dataset_id, motif_size, initialisation):
#    
#    path = os.getcwd()
#    path += r'\FanMod__results\\' + initialisation + '\\' + \
#            's' + str(motif_size) + '\\' +  \
#            str(dataset_id)
#    if (weighted == "u"):
#        path += r'\unweighted\\'
#    else:
#        path += r'\weighted\\'
#    #end
#    path_file = path + str(dataset_id) + '_s' +  \
#            str(motif_size) + '_' + str(weighted) + '_out.csv'
#    return path_file
##end

def motifs_load(path_file,offset,size):

    """
    Read the output file of the FANMOD motifs mining tool
    
    Input:
        ~ path_file         string, where the FANMOD output motifs log file is stored
        ~ offset            integer. Depending on whether the output refers to a weighted
                            or unweighted analysis, the format of the log file changes 
                            of those two rows. Specifying this detail allows to use the
                            function regardless.
                            Offset = 0 :: unweighted analysis log file
                            Offset = 2 :: (skip two rows) weighted analysis log file
        ~ size              integer. The size of the motifs accounted for. 
                            In out analyses, only 4-nodes and 6-nodes motifs are inspected
    
    Returns:
        ~ motifs            dictionary in which the keys are the identifiers of the topological
                            isomorphic groups and the associated values are lists of instances of
                            such a topological group. Each of these instances is a motifsreader.MotifObj
                            instance, those defined above
    """
    
    header = []
    motifs = {}
#    lineNumber = 0
    AdjMat = np.zeros((size,size))
    i = 0
    count = 1
    
    with open(path_file,'r') as f_file:
        
        reader = csv.reader(f_file, delimiter = ",")
        
        for row in islice(reader,28+offset,29+offset):
            for name in row:
                header.append(name)
            #end
            header[2] += '[%]'
            header[3] += '[%]'
        #end
        
        for row in islice(reader,1,None):
#            print(row)
            
            if (len(row) == 0):
                AdjMat = np.zeros((size,size))
                i = 0
            else:
                if (len(row) == len(header)):
                    
                    motifID = int(row[0])
                    for j in range(len(row[1])):
                        add = float(row[1][j])
                        AdjMat[i,j] = add
                    #end
                    i += 1
                    freq = float(row[2][:-1])
                    mean_freq = float(row[3][:-1])
                    std_dev = float(row[4])
                    if (row[5] == 'undefined'):
#                        Zscore= -999
                        continue
                    else:
                        Zscore = float(row[5])
                    #end
                    pValue = float(row[6])
                else:
                    for j in range(len(row[1])):
                        add = float(row[1][j])
                        AdjMat[i,j] = add
                    #end
                    i += 1
                #end
                
                if (i == 4):
                    
                    entropy = 0.0
                    newMotif = MotifObj(motifID, count,
                                        AdjMat, freq,
                                        mean_freq, std_dev,
                                        Zscore, pValue, entropy)
#                    if (offset == 2): newMotif.EntropyCompute()
                    
                    if (motifID in motifs):
                        motifs[motifID].append(newMotif)
                    else:
                        motifs.update( {motifID : [newMotif]} )
                    #end
                    count += 1
                #end
            #end
        #end
    #end
    
    for ID in motifs:
        motifs[ID] = [motif for motif in motifs[ID]  \
                      if (motif.frequency >= 0.0  and \
                          motif.mean_freq >= 0.0) and 
                          motif.pValue < 0.05]
        if (offset == 2):
            motifs[ID] = [motif for motif in motifs[ID]  \
                          if (4. not in motif.AdjMat.flatten().tolist())]
    #end
    
    return motifs
#end
