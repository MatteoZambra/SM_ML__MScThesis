
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
