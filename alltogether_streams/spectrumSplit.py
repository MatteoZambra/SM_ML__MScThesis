# -*- coding: utf-8 -*-
"""
Perform KDE on a given sample (it the actual program scope such sample
is either the weights list or the biases list), then compute the histogram
bins in a way that one bin comprehends all of the data samples falling in
the region for which the data have an associated (estimated) pdf greater
that a given factor (free parameter). It returns the bins edges. 
Subsequently it is in order to use these bins edges to assign each connection
weight and bias (of the neural system) a belonging category, in that
motif mining software can deal with "colors". 


*** IMPROVEMENT: 
    
    instead of chosing an arbitrary factor in order to select which data 
    points belong to the central bin, it could be a neater choice to 
    select a smart quantile, so that there is a statistical significance



"""

from math import floor, ceil
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("bmh")




def smartBins(d, bwMethod, factor, colors):

	"""
	simple logic: loop over all the samples (weights or biases),
	if the next in the serie features a y-value (that comes from
	the KDE of the samples) greater that a given threshold 
	
	AND 
	
	the previous instead has a y-value smaller that such threshold,
	then the current point separates two regions:
	By thus doing, we require that all of the samples that feature a
	value of the estimated density greater that a certain value, are 
	forced to belong to the same bin. It is done because the number
	of _colors_ that FanMod supports in limited, and a neat choice of
	the division is crucial. 
	The threshold value is motivated by the density plots of the weights
	and biases, see main text
	"""


    #d = np.random.normal(0,1,100)
    k = gaussian_kde(d, bw_method = bwMethod)
    
    x = np.sort(d)
    y = np.reshape(k(x).T, d.size)
    
    plt.plot(x,y)
    plt.xlabel("Sorted values")
    plt.ylabel("Estimated density")
    plt.title("Kernel density estimate")
    plt.show()
    
    bins = []
    thre = factor*max(y)
    
    
    for i in range(1,d.size):
        if ( ((y[i] >= thre) and (y[i-1] < thre)) or \
             ((y[i] < thre) and (y[i-1] >= thre))   ):
            
            bins.append(x[i])
        #end
    #end
    
    ticks = ceil(colors/2.)
    

#                        stugazz' di min(x) + ... è stra creep
#                              |
#                              |
#                              v
    first  = np.linspace(min(x)+0.01*min(x), bins[0], ticks)
    second = np.linspace(bins[1], max(x)+0.01*max(x), ticks)
    binsEdges = np.hstack((first,second))
    
    if (colors % 2 == 0):
        for k in range(2*ticks):
            
            if (k == floor(colors/2.)):
                binsEdges = binsEdges.tolist()
                newTick = (binsEdges[k-1] + binsEdges[k])/2.
                binsEdges.insert(k,newTick)
                break
            #end
        #end
    #end
    binsEdges = np.asarray(sorted(binsEdges))
    
    hist,bins_edges = np.histogram(d, bins = binsEdges, density = False)
    plt.hist(d, bins = bins_edges)
    plt.xticks(bins_edges, rotation = 45)
    plt.xlabel("Sorted values")
    plt.ylabel("Frequency")
    plt.title("Frequency plot with smart bins")
    plt.show()
    
    return hist,binsEdges
#end

        
def CategoriseWeightsBiases(df, binsEdges):

    
    df["cats"] = pd.Series(np.zeros(df.shape[0]),
                           index = df.index)
    for i in range(df.shape[0]):
        par = df.iloc[i]["param"]
        
        for cate in range(1, len(binsEdges)):
            
            if (par >= binsEdges[cate-1] and par < binsEdges[cate]):
                
                df.at[i+1,"cats"] = cate
            #end
        #end
    #end
    
    return df
#end
        

		
# test shell
#
#d = np.random.normal(0,0.1,100)
#count,bins = smartBins(d, 
#                       bwMethod = 'silverman', 
#                       factor = 0.5, 
#                       colors = 7)
#        
    
        