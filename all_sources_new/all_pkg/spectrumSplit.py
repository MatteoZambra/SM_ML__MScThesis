

"""
Perform KDE on a given sample (it the actual program scope such sample
is either the weights list or the biases list), then compute the histogram
bins in a way that one bin comprehends all of the data samples falling in
the region for which the data have an associated (estimated) pdf greater
that a given factor (free parameter). It returns the bins edges. 
Subsequently it is in order to use these bins edges to assign each connection
weight and bias (of the neural system) a belonging category, in that
motif mining software can deal with "colors". 


Instead of KDE for each model, the whole weights population is 
gathered in one array and a gaussian fit is made on that values.
See the ``preprocess_kernel'' module

At state, only the spectrum_split_plot and CategoriseWeightsBiases 
functions are used
"""

from math import floor, ceil
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
plt.style.use("seaborn-ticks")




def smartBins(d, path, plot, init_scheme, factor, bwMethod, quantities, colors):

    """
    *******************
    D E P R E C A T E D
    *******************
    
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

    k = gaussian_kde(d, bw_method = bwMethod)
    
    x = np.sort(d)
    y = np.reshape(k(x).T, d.size)
    
    if (plot['preprocess']):
        plt.subplots(figsize=(6,3))
        plt.plot(x,y, color = 'k')
        plt.xlabel("Sorted values")
        plt.ylabel("Estimated density")
        plt.title("Kernel density estimate")
        plt.savefig(path + "_kde_"+str(quantities)+".png",
                    dpi=300, bbox_inches = "tight")
        plt.show()
    #end
    
    
    bins = []
    
    """
    thre = factor*max(y)
    for i in range(1,d.size):
        if ( ((y[i] >= thre) and (y[i-1] < thre)) or \
             ((y[i] < thre) and (y[i-1] >= thre))   ):
            
            bins.append(x[i])
        #end
    #end
    """
    for i in range(1,x.size):
        if (x[i-1] <= -factor <= x[i] or \
            x[i-1] <=  factor <= x[i]):
            
            bins.append(x[i])
        #end
    #end
    
    
    if (colors == 5):
        
        """
        In the case discussed in the main_2, n the case in which one wants to 
        account solely for POSITIVE, NEGATIVE, WEAK but not too weak, it is
        thought better, instead of spacing evenly the extremal bins, to 
        partition in a way to account for the 1/3, so that the space for the 
        strongly positive or negative is increased.
        The rationale of this lays in the results of FANMOD: if evenly spaced
        bins are used, the motifs found are almost only neutral or mixed. 
        That is, the presence of few relevant connections is minimized.
        """
        
        min_ = min(x) + 0.01 * min(x)
        third_negative = 1/3. * (min_ - bins[0]) + bins[0]
        max_ = max(x) + 0.01 * max(x)
        third_positive = 1/3. * (max_ - bins[1]) + bins[1]
        
        if (init_scheme == 'normal'):
            binsEdges = [min_, bins[0], bins[1], max_]
        else:
            binsEdges = [min_, third_negative, bins[0], bins[1], third_positive, max_]
        #end
        
    else:
        ticks = ceil(colors/2.)
        
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
    #end
    
    
    binsEdges = np.asarray(sorted(binsEdges))
    
    hist,bins_edges = np.histogram(d, bins = binsEdges, density = False)
    if (plot['preprocess']):
        plt.subplots(figsize = (6,3))
        ax = plt.gca()
        N,bins,patches = ax.hist(d, bins = bins_edges)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if (init_scheme == 'normal'):
            patches[0].set_facecolor('y')
            patches[1].set_facecolor('lightgray')
            patches[2].set_facecolor('y')
        else:
            patches[0].set_facecolor('r')
            patches[1].set_facecolor('y')
            patches[2].set_facecolor('lightgray')
            patches[3].set_facecolor('y')
            patches[4].set_facecolor('g') 
        #end
        
        plt.xticks(bins_edges, rotation = 45)
        plt.xlabel("Sorted values")
        plt.ylabel("Frequency")
        plt.title("Frequency plot with modified bins")
        plt.savefig(path + "_hist_"+str(quantities)+".png",
                    dpi=300, bbox_inches = "tight")
        plt.show()
    #end
    
    return hist,binsEdges
#end
    

def spectrum_split_plot(weights, path_save_pic, dataset_id,
                   bins_edges):
                   
    """
    Histogram plot.
    The number of categories is set before (binsEdges), and accordingly the 
    bars assume a different color
    
    Input:
        ~ weights               numpy.array that contains the weights values (all the population)
        ~ path_save_pic         string, path to save the figures
        ~ dataset_id            string, identifies the data set
        ~ bins_edges            list of floats, see above
        
    Returns:
        nothing
    """
    
    ds_dict = {'init' : 'Initial',
               'tree' : 'Tree',
               'clus' : 'Clusters',
               'mvg'  : 'Multitask'}
    
#    k = gaussian_kde(weights, bw_method = 'silverman')
#    x = np.sort(weights)
#    y = np.reshape(k(x).T, weights.size)
    
    fig,ax = plt.subplots(figsize = (8,4))
#    ax.plot(weights,y, 'k', lw = 2, alpha = 0.3)
    N,bins,patches = ax.hist(weights, bins = bins_edges, alpha = 0.7)
    
    patches[0].set_facecolor('r')
    patches[1].set_facecolor('y')
    patches[2].set_facecolor('lightgray')
    patches[3].set_facecolor('y')
    patches[4].set_facecolor('g')
    
    legend_elements = [Patch(facecolor = 'r', edgecolor = 'r',
                             alpha = 0.7, label = str(int(N[0]))),
                       Patch(facecolor = 'y', edgecolor = 'y', 
                             alpha = 0.7, label = str(int(N[1] + N[3]))),
                       Patch(facecolor = 'g', edgecolor = 'g', 
                             alpha = 0.7, label = str(int(N[4]))),
                       Patch(facecolor = 'lightgray', edgecolor = 'lightgray', 
                             alpha = 0.7, label = str(int(N[2])))]
    
    ax.legend(handles = legend_elements, loc = 'best', prop = {'size' : 10},
               fancybox = True)
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(bins_edges, rotation = 45)
    plt.xlabel("Sorted values")
    plt.ylabel("Frequency")
    
    plt.title("Frequency plot with modified bins, {} environment".format(ds_dict[dataset_id]))
    plt.savefig(path_save_pic + "_hist_weights.png",
                dpi=300, bbox_inches = "tight")
    plt.show()
#end

        
def CategoriseWeightsBiases(df, bins_edges):

    """
    categories are numerical values, integers.
    Edges and nodes are set to a category according
    to their value.
    The number of categories is the length of the 
    binsEdges array, namely, the 5 slices of the spectrum
    
    Input:
        ~ df            pandas.DataFrame of the edges, see above
        ~ bins_edges    list of floats, see above
        
    Returns:
        ~ df            pandas.DataFrame of edges, in which the categories
                        are set
    """

    df["cats"] = pd.Series(np.zeros(df.shape[0]),
                           index = df.index,
                           dtype = np.int32)
    for i in range(df.shape[0]):
        param = df.iloc[i]["param"]
        
        for cate in range(1, len(bins_edges)):
            
            if (param >= bins_edges[cate-1] and param < bins_edges[cate]):
                
                df.at[i+1,"cats"] = cate
            #end
        #end
    #end
    
    return df
#end
