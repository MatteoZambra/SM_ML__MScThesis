
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotCovMat(X,path):
    X_df = pd.DataFrame(X)
    covMat = X_df.cov()
    plt.figure(figsize=(4,6))
    ax = plt.gca()
    im = ax.matshow(covMat)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im,cax=cax)
    plt.savefig(path + 'cov.png')
#    plt.show()
#enddef


def jointPlotter(numLayers, path, params_pre, params_post, plot_hist):
    
    params_pre = np.asarray(params_pre)
    params_post = np.asarray(params_post)
    
    if (plot_hist == True):
        
        for i in range(numLayers):
    
            j = 2*i
            # as before: weights on the even slots
            plt.figure(figsize=(15,6))
            ax = plt.subplot(1,2,1)
            w1 = params_pre[j].flatten()
            w2 = params_post[j].flatten()
            
            interMax = np.max(w2)
            interMin = np.min(w2)
            h = 2 * iqr(w2) * w2.shape[0]**(-1/3.)
            nBins = (interMax - interMin)/h
            bins = np.linspace(interMin, interMax, nBins)
            
            plt.hist(w1, bins, alpha = 0.5, 
                     label = "Weights before training")
            plt.hist(w2, bins, alpha = 0.5, 
                     label = "Weights after training")
            title = "Weights frequencies of layer "+str(i+1)
            ax.set_xticks(bins)
            plt.xticks(bins, rotation = 45)
            plt.title(title,fontsize=20)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.legend()

            j = 2*i + 1
            # and biases on odd slots
            plt.subplot(1,2,2)
            b1 = params_pre[j].flatten()
            b2 = params_post[j].flatten()
            
            interMax = np.max(b2)
            interMin = np.min(b2)
            
            h = 0.5 * iqr(b2) * b2.shape[0]**(-1/3.)
            nBins = (interMax - interMin)/h
            
            bins = np.linspace(interMin, interMax, nBins)
            
            plt.hist(b1, bins, alpha = 0.5, 
                     label = "Biases before training")
            plt.hist(b2, bins, alpha = 0.5, 
                     label = "Biases after training")
            title = "Biases frequencies of layer "+str(i+1)
            plt.xticks(bins, rotation = 45)
            plt.title(title,fontsize=20)
            plt.xlabel("Bias")
            plt.ylabel("Frequency")
            plt.legend()
            
            plt.savefig(path + 'hist' + r'_L' + str(i+1) + '.png',
                        dpi=300, bbox_inches = "tight")
        #end
        plt.show()

    else:
            
        for i in range(numLayers):
    
            j = 2*i
            plt.figure(figsize=(15,6))
            plt.subplot(1,2,1)
            w1 = params_pre[j].flatten()
            bandwidth = 1.06 * w1.std() * w1.size ** (-1 / 5.)
            sns.kdeplot(w1, bw=bandwidth, shade=True, 
                        label = "Weights before train")

            w2 = params_post[j].flatten()
            bandwidth = 1.06 * w2.std() * w2.size ** (-1 / 5.)
            sns.kdeplot(w2, bw=bandwidth, shade=True, 
                        label = "Weights after train")
            title = "Weights distribution of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Weight")
            plt.ylabel("Pdf")
            plt.legend()

            j = 2*i + 1
            plt.subplot(1,2,2)
            b1 = params_pre[j].flatten()
            bandwidth = 1.06 * b1.std() * b1.size ** (-1 / 5.)
            sns.kdeplot(b1, bw=bandwidth, shade=True, 
                        label = "Biases before train")

            b2 = params_post[j].flatten()
            bandwidth = 1.06 * b2.std() * b2.size ** (-1 / 5.)
            sns.kdeplot(b2, bw=bandwidth, shade=True, 
                        label = "Biases after train")
            plt.legend()
            title = "Bias distribution of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Bias")
            plt.ylabel("Pdf")
            
            plt.savefig(path + 'kde' + r'_L' + str(i+1) + '.png',
                        dpi=300, bbox_inches = "tight")
        #end
        plt.show()
#enddef
            
